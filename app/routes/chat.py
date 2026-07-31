import asyncio
from datetime import datetime, timezone
import json
import uuid
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
import motor.motor_asyncio
import redis.asyncio as redis

from app.auth import verify_jwt
from app.config import settings
from app.db import get_db, get_redis
from app.models import ChatRequest, serialize_doc, serialize_docs
from app.services.context import append_context, get_context
from app.services.llm import stream_llm_response
from app.services.retrieval import search_all_collections

router = APIRouter(prefix="/chat", tags=["Chatbot"])


def _build_rag_system_prompt(context_chunks: list[dict]) -> str:
    if not context_chunks:
        return (
            "You are a wise and respectful assistant specializing in sacred Hindu scriptures, "
            "especially the Bhagavad Gita. No specific passages were retrieved for this query. "
            "Answer from your general knowledge of the scriptures, clearly noting that no specific "
            "passages were retrieved."
        )

    context_str_list = []
    for idx, chunk in enumerate(context_chunks, 1):
        source = chunk.get("source", "Unknown Source")
        chapter = chunk.get("chapter")
        verse = chunk.get("verse")
        citation = f"{source}, Chapter {chapter}, Verse {verse}"

        lines = [f"[{idx}] {citation}"]

        sanskrit = chunk.get("sanskrit", "").strip()
        if sanskrit:
            lines.append(f"Sanskrit: {sanskrit}")

        transliteration = chunk.get("transliteration", "").strip()
        if transliteration:
            lines.append(f"Transliteration: {transliteration}")

        lines.append(f"Primary meaning: {chunk.get('text', '').strip()}")

        # Include alternate commentator translations if available
        translations = chunk.get("translations", {})
        primary_key = chunk.get("primary_translator", "")
        alternates = [
            f"  [{t['author']}]: {t['translation']}"
            for k, t in translations.items()
            if k != primary_key and t.get("translation")
        ]
        if alternates:
            lines.append("Other commentaries:")
            lines.extend(alternates[:3])  # cap at 3 alternates to avoid prompt bloat

        context_str_list.append("\n".join(lines))

    formatted_context = "\n\n".join(context_str_list)

    return (
        "You are a wise and scholarly assistant specializing in sacred Hindu texts, "
        "especially the Bhagavad Gita.\n\n"
        "Guidelines:\n"
        "1. Ground your answer in the retrieved scripture passages below.\n"
        "2. Always cite the chapter and verse when referencing a passage.\n"
        "3. Where multiple commentaries are provided, synthesize the perspectives to give a rich answer.\n"
        "4. You may draw on your broader knowledge of the Gita to add depth, but clearly distinguish "
        "retrieved passages from your general knowledge.\n"
        "5. Be respectful, clear, and spiritually insightful.\n\n"
        f"--- RETRIEVED SCRIPTURE PASSAGES ---\n{formatted_context}\n--- END PASSAGES ---"
    )


async def _persist_chat_interaction(
    db: motor.motor_asyncio.AsyncIOMotorDatabase,
    redis_client: redis.Redis,
    conversation_id: str,
    user_id: str,
    user_message: str,
    assistant_reply: str,
):
    """
    Background worker task to persist user message and assistant reply to MongoDB,
    update rolling Redis context, and update conversation's last_message_at timestamp.
    """
    now = datetime.now(timezone.utc)

    # 1. Update Redis context with assistant reply
    await append_context(redis_client, conversation_id, "assistant", assistant_reply)

    # 2. Persist user and assistant messages in MongoDB
    user_msg_doc = {
        "conversation_id": conversation_id,
        "user_id": user_id,
        "role": "user",
        "content": user_message,
        "timestamp": now,
    }
    assistant_msg_doc = {
        "conversation_id": conversation_id,
        "user_id": user_id,
        "role": "assistant",
        "content": assistant_reply,
        "timestamp": now,
    }
    await db.messages.insert_many([user_msg_doc, assistant_msg_doc])

    # 3. Update conversation's last_message_at
    await db.conversations.update_one(
        {"conversation_id": conversation_id},
        {"$set": {"last_message_at": now}}
    )


@router.get("/me")
def get_current_user_info(payload: Dict[str, Any] = Depends(verify_jwt)):
    """Protected endpoint returning decoded JWT payload for authentication testing."""
    return {"status": "authenticated", "user": payload}


@router.get("/collections")
def list_available_collections():
    """
    Public/Protected endpoint returning available vector collections for frontend selection.
    """
    return {"collections": settings.parsed_collections}


@router.post("/stream")
async def stream_chat(
    req: ChatRequest,
    payload: Dict[str, Any] = Depends(verify_jwt),
    db: motor.motor_asyncio.AsyncIOMotorDatabase = Depends(get_db),
    redis_client: redis.Redis = Depends(get_redis),
):
    """
    Protected endpoint to stream RAG-augmented LLM chat responses as SSE.
    """
    user_id = payload["user_id"]
    conv_id = req.conversation_id

    # Validate or create conversation doc
    if not conv_id:
        conv_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        title_snippet = req.message[:30] + ("..." if len(req.message) > 30 else "")
        conv_doc = {
            "conversation_id": conv_id,
            "user_id": user_id,
            "title": title_snippet or "New Conversation",
            "created_at": now,
            "last_message_at": now,
        }
        await db.conversations.insert_one(conv_doc)
    else:
        existing_conv = await db.conversations.find_one({"conversation_id": conv_id})
        if not existing_conv:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )
        if existing_conv.get("user_id") != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this conversation",
            )

    # Determine target Qdrant collections to query
    target_collections = req.collections or settings.parsed_collections

    # 1. Retrieve RAG context chunks from Qdrant Cloud
    context_chunks = search_all_collections(
        query=req.message,
        collection_names=target_collections,
        top_k_per_collection=3
    )

    # 2. Build RAG System Prompt
    system_prompt = _build_rag_system_prompt(context_chunks)

    # 3. Fetch rolling conversation history from Redis
    current_context = await get_context(redis_client, conv_id)

    # 4. Append new user message to Redis context
    await append_context(redis_client, conv_id, "user", req.message)

    # 5. Format messages array for Anthropic LLM API call
    messages_for_llm = current_context + [{"role": "user", "content": req.message}]

    async def event_generator():
        # Yield first SSE event containing conversation_id metadata
        yield f"data: {json.dumps({'conversation_id': conv_id})}\n\n"

        full_reply = ""
        # Stream text chunks from Claude with RAG system prompt
        async for chunk in stream_llm_response(messages_for_llm, system_prompt=system_prompt):
            full_reply += chunk
            yield f"data: {json.dumps({'text': chunk})}\n\n"

        yield "data: [DONE]\n\n"

        # Trigger background task for MongoDB & Redis persistence
        asyncio.create_task(
            _persist_chat_interaction(
                db=db,
                redis_client=redis_client,
                conversation_id=conv_id,
                user_id=user_id,
                user_message=req.message,
                assistant_reply=full_reply,
            )
        )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/conversations")
async def list_user_conversations(
    payload: Dict[str, Any] = Depends(verify_jwt),
    db: motor.motor_asyncio.AsyncIOMotorDatabase = Depends(get_db),
):
    """List user's conversations sorted by last_message_at desc."""
    user_id = payload["user_id"]
    cursor = db.conversations.find({"user_id": user_id}).sort("last_message_at", -1)
    conversations = await cursor.to_list(length=100)
    return {"conversations": serialize_docs(conversations)}


@router.get("/conversations/{conversation_id}")
async def get_conversation_history(
    conversation_id: str,
    payload: Dict[str, Any] = Depends(verify_jwt),
    db: motor.motor_asyncio.AsyncIOMotorDatabase = Depends(get_db),
):
    """Fetch conversation details and full message history."""
    user_id = payload["user_id"]
    conv = await db.conversations.find_one({"conversation_id": conversation_id})
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    if conv.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this conversation",
        )

    cursor = db.messages.find({"conversation_id": conversation_id}).sort("timestamp", 1)
    messages = await cursor.to_list(length=1000)

    return {
        "conversation": serialize_doc(conv),
        "messages": serialize_docs(messages),
    }


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    payload: Dict[str, Any] = Depends(verify_jwt),
    db: motor.motor_asyncio.AsyncIOMotorDatabase = Depends(get_db),
    redis_client: redis.Redis = Depends(get_redis),
):
    """Delete conversation, messages, and Redis context key."""
    user_id = payload["user_id"]
    conv = await db.conversations.find_one({"conversation_id": conversation_id})
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    if conv.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this conversation",
        )

    await db.conversations.delete_one({"conversation_id": conversation_id})
    await db.messages.delete_many({"conversation_id": conversation_id})
    await redis_client.delete(f"chat:context:{conversation_id}")

    return {
        "message": "Conversation deleted successfully",
        "conversation_id": conversation_id,
    }
