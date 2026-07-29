from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from bson import ObjectId
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = Field(..., description="Role of the message sender: 'user' or 'assistant'")
    content: str = Field(..., description="Text content of the message")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Conversation(BaseModel):
    conversation_id: str = Field(..., description="Unique UUID string for the conversation")
    user_id: str = Field(..., description="Owner user ID from JWT payload")
    title: str = Field(..., description="Title of the conversation")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_message_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User prompt or message content")
    conversation_id: Optional[str] = Field(None, description="Optional conversation UUID")
    collections: Optional[List[str]] = Field(
        None,
        description="Optional list of Qdrant collection names to retrieve context from"
    )


def serialize_doc(doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Helper to convert MongoDB ObjectId to string for JSON responses."""
    if not doc:
        return None
    res = dict(doc)
    if "_id" in res:
        res["id"] = str(res["_id"])
        del res["_id"]
    for key, value in res.items():
        if isinstance(value, ObjectId):
            res[key] = str(value)
        elif isinstance(value, datetime):
            res[key] = value.isoformat()
    return res


def serialize_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Helper to convert a list of MongoDB documents."""
    return [serialize_doc(doc) for doc in docs if doc is not None]
