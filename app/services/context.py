import json
from typing import List, Dict, Any
import redis.asyncio as redis


def _context_key(conversation_id: str) -> str:
    return f"chat:context:{conversation_id}"


async def get_context(redis_client: redis.Redis, conversation_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves the last 10 messages stored in Redis for the given conversation.
    """
    key = _context_key(conversation_id)
    raw_messages = await redis_client.lrange(key, 0, -1)
    context = []
    for item in raw_messages:
        try:
            msg = json.loads(item)
            context.append(msg)
        except Exception:
            continue
    return context


async def append_context(redis_client: redis.Redis, conversation_id: str, role: str, content: str) -> None:
    """
    Appends a message to Redis context list, trims to keep only the last 10 messages,
    and sets/refreshes a 1-hour (3600 seconds) TTL.
    """
    key = _context_key(conversation_id)
    msg_json = json.dumps({"role": role, "content": content})
    
    async with redis_client.pipeline(transaction=True) as pipe:
        await pipe.rpush(key, msg_json)
        await pipe.ltrim(key, -10, -1)
        await pipe.expire(key, 3600)
        await pipe.execute()
