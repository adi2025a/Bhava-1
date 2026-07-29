from typing import AsyncGenerator, List, Dict, Any, Optional
import anthropic
from app.config import settings


async def stream_llm_response(
    messages: List[Dict[str, Any]],
    system_prompt: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Streams response text chunks from Anthropic Claude API.
    Supports an optional system_prompt parameter for RAG instructions and context.
    Handles exceptions gracefully by yielding an error message block.
    """
    try:
        client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        kwargs: Dict[str, Any] = {
            "model": "claude-sonnet-5",
            "max_tokens": 1024,
            "messages": messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
    except anthropic.APIError as err:
        yield f"\n[LLM API Error: {err.message}]"
    except Exception as e:
        yield f"\n[LLM Streaming Error: {str(e)}]"
