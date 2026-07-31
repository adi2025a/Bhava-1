from typing import AsyncGenerator, List, Dict, Any, Optional

from google import genai
from google.genai import types

from app.config import settings


async def stream_llm_response(
    messages: List[Dict[str, Any]],
    system_prompt: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    try:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)

        # Gemini uses "model" for assistant role, not "assistant"
        contents = [
            types.Content(
                role="model" if msg["role"] == "assistant" else "user",
                parts=[types.Part.from_text(text=msg["content"])],
            )
            for msg in messages
        ]

        config = types.GenerateContentConfig(
            max_output_tokens=1024,
            system_instruction=system_prompt or None,
        )

        stream = await client.aio.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=contents,
            config=config,
        )
        async for chunk in stream:
            if chunk.text:
                yield chunk.text

    except Exception as e:
        yield f"\n[LLM Streaming Error: {str(e)}]"
