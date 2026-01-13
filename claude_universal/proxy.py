"""Embedded proxy server for Claude Universal.

Translates Claude API requests to OpenAI-compatible format.
"""

import json
import os
import sys
import time
from typing import Any, AsyncGenerator

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse


# Configuration from environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION", "2024-08-01-preview")

BIG_MODEL = os.environ.get("BIG_MODEL", "gpt-4-turbo")
MIDDLE_MODEL = os.environ.get("MIDDLE_MODEL", "gpt-4o")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "gpt-4o-mini")

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8082"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "WARNING").lower()


app = FastAPI(title="Claude Universal Proxy")


def map_model(claude_model: str) -> str:
    """Map Claude model names to configured backend models."""
    model_lower = claude_model.lower()

    if "opus" in model_lower:
        return BIG_MODEL
    elif "sonnet" in model_lower:
        return MIDDLE_MODEL
    elif "haiku" in model_lower:
        return SMALL_MODEL

    # Default to middle model
    return MIDDLE_MODEL


def convert_messages(claude_messages: list[dict]) -> list[dict]:
    """Convert Claude message format to OpenAI format."""
    openai_messages = []

    for msg in claude_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Handle different content types
        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Handle multimodal content
            parts = []
            for block in content:
                if block.get("type") == "text":
                    parts.append({"type": "text", "text": block.get("text", "")})
                elif block.get("type") == "image":
                    # Handle image content
                    source = block.get("source", {})
                    if source.get("type") == "base64":
                        parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
                            }
                        })
                elif block.get("type") == "tool_use":
                    # Tool use becomes assistant message with tool calls
                    pass
                elif block.get("type") == "tool_result":
                    # Tool result handling
                    pass

            if parts:
                openai_messages.append({"role": role, "content": parts})
            else:
                # Fallback to string content
                text_parts = [b.get("text", "") for b in content if b.get("type") == "text"]
                openai_messages.append({"role": role, "content": " ".join(text_parts)})

    return openai_messages


def convert_tools(claude_tools: list[dict]) -> list[dict]:
    """Convert Claude tools to OpenAI function format."""
    openai_tools = []

    for tool in claude_tools:
        if tool.get("type") == "function":
            # Already in function format
            openai_tools.append(tool)
        else:
            # Convert from Claude format
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                }
            })

    return openai_tools


def build_openai_request(claude_request: dict) -> dict:
    """Build OpenAI-compatible request from Claude request."""
    openai_request = {
        "model": map_model(claude_request.get("model", "")),
        "messages": convert_messages(claude_request.get("messages", [])),
        "stream": claude_request.get("stream", False),
    }

    # Handle max_tokens - cap at 128K for GPT-5.2
    max_tokens = claude_request.get("max_tokens", 16384)
    # Cap at 128000 (GPT-5.2's max output tokens)
    max_tokens = min(max_tokens, 128000)
    openai_request["max_completion_tokens"] = max_tokens

    # Handle temperature
    if "temperature" in claude_request:
        openai_request["temperature"] = claude_request["temperature"]

    # Handle tools
    if "tools" in claude_request:
        openai_request["tools"] = convert_tools(claude_request["tools"])

    # Handle system message
    if "system" in claude_request:
        system_content = claude_request["system"]
        if isinstance(system_content, str):
            openai_request["messages"].insert(0, {"role": "system", "content": system_content})
        elif isinstance(system_content, list):
            text = " ".join(b.get("text", "") for b in system_content if b.get("type") == "text")
            openai_request["messages"].insert(0, {"role": "system", "content": text})

    return openai_request


def convert_response(openai_response: dict, model: str) -> dict:
    """Convert OpenAI response to Claude format."""
    choice = openai_response.get("choices", [{}])[0]
    message = choice.get("message", {})

    content = []

    # Handle text content
    if message.get("content"):
        content.append({
            "type": "text",
            "text": message["content"]
        })

    # Handle tool calls
    if message.get("tool_calls"):
        for tool_call in message["tool_calls"]:
            content.append({
                "type": "tool_use",
                "id": tool_call.get("id", ""),
                "name": tool_call.get("function", {}).get("name", ""),
                "input": json.loads(tool_call.get("function", {}).get("arguments", "{}"))
            })

    # Determine stop reason
    finish_reason = choice.get("finish_reason", "end_turn")
    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }
    stop_reason = stop_reason_map.get(finish_reason, "end_turn")

    return {
        "id": openai_response.get("id", f"msg_{int(time.time())}"),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": openai_response.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": openai_response.get("usage", {}).get("completion_tokens", 0),
        }
    }


async def stream_response(openai_stream: AsyncGenerator, model: str) -> AsyncGenerator[str, None]:
    """Convert OpenAI streaming response to Claude SSE format."""
    message_id = f"msg_{int(time.time())}"
    input_tokens = 0
    output_tokens = 0
    content_blocks = []
    current_text = ""

    # Send message_start
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': message_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"

    # Send content_block_start
    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

    async for chunk in openai_stream:
        if chunk.startswith("data: "):
            data_str = chunk[6:].strip()
            if data_str == "[DONE]":
                break

            try:
                data = json.loads(data_str)
                choices = data.get("choices", [])

                if choices:
                    delta = choices[0].get("delta", {})

                    # Handle content delta
                    if "content" in delta and delta["content"]:
                        text = delta["content"]
                        current_text += text
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"

                # Handle usage info
                if "usage" in data:
                    input_tokens = data["usage"].get("prompt_tokens", input_tokens)
                    output_tokens = data["usage"].get("completion_tokens", output_tokens)

            except json.JSONDecodeError:
                continue

    # Send content_block_stop
    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

    # Send message_delta with stop reason
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}})}\n\n"

    # Send message_stop
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


def get_api_url(endpoint: str, model: str = "") -> str:
    """Get the full API URL for the configured backend."""
    base = OPENAI_BASE_URL.rstrip("/")

    # Handle Azure-specific URL format
    if "azure" in base.lower() or "openai.azure.com" in base:
        # Use OpenAI-compatible v1 endpoint format (no api_version needed)
        if not base.endswith("/v1"):
            base = f"{base}/openai/v1"
        return f"{base}/chat/completions"

    return f"{base}/{endpoint}"


def get_headers() -> dict:
    """Get headers for API requests."""
    base = OPENAI_BASE_URL.lower()

    if "azure" in base or "openai.azure.com" in base:
        return {"api-key": OPENAI_API_KEY, "Content-Type": "application/json"}

    return {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "provider": "azure" if "azure" in OPENAI_BASE_URL.lower() else "openai",
    }


@app.post("/v1/messages")
async def messages(request: Request):
    """Handle Claude Messages API requests."""
    try:
        claude_request = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    # Build OpenAI request
    openai_request = build_openai_request(claude_request)
    original_model = claude_request.get("model", "")

    # Make request to backend
    mapped_model = openai_request.get("model", MIDDLE_MODEL)
    url = get_api_url("chat/completions", mapped_model)
    headers = get_headers()

    # Log the request
    print(f"[PROXY] {original_model} -> {mapped_model} @ {url}")

    if openai_request.get("stream"):
        # Streaming response - create generator that manages its own client
        async def generate_stream():
            # Use longer timeout for large requests - 10 minutes
            timeout = httpx.Timeout(600.0, connect=10.0)
            client = httpx.AsyncClient(timeout=timeout)
            try:
                async with client.stream("POST", url, json=openai_request, headers=headers) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        raise HTTPException(status_code=response.status_code, detail=error_text.decode())

                    async for line in response.aiter_lines():
                        yield line + "\n"
            finally:
                await client.aclose()

        return StreamingResponse(
            stream_response(generate_stream(), original_model),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming response
        # Use longer timeout for large requests - 10 minutes
        timeout = httpx.Timeout(600.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=openai_request, headers=headers)

            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)

            openai_response = response.json()
            claude_response = convert_response(openai_response, original_model)

            return JSONResponse(content=claude_response)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all(path: str, request: Request):
    """Catch-all for unhandled routes - return success for various endpoints."""
    # Return success for auth/validation endpoints
    if "auth" in path or "validate" in path or "token" in path:
        return {"status": "ok"}

    # Return success for telemetry/logging endpoints (we don't need to log)
    if "event_logging" in path or "telemetry" in path or "analytics" in path:
        return {"status": "ok"}

    # Return 404 for unknown endpoints
    raise HTTPException(status_code=404, detail=f"Not found: /{path}")


def main():
    """Run the proxy server."""
    print(f"Claude Universal Proxy starting on {HOST}:{PORT}")
    print(f"Backend: {OPENAI_BASE_URL}")
    print(f"Models: opus={BIG_MODEL}, sonnet={MIDDLE_MODEL}, haiku={SMALL_MODEL}")

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level=LOG_LEVEL,
    )


if __name__ == "__main__":
    main()
