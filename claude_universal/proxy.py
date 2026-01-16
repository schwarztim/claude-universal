"""Embedded proxy server for Claude Universal.

Translates Claude API requests to OpenAI-compatible format.
"""

import json
import os
import time
from typing import AsyncGenerator

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

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
    """Convert Claude message format to OpenAI format.

    Handles:
    - Text content (simple and multimodal)
    - Image content (base64)
    - Tool use blocks (assistant -> tool_calls)
    - Tool result blocks (user -> tool role messages)
    """
    openai_messages = []

    for msg in claude_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Handle simple string content
        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
            continue

        # Handle list content (multimodal, tool_use, tool_result)
        if isinstance(content, list):
            # Separate different block types
            text_parts = []
            image_parts = []
            tool_use_blocks = []
            tool_result_blocks = []

            for block in content:
                block_type = block.get("type", "")

                if block_type == "text":
                    text_parts.append({"type": "text", "text": block.get("text", "")})

                elif block_type == "image":
                    source = block.get("source", {})
                    if source.get("type") == "base64":
                        media_type = source.get("media_type", "image/png")
                        data = source.get("data", "")
                        image_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{data}"}
                        })

                elif block_type == "tool_use":
                    tool_use_blocks.append(block)

                elif block_type == "tool_result":
                    tool_result_blocks.append(block)

            # Handle assistant messages with tool_use blocks
            if role == "assistant" and tool_use_blocks:
                # Build tool_calls array for OpenAI format
                tool_calls = []
                for tool_block in tool_use_blocks:
                    tool_calls.append({
                        "id": tool_block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tool_block.get("name", ""),
                            "arguments": json.dumps(tool_block.get("input", {}))
                        }
                    })

                # Include any text content
                text_content = ""
                if text_parts:
                    text_content = " ".join(p.get("text", "") for p in text_parts)

                openai_messages.append({
                    "role": "assistant",
                    "content": text_content or None,
                    "tool_calls": tool_calls
                })

            # Handle user messages with tool_result blocks
            elif tool_result_blocks:
                # Each tool_result becomes a separate "tool" role message in OpenAI
                for result_block in tool_result_blocks:
                    tool_call_id = result_block.get("tool_use_id", "")
                    result_content = result_block.get("content", "")

                    # Handle content that might be a list of blocks
                    if isinstance(result_content, list):
                        text_pieces = []
                        for item in result_content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_pieces.append(item.get("text", ""))
                            elif isinstance(item, str):
                                text_pieces.append(item)
                        result_content = "\n".join(text_pieces)
                    elif not isinstance(result_content, str):
                        result_content = json.dumps(result_content)

                    # Check for error in tool result
                    is_error = result_block.get("is_error", False)

                    openai_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": result_content if not is_error else f"Error: {result_content}"
                    })

                # Also include any text content from the user message
                if text_parts:
                    combined_text = " ".join(p.get("text", "") for p in text_parts)
                    if combined_text.strip():
                        openai_messages.append({"role": "user", "content": combined_text})

            # Handle regular content (text and images)
            elif text_parts or image_parts:
                combined_parts = text_parts + image_parts
                if len(combined_parts) == 1 and combined_parts[0].get("type") == "text":
                    # Single text block - use simple string content
                    openai_messages.append({
                        "role": role,
                        "content": combined_parts[0].get("text", "")
                    })
                else:
                    openai_messages.append({"role": role, "content": combined_parts})

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

    # Handle max_tokens - use 128K for GPT-5.2
    max_tokens = claude_request.get("max_tokens", 128000)
    # Always use at least 64k, max 128k (GPT-5.2's limits)
    max_tokens = max(64000, min(max_tokens, 128000))
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


def _sse_event(event_type: str, data: dict) -> str:
    """Format an SSE event with the given type and data."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def stream_response(
    openai_stream: AsyncGenerator, model: str
) -> AsyncGenerator[str, None]:
    """Convert OpenAI streaming response to Claude SSE format.

    Handles both text content and tool calls in the stream.
    """
    message_id = f"msg_{int(time.time())}"
    output_tokens = 0
    finish_reason = "end_turn"

    # Track tool calls being built
    tool_calls: dict[int, dict] = {}  # index -> {id, name, arguments}
    current_block_index = 0
    text_block_started = False
    tool_blocks_started: set[int] = set()
    chunk_count = 0

    # Send message_start
    msg_start = {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    }
    yield _sse_event("message_start", msg_start)

    async for chunk in openai_stream:
        chunk_count += 1
        chunk = chunk.strip()  # Remove trailing newlines

        # Skip empty lines
        if not chunk:
            continue

        if chunk.startswith("data: "):
            data_str = chunk[6:].strip()
            if data_str == "[DONE]":
                break

            try:
                data = json.loads(data_str)
                choices = data.get("choices", [])

                if choices:
                    choice = choices[0]
                    delta = choice.get("delta", {})

                    # Check finish reason
                    if choice.get("finish_reason"):
                        fr = choice["finish_reason"]
                        if fr == "tool_calls":
                            finish_reason = "tool_use"
                        elif fr == "length":
                            finish_reason = "max_tokens"
                        else:
                            finish_reason = "end_turn"

                    # Handle text content delta
                    if "content" in delta and delta["content"]:
                        # Start text block if not started
                        if not text_block_started:
                            block_start = {
                                "type": "content_block_start",
                                "index": current_block_index,
                                "content_block": {"type": "text", "text": ""},
                            }
                            yield _sse_event("content_block_start", block_start)
                            text_block_started = True

                        text = delta["content"]
                        block_delta = {
                            "type": "content_block_delta",
                            "index": current_block_index,
                            "delta": {"type": "text_delta", "text": text},
                        }
                        yield _sse_event("content_block_delta", block_delta)

                    # Handle tool calls delta
                    if "tool_calls" in delta:
                        print(f"[PROXY] Tool call delta received: {delta['tool_calls']}")
                        # Close text block if open
                        if text_block_started:
                            if current_block_index not in tool_blocks_started:
                                block_stop = {
                                    "type": "content_block_stop",
                                    "index": current_block_index,
                                }
                                yield _sse_event("content_block_stop", block_stop)
                                current_block_index += 1
                                text_block_started = False

                        for tc in delta["tool_calls"]:
                            tc_index = tc.get("index", 0)
                            tool_block_index = current_block_index + tc_index

                            # Initialize tool call if new
                            if tc_index not in tool_calls:
                                tool_calls[tc_index] = {
                                    "id": tc.get("id", ""),
                                    "name": "",
                                    "arguments": "",
                                }

                            # Update tool call data
                            if tc.get("id"):
                                tool_calls[tc_index]["id"] = tc["id"]
                            if tc.get("function", {}).get("name"):
                                tool_calls[tc_index]["name"] = tc["function"]["name"]
                            if tc.get("function", {}).get("arguments"):
                                tool_calls[tc_index]["arguments"] += (
                                    tc["function"]["arguments"]
                                )

                            # Send tool_use block start if not sent
                            tc_data = tool_calls[tc_index]
                            if tool_block_index not in tool_blocks_started and tc_data["name"]:
                                tool_start = {
                                    "type": "content_block_start",
                                    "index": tool_block_index,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": tc_data["id"],
                                        "name": tc_data["name"],
                                        "input": {},
                                    },
                                }
                                yield _sse_event("content_block_start", tool_start)
                                tool_blocks_started.add(tool_block_index)

                            # Send arguments as input_json_delta
                            func_args = tc.get("function", {}).get("arguments")
                            if func_args and tool_block_index in tool_blocks_started:
                                arg_delta = {
                                    "type": "content_block_delta",
                                    "index": tool_block_index,
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": func_args,
                                    },
                                }
                                yield _sse_event("content_block_delta", arg_delta)

                # Handle usage info
                if "usage" in data:
                    output_tokens = data["usage"].get(
                        "completion_tokens", output_tokens
                    )

            except json.JSONDecodeError as e:
                print(f"[PROXY] JSON decode error: {e} for chunk: {chunk[:100]}")
                continue

    print(f"[PROXY] Stream processing complete. Chunks: {chunk_count}, "
          f"text_started: {text_block_started}, tool_blocks: {len(tool_blocks_started)}")

    # Close any open text block
    if text_block_started and current_block_index not in tool_blocks_started:
        yield _sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": current_block_index},
        )

    # Close any open tool blocks
    for tool_block_index in sorted(tool_blocks_started):
        yield _sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": tool_block_index},
        )

    # If we never started any blocks, send an empty text block
    # This can happen if the model returns nothing
    if not text_block_started and not tool_blocks_started:
        print("[PROXY] Warning: No content blocks were started, sending empty text block")
        yield _sse_event(
            "content_block_start",
            {"type": "content_block_start", "index": 0,
             "content_block": {"type": "text", "text": ""}},
        )
        yield _sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        )

    # Send message_delta with stop reason
    msg_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": finish_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    }
    yield _sse_event("message_delta", msg_delta)

    # Send message_stop
    yield _sse_event("message_stop", {"type": "message_stop"})
    print("[PROXY] Stream response complete")


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
    max_tokens_requested = openai_request.get("max_completion_tokens", 0)
    print(f"[PROXY] {original_model} -> {mapped_model} @ {url} (max_tokens={max_tokens_requested})")

    if openai_request.get("stream"):
        # Streaming response - create generator that manages its own client
        async def generate_stream():
            # 5 minute timeout for large requests
            timeout = httpx.Timeout(300.0, connect=10.0)
            client = httpx.AsyncClient(timeout=timeout)
            try:
                print("[PROXY] Sending request to Azure...")
                stream_req = client.stream(
                    "POST", url, json=openai_request, headers=headers
                )
                async with stream_req as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        print(f"[PROXY] Error response: {response.status_code}")
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=error_text.decode(),
                        )

                    print("[PROXY] Receiving stream from Azure...")
                    chunk_count = 0
                    async for line in response.aiter_lines():
                        chunk_count += 1
                        # Log first 5 chunks and every 100th after
                        if chunk_count <= 5 or chunk_count % 100 == 0:
                            preview = line[:80] if len(line) > 80 else line
                            print(f"[PROXY] Chunk {chunk_count}: {preview}")
                        yield line + "\n"
                    print(f"[PROXY] Stream complete. Total chunks: {chunk_count}")
            except Exception as e:
                print(f"[PROXY] Error during streaming: {e}")
                raise
            finally:
                await client.aclose()

        return StreamingResponse(
            stream_response(generate_stream(), original_model),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming response
        # 5 minute timeout for large requests
        timeout = httpx.Timeout(300.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            print("[PROXY] Sending non-streaming request to Azure...")
            response = await client.post(url, json=openai_request, headers=headers)

            if response.status_code != 200:
                print(f"[PROXY] Error response: {response.status_code}")
                raise HTTPException(status_code=response.status_code, detail=response.text)

            print("[PROXY] Received response, converting format...")
            openai_response = response.json()
            claude_response = convert_response(openai_response, original_model)

            print("[PROXY] Response complete")
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
