"""Tests for the proxy message conversion functionality."""

import json

from claude_universal.proxy import (
    build_openai_request,
    convert_messages,
    convert_response,
    convert_tools,
    map_model,
)


class TestMapModel:
    """Tests for model mapping."""

    def test_maps_opus_to_big_model(self):
        assert map_model("claude-opus-4-20250514") == "gpt-4-turbo"
        assert map_model("claude-3-opus-20240229") == "gpt-4-turbo"

    def test_maps_sonnet_to_middle_model(self):
        assert map_model("claude-sonnet-4-20250514") == "gpt-4o"
        assert map_model("claude-3-5-sonnet-20240620") == "gpt-4o"

    def test_maps_haiku_to_small_model(self):
        assert map_model("claude-haiku-3-20240307") == "gpt-4o-mini"

    def test_defaults_to_middle_model(self):
        assert map_model("unknown-model") == "gpt-4o"


class TestConvertMessages:
    """Tests for Claude to OpenAI message conversion."""

    def test_simple_text_message(self):
        """Test converting simple text messages."""
        claude_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = convert_messages(claude_messages)
        assert result == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

    def test_multimodal_text_content(self):
        """Test converting messages with text blocks."""
        claude_messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What's in this image?"}],
            }
        ]
        result = convert_messages(claude_messages)
        assert result == [{"role": "user", "content": "What's in this image?"}]

    def test_tool_use_conversion(self):
        """Test converting assistant tool_use messages to OpenAI format.

        This is critical for Claude Code filesystem operations!
        """
        claude_messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me read that file."},
                    {
                        "type": "tool_use",
                        "id": "toolu_01abc123",
                        "name": "Read",
                        "input": {"file_path": "/path/to/file.txt"},
                    },
                ],
            }
        ]
        result = convert_messages(claude_messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Let me read that file."
        assert len(result[0]["tool_calls"]) == 1

        tool_call = result[0]["tool_calls"][0]
        assert tool_call["id"] == "toolu_01abc123"
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "Read"
        assert json.loads(tool_call["function"]["arguments"]) == {
            "file_path": "/path/to/file.txt"
        }

    def test_tool_result_conversion(self):
        """Test converting user tool_result messages to OpenAI format.

        This is the ROOT CAUSE of the filesystem access issue!
        Tool results must be converted to 'role: tool' messages.
        """
        claude_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01abc123",
                        "content": "File contents here:\nline 1\nline 2",
                    }
                ],
            }
        ]
        result = convert_messages(claude_messages)

        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "toolu_01abc123"
        assert result[0]["content"] == "File contents here:\nline 1\nline 2"

    def test_tool_result_with_error(self):
        """Test converting tool results that are errors."""
        claude_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01abc123",
                        "content": "File not found",
                        "is_error": True,
                    }
                ],
            }
        ]
        result = convert_messages(claude_messages)

        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert "Error:" in result[0]["content"]

    def test_tool_result_with_nested_content(self):
        """Test tool results where content is a list of blocks."""
        claude_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01abc123",
                        "content": [
                            {"type": "text", "text": "Line 1"},
                            {"type": "text", "text": "Line 2"},
                        ],
                    }
                ],
            }
        ]
        result = convert_messages(claude_messages)

        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["content"] == "Line 1\nLine 2"

    def test_multiple_tool_results(self):
        """Test converting multiple tool results in one message."""
        claude_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01",
                        "content": "Result 1",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_02",
                        "content": "Result 2",
                    },
                ],
            }
        ]
        result = convert_messages(claude_messages)

        # Each tool result becomes a separate message
        assert len(result) == 2
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "toolu_01"
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "toolu_02"


class TestConvertTools:
    """Tests for tool definition conversion."""

    def test_claude_tool_format(self):
        """Test converting Claude tool definitions to OpenAI format."""
        claude_tools = [
            {
                "name": "Read",
                "description": "Read a file from the filesystem",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                    },
                    "required": ["file_path"],
                },
            }
        ]
        result = convert_tools(claude_tools)

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "Read"
        assert result[0]["function"]["description"] == "Read a file from the filesystem"


class TestConvertResponse:
    """Tests for OpenAI to Claude response conversion."""

    def test_text_response(self):
        """Test converting a simple text response."""
        openai_response = {
            "id": "chatcmpl-123",
            "choices": [
                {
                    "message": {"content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = convert_response(openai_response, "claude-3-sonnet")

        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello!"
        assert result["stop_reason"] == "end_turn"

    def test_tool_call_response(self):
        """Test converting a response with tool calls."""
        openai_response = {
            "id": "chatcmpl-123",
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "function": {
                                    "name": "Read",
                                    "arguments": '{"file_path": "/test.txt"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }
        result = convert_response(openai_response, "claude-3-sonnet")

        assert result["stop_reason"] == "tool_use"
        # Find tool_use content
        tool_use = next(c for c in result["content"] if c["type"] == "tool_use")
        assert tool_use["id"] == "call_abc123"
        assert tool_use["name"] == "Read"
        assert tool_use["input"] == {"file_path": "/test.txt"}


class TestBuildOpenaiRequest:
    """Tests for building complete OpenAI requests."""

    def test_basic_request(self):
        """Test building a basic request."""
        claude_request = {
            "model": "claude-3-sonnet",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 1024,
        }
        result = build_openai_request(claude_request)

        assert result["model"] == "gpt-4o"
        assert result["messages"] == [{"role": "user", "content": "Hello"}]
        # max_completion_tokens should be at least 64k
        assert result["max_completion_tokens"] >= 64000

    def test_request_with_system(self):
        """Test that system prompt is included."""
        claude_request = {
            "model": "claude-3-sonnet",
            "messages": [{"role": "user", "content": "Hello"}],
            "system": "You are a helpful assistant.",
            "max_tokens": 1024,
        }
        result = build_openai_request(claude_request)

        # System message should be first
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are a helpful assistant."

    def test_request_with_tools(self):
        """Test that tools are properly converted."""
        claude_request = {
            "model": "claude-3-sonnet",
            "messages": [{"role": "user", "content": "Read /test.txt"}],
            "tools": [
                {
                    "name": "Read",
                    "description": "Read a file",
                    "input_schema": {"type": "object"},
                }
            ],
            "max_tokens": 1024,
        }
        result = build_openai_request(claude_request)

        assert "tools" in result
        assert result["tools"][0]["function"]["name"] == "Read"
