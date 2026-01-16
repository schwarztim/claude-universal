# Claude Universal - Publish History

## Fix Applied - 2026-01-16

### Issues Fixed

1. **CRITICAL: Filesystem Access Issue (tool_use/tool_result conversion)**
   - Root cause: `convert_messages()` had empty `pass` statements for `tool_use` and `tool_result` blocks
   - Impact: Claude Code couldn't read/write files through the proxy because tool results were being dropped
   - Fix: Implemented complete tool_use/tool_result conversion to OpenAI format:
     - Assistant `tool_use` blocks → OpenAI `tool_calls` array
     - User `tool_result` blocks → OpenAI `role: "tool"` messages

2. **Streaming Tool Calls Not Working**
   - The `stream_response()` function only handled text content deltas
   - Now properly handles `tool_calls` in streaming responses with correct Claude SSE format

3. **33 Lint Errors Fixed**
   - Unused imports removed
   - Import blocks sorted
   - Line length issues resolved
   - f-strings without placeholders fixed
   - Deprecated ruff config updated

### Performance Improvements
- Refactored streaming code with `_sse_event()` helper for cleaner SSE formatting

### Documentation Updated
- Local: ✅
- GitHub: Pending (user can push changes)

### Files Modified
- `claude_universal/proxy.py` - Complete rewrite of `convert_messages()` and `stream_response()`
- `claude_universal/launcher.py` - Line length fixes
- `claude_universal/wizard.py` - Line length fixes
- `pyproject.toml` - Updated ruff config to use lint section
- `tests/test_proxy.py` - NEW: 17 tests for message conversion

### Test Results
- 17 tests added and passing
- Tests cover critical tool_use/tool_result conversion that caused filesystem issue
