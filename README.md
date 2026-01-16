# Claude Universal

Use Claude Code with any AI backend - Azure OpenAI, OpenAI, Ollama, or any OpenAI-compatible API - **without an Anthropic subscription**.

## What This Does

Claude Universal is a launcher that lets you use [Claude Code](https://github.com/anthropics/claude-code) (Anthropic's CLI coding assistant) with alternative AI backends like Azure OpenAI or local models.

### How It Works

```
┌──────────────────────────────────────────────────────────────┐
│                      claude-universal                         │
│                                                              │
│  1. Starts a local proxy server on a random port            │
│  2. Sets ANTHROPIC_BASE_URL to point to the proxy           │
│  3. Launches Claude Code (official CLI)                      │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                     Embedded Proxy                            │
│                                                              │
│  • Receives Claude API requests from Claude Code             │
│  • Translates them to OpenAI-compatible format               │
│  • Forwards to your configured backend (Azure, etc.)         │
│  • Translates responses back to Claude format                │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                    Your AI Backend                            │
│                                                              │
│  Azure OpenAI  │  OpenAI  │  Ollama  │  Custom API           │
└──────────────────────────────────────────────────────────────┘
```

### Important: What to Expect

Claude Code's UI will still show "Claude" branding (model names like "Opus", "Sonnet", "Haiku", the welcome message, etc.) because Claude Code doesn't know it's talking to a different backend. **Under the hood, your requests are routed to your configured provider.**

Use `--verbose` to verify requests are going to your backend:
```
[PROXY] claude-sonnet-4-20250514 -> gpt-5.2 @ https://your-azure.openai.azure.com/openai/v1/chat/completions
```

## Installation

### Prerequisites

- Python 3.10+
- [Claude Code](https://github.com/anthropics/claude-code) installed (`npm install -g @anthropic-ai/claude-code`)

### Install via pipx (recommended)

```bash
pipx install git+https://github.com/schwarztim/claude-universal.git
```

### Install via pip

```bash
pip install git+https://github.com/schwarztim/claude-universal.git
```

### Install from source

```bash
git clone https://github.com/schwarztim/claude-universal
cd claude-universal
pip install -e .
```

## Usage

```bash
# First run - triggers setup wizard
claude-universal

# Reconfigure provider/credentials
claude-universal --setup

# Show proxy request logs (verify requests go to your backend)
claude-universal --verbose

# Update to latest version
claude-universal --update

# Show help
claude-universal --help

# Pass arguments to Claude Code
claude-universal -c  # continue last conversation
```

## Configuration

### Setup Wizard

On first run, the setup wizard guides you through:

1. **Select provider** - Azure OpenAI, OpenAI, Ollama, or custom
2. **Enter credentials** - API key and endpoint
3. **Configure model** - Which model/deployment to use
4. **Validation** - Tests the connection before saving

### Config File

Config is stored in `~/.claude-universal/config.json`:

```json
{
  "provider": "azure",
  "azure": {
    "api_key": "your-api-key",
    "endpoint": "https://your-resource.openai.azure.com/"
  },
  "models": {
    "opus": "gpt-5.2",
    "sonnet": "gpt-5.2",
    "haiku": "gpt-5.2"
  }
}
```

### Model Mapping

Claude Code uses three model tiers. You map them to your backend models:

| Claude Code Tier | Description | Example Azure Mapping |
|------------------|-------------|----------------------|
| Opus | Most capable | gpt-4-turbo, gpt-5.2 |
| Sonnet | Balanced | gpt-4o, gpt-5.2 |
| Haiku | Fast | gpt-4o-mini |

By default, the wizard sets all tiers to the same model. You can configure different models during setup or edit the config file.

## Supported Providers

| Provider | Description |
|----------|-------------|
| **Azure OpenAI** | Azure-hosted OpenAI models (GPT-4, GPT-5, etc.) |
| **OpenAI** | OpenAI API directly |
| **Anthropic** | Passthrough mode (requires Anthropic subscription) |
| **Ollama** | Local models via Ollama |
| **Custom** | Any OpenAI-compatible API |

## Running Both Claude and Claude Universal

You can use both commands independently:

- `claude` - Uses your claude.ai account (Anthropic)
- `claude-universal` - Uses your configured backend (Azure, etc.)

They don't conflict with each other.

## Troubleshooting

### "claude command not found"

Install Claude Code first:
```bash
npm install -g @anthropic-ai/claude-code
```

### Verify requests go to your backend

Run with `--verbose` to see proxy logs:
```bash
claude-universal --verbose
```

You should see lines like:
```
[PROXY] claude-sonnet-4-20250514 -> gpt-5.2 @ https://your-azure.openai.azure.com/...
```

### Connection errors

Run `claude-universal --setup` to reconfigure your backend settings.

### Model validation fails

The setup wizard tests your model by making a small request. If it fails:
- Check your API key is correct
- Check your endpoint URL is correct
- Check the deployment/model name exists in your backend

### Filesystem access not working

If Claude Code can't read/write files through the proxy, ensure you're using the latest version. Earlier versions had incomplete tool result handling. Update with:
```bash
claude-universal --update
```

### Web search limitations

Web search functionality depends on your backend model's capabilities:
- **Azure OpenAI / OpenAI**: Standard models don't have built-in web search. Consider using models with web search plugins or Bing integration if available.
- **Anthropic passthrough**: Full web search works as normal.
- **Custom backends**: Check if your backend supports function calling for web search tools.

## How the Proxy Works

The proxy translates between Claude's API format and OpenAI's format:

1. **Request translation**: Converts Claude message format to OpenAI chat completion format
2. **Model mapping**: Maps `claude-opus-4` → `gpt-5.2` (or your configured model)
3. **Streaming**: Converts OpenAI SSE streaming to Claude SSE format
4. **Response translation**: Converts OpenAI responses back to Claude format

The proxy also handles:
- Tool/function calling translation
- Image/multimodal content
- Telemetry endpoints (returns success, doesn't forward)

## License

MIT
