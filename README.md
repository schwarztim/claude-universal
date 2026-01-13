# Claude Universal

Use Claude Code with any AI backend - Azure OpenAI, OpenAI, Ollama, or any OpenAI-compatible API.

## Features

- **Multi-provider support**: Azure OpenAI, OpenAI, Anthropic, Ollama, and custom APIs
- **Interactive setup wizard**: Easy first-time configuration
- **Per-session proxy**: No global services to manage
- **Cross-platform**: Works on Linux, macOS, and Windows
- **Auto-updates**: Uses official Claude Code, which updates independently

## Quick Start

### Prerequisites

- Python 3.10+
- [Claude Code](https://claude.ai/code) installed

### Installation

```bash
pip install claude-universal
```

Or clone and install:

```bash
git clone https://github.com/schwarztim/claude-universal
cd claude-universal
pip install -e .
```

### Usage

```bash
# First run triggers setup wizard
claude-universal

# Reconfigure
claude-universal --setup

# Pass arguments to Claude Code
claude-universal --help
```

## How It Works

```
┌─────────────────────────────┐
│     claude-universal        │
│  1. Load config             │
│  2. Start embedded proxy    │
│  3. Launch claude           │
└─────────────────────────────┘
            ↓
┌─────────────────────────────┐
│    Embedded Proxy           │
│  - Translates Claude API    │
│    to OpenAI format         │
│  - Routes to your backend   │
└─────────────────────────────┘
            ↓
┌─────────────────────────────┐
│   Your AI Backend           │
│  - Azure OpenAI             │
│  - OpenAI                   │
│  - Ollama                   │
│  - Custom                   │
└─────────────────────────────┘
```

## Configuration

Config is stored in `~/.claude-universal/config.json`:

```json
{
  "provider": "azure",
  "azure": {
    "api_key": "your-api-key",
    "endpoint": "https://your-resource.openai.azure.com/",
    "api_version": "2024-08-01-preview"
  },
  "models": {
    "opus": "gpt-4-turbo",
    "sonnet": "gpt-4o",
    "haiku": "gpt-4o-mini"
  }
}
```

## Supported Providers

| Provider | Description |
|----------|-------------|
| Azure OpenAI | Azure-hosted OpenAI models |
| OpenAI | OpenAI API directly |
| Anthropic | Passthrough to Anthropic (requires subscription) |
| Ollama | Local models via Ollama |
| Custom | Any OpenAI-compatible API |

## Model Mapping

Claude models are mapped to your backend:

| Claude Model | Default Mapping |
|--------------|-----------------|
| Opus | gpt-4-turbo |
| Sonnet | gpt-4o |
| Haiku | gpt-4o-mini |

Customize in the setup wizard or config file.

## Troubleshooting

### "claude command not found"

Install Claude Code first:
- Visit https://claude.ai/code
- Or use the official installer

### Connection errors

Run `claude-universal --setup` to reconfigure your backend settings.

### Proxy issues

Check `~/.claude-universal/` for logs.

## License

MIT
