# Claude Azure

<div align="center">

**Use Claude Code with Azure OpenAI - no Anthropic subscription required**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## What This Does

Claude Azure is a launcher that lets you use [Claude Code](https://github.com/anthropics/claude-code) (Anthropic's CLI coding assistant) with **Azure OpenAI** or other OpenAI-compatible backends.

### How It Works

```
┌──────────────────────────────────────────────────────────────┐
│                        claude-azure                           │
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
│  • Forwards to your Azure OpenAI endpoint                    │
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

## Installation

### Prerequisites

- Python 3.10+
- [Claude Code](https://github.com/anthropics/claude-code) installed (`npm install -g @anthropic-ai/claude-code`)

### Install via pipx (recommended)

```bash
pipx install git+https://github.com/schwarztim/claude-azure.git
```

### Install via pip

```bash
pip install git+https://github.com/schwarztim/claude-azure.git
```

## Quick Start

```bash
# First run - triggers setup wizard
claude-azure

# You'll be prompted for:
# - Azure endpoint (e.g., https://your-resource.openai.azure.com)
# - API key
# - Deployment name (e.g., gpt-4o, gpt-5.2)
```

## Usage

```bash
# Run with your Azure backend
claude-azure

# Show proxy logs (diagnose issues)
claude-azure --verbose

# Reconfigure settings
claude-azure --setup

# Update to latest version
claude-azure --update

# Show version
claude-azure --version

# Pass arguments to Claude Code
claude-azure -c  # continue last conversation
```

## Configuration

Config is stored in `~/.claude-azure/config.json`:

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

Claude Code uses three model tiers. You map them to your Azure deployments:

| Claude Code Tier | Description | Example Mapping |
|------------------|-------------|-----------------|
| Opus | Most capable | gpt-4-turbo, gpt-5.2 |
| Sonnet | Balanced (default) | gpt-4o, gpt-5.2 |
| Haiku | Fast | gpt-4o-mini |

## Supported Providers

| Provider | Description |
|----------|-------------|
| **Azure OpenAI** | Azure-hosted OpenAI models (GPT-4, GPT-5, etc.) |
| **OpenAI** | OpenAI API directly |
| **Anthropic** | Passthrough mode (requires Anthropic subscription) |
| **Ollama** | Local models via Ollama |
| **Custom** | Any OpenAI-compatible API |

## Troubleshooting

### Verify requests go to Azure

Run with `--verbose` to see proxy logs:

```bash
claude-azure --verbose
```

You'll see lines like:

```
[PROXY] claude-sonnet-4-20250514 -> gpt-5.2 @ https://your-azure.openai.azure.com/openai/v1/chat/completions
```

### "claude command not found"

Install Claude Code first:

```bash
npm install -g @anthropic-ai/claude-code
```

### Connection errors

Run `claude-azure --setup` to reconfigure your backend settings.

## License

MIT
