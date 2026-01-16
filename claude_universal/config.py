"""Configuration management for Claude Universal."""

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class AzureConfig(BaseModel):
    """Azure OpenAI configuration."""
    api_key: str = ""
    endpoint: str = ""
    api_version: str = "2024-08-01-preview"
    deployment: str = ""


class OpenAIConfig(BaseModel):
    """OpenAI configuration."""
    api_key: str = ""
    org_id: str = ""
    base_url: str = "https://api.openai.com/v1"


class AnthropicConfig(BaseModel):
    """Anthropic configuration (passthrough mode)."""
    api_key: str = ""


class OllamaConfig(BaseModel):
    """Ollama configuration."""
    endpoint: str = "http://localhost:11434"


class ModelMapping(BaseModel):
    """Model mapping configuration."""
    opus: str = "gpt-4-turbo"
    sonnet: str = "gpt-4o"
    haiku: str = "gpt-4o-mini"


class ProxyConfig(BaseModel):
    """Proxy server configuration."""
    port: str = "auto"
    log_level: str = "WARNING"


class Config(BaseModel):
    """Main configuration model."""
    provider: str = ""  # azure, openai, anthropic, ollama, custom
    azure: AzureConfig = Field(default_factory=AzureConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    models: ModelMapping = Field(default_factory=ModelMapping)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)

    # Custom provider settings
    custom_base_url: str = ""
    custom_api_key: str = ""


def get_config_dir() -> Path:
    """Get the configuration directory path."""
    return Path.home() / ".claude-azure"


def get_config_path() -> Path:
    """Get the configuration file path."""
    return get_config_dir() / "config.json"


def config_exists() -> bool:
    """Check if configuration file exists."""
    return get_config_path().exists()


def load_config() -> Config:
    """Load configuration from file."""
    config_path = get_config_path()
    if not config_path.exists():
        return Config()

    with open(config_path) as f:
        data = json.load(f)

    return Config(**data)


def save_config(config: Config) -> None:
    """Save configuration to file."""
    config_dir = get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path = get_config_path()
    with open(config_path, "w") as f:
        json.dump(config.model_dump(), f, indent=2)

    # Set restrictive permissions (Unix only)
    if os.name != "nt":
        os.chmod(config_path, 0o600)


def get_provider_config(config: Config) -> dict[str, Any]:
    """Get the active provider's configuration."""
    provider = config.provider

    if provider == "azure":
        return {
            "api_key": config.azure.api_key,
            "base_url": config.azure.endpoint,
            "api_version": config.azure.api_version,
            "deployment": config.azure.deployment,
        }
    elif provider == "openai":
        return {
            "api_key": config.openai.api_key,
            "base_url": config.openai.base_url,
            "org_id": config.openai.org_id,
        }
    elif provider == "anthropic":
        return {
            "api_key": config.anthropic.api_key,
            "passthrough": True,
        }
    elif provider == "ollama":
        return {
            "base_url": config.ollama.endpoint,
            "api_key": "ollama",  # Ollama doesn't require API key
        }
    elif provider == "custom":
        return {
            "api_key": config.custom_api_key,
            "base_url": config.custom_base_url,
        }
    else:
        raise ValueError(f"Unknown provider: {provider}")
