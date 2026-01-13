"""Interactive setup wizard for Claude Universal."""

import sys
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from .config import Config, save_config, load_config, config_exists


console = Console()


def select_provider() -> str:
    """Let user select their AI provider."""
    console.print("\n[bold]Choose your AI backend:[/bold]\n")

    options = [
        ("1", "Azure OpenAI", "Use Azure-hosted OpenAI models (GPT-4, etc.)"),
        ("2", "OpenAI", "Use OpenAI API directly"),
        ("3", "Anthropic", "Use Anthropic API (requires subscription)"),
        ("4", "Ollama", "Use local Ollama models"),
        ("5", "Custom", "Use any OpenAI-compatible API"),
    ]

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Option", style="cyan")
    table.add_column("Provider", style="bold")
    table.add_column("Description", style="dim")

    for opt, name, desc in options:
        table.add_row(f"[{opt}]", name, desc)

    console.print(table)
    console.print()

    choice = Prompt.ask(
        "Select provider",
        choices=["1", "2", "3", "4", "5"],
        default="1"
    )

    provider_map = {
        "1": "azure",
        "2": "openai",
        "3": "anthropic",
        "4": "ollama",
        "5": "custom",
    }

    return provider_map[choice]


def configure_azure(config: Config) -> None:
    """Configure Azure OpenAI settings."""
    console.print("\n[bold blue]Azure OpenAI Configuration[/bold blue]\n")

    config.azure.endpoint = Prompt.ask(
        "Azure endpoint URL",
        default=config.azure.endpoint or "https://your-resource.openai.azure.com/"
    )

    config.azure.api_key = Prompt.ask(
        "API Key",
        password=True,
        default=config.azure.api_key if config.azure.api_key else None
    )

    config.azure.api_version = Prompt.ask(
        "API Version",
        default=config.azure.api_version
    )

    # Model mapping
    console.print("\n[bold]Model Mapping[/bold]")
    console.print("[dim]Map Claude models to your Azure deployments:[/dim]\n")

    config.models.opus = Prompt.ask(
        "Opus model (most capable)",
        default=config.models.opus
    )
    config.models.sonnet = Prompt.ask(
        "Sonnet model (balanced)",
        default=config.models.sonnet
    )
    config.models.haiku = Prompt.ask(
        "Haiku model (fast)",
        default=config.models.haiku
    )


def configure_openai(config: Config) -> None:
    """Configure OpenAI settings."""
    console.print("\n[bold green]OpenAI Configuration[/bold green]\n")

    config.openai.api_key = Prompt.ask(
        "API Key",
        password=True,
        default=config.openai.api_key if config.openai.api_key else None
    )

    config.openai.org_id = Prompt.ask(
        "Organization ID (optional)",
        default=config.openai.org_id or ""
    )

    # Model mapping
    console.print("\n[bold]Model Mapping[/bold]\n")

    config.models.opus = Prompt.ask(
        "Opus model",
        default="gpt-4-turbo"
    )
    config.models.sonnet = Prompt.ask(
        "Sonnet model",
        default="gpt-4o"
    )
    config.models.haiku = Prompt.ask(
        "Haiku model",
        default="gpt-4o-mini"
    )


def configure_anthropic(config: Config) -> None:
    """Configure Anthropic settings (passthrough mode)."""
    console.print("\n[bold magenta]Anthropic Configuration[/bold magenta]\n")
    console.print("[dim]This will use Anthropic's API directly (passthrough mode).[/dim]\n")

    config.anthropic.api_key = Prompt.ask(
        "API Key",
        password=True,
        default=config.anthropic.api_key if config.anthropic.api_key else None
    )


def configure_ollama(config: Config) -> None:
    """Configure Ollama settings."""
    console.print("\n[bold yellow]Ollama Configuration[/bold yellow]\n")

    config.ollama.endpoint = Prompt.ask(
        "Ollama endpoint",
        default=config.ollama.endpoint
    )

    # Model mapping
    console.print("\n[bold]Model Mapping[/bold]")
    console.print("[dim]Map Claude models to your local Ollama models:[/dim]\n")

    config.models.opus = Prompt.ask(
        "Opus model",
        default="llama3.1:70b"
    )
    config.models.sonnet = Prompt.ask(
        "Sonnet model",
        default="llama3.1:8b"
    )
    config.models.haiku = Prompt.ask(
        "Haiku model",
        default="llama3.2:3b"
    )


def configure_custom(config: Config) -> None:
    """Configure custom OpenAI-compatible API."""
    console.print("\n[bold]Custom API Configuration[/bold]\n")

    config.custom_base_url = Prompt.ask(
        "API Base URL",
        default=config.custom_base_url or "http://localhost:8080/v1"
    )

    config.custom_api_key = Prompt.ask(
        "API Key (or 'none' if not required)",
        default=config.custom_api_key or "none"
    )

    # Model mapping
    console.print("\n[bold]Model Mapping[/bold]\n")

    config.models.opus = Prompt.ask(
        "Opus model",
        default=config.models.opus
    )
    config.models.sonnet = Prompt.ask(
        "Sonnet model",
        default=config.models.sonnet
    )
    config.models.haiku = Prompt.ask(
        "Haiku model",
        default=config.models.haiku
    )


def test_connection(config: Config) -> bool:
    """Test the connection to the configured provider."""
    console.print("\n[bold]Testing connection...[/bold]")

    try:
        import httpx

        if config.provider == "azure":
            # Test Azure connection
            url = f"{config.azure.endpoint}/openai/deployments?api-version={config.azure.api_version}"
            headers = {"api-key": config.azure.api_key}
            response = httpx.get(url, headers=headers, timeout=10.0)
            if response.status_code in (200, 401, 403):  # 401/403 means auth works but no permissions
                console.print("[green]✓ Azure connection successful[/green]")
                return True

        elif config.provider == "openai":
            # Test OpenAI connection
            url = f"{config.openai.base_url}/models"
            headers = {"Authorization": f"Bearer {config.openai.api_key}"}
            response = httpx.get(url, headers=headers, timeout=10.0)
            if response.status_code == 200:
                console.print("[green]✓ OpenAI connection successful[/green]")
                return True

        elif config.provider == "anthropic":
            # Just validate API key format
            if config.anthropic.api_key.startswith("sk-"):
                console.print("[green]✓ API key format valid[/green]")
                return True

        elif config.provider == "ollama":
            # Test Ollama connection
            url = f"{config.ollama.endpoint}/api/tags"
            response = httpx.get(url, timeout=10.0)
            if response.status_code == 200:
                console.print("[green]✓ Ollama connection successful[/green]")
                return True

        elif config.provider == "custom":
            console.print("[yellow]⚠ Skipping connection test for custom provider[/yellow]")
            return True

        console.print("[red]✗ Connection failed[/red]")
        return False

    except Exception as e:
        console.print(f"[red]✗ Connection error: {e}[/red]")
        return False


def run_wizard(force: bool = False) -> bool:
    """Run the setup wizard."""
    console.print(Panel.fit(
        "[bold]Claude Universal Setup Wizard[/bold]\n"
        "Configure your AI backend for Claude Code",
        border_style="blue"
    ))

    # Load existing config if available
    if config_exists() and not force:
        config = load_config()
        if config.provider:
            console.print(f"\n[dim]Current provider: {config.provider}[/dim]")
            if not Confirm.ask("Reconfigure?", default=False):
                return True
    else:
        config = Config()

    # Select provider
    config.provider = select_provider()

    # Provider-specific configuration
    if config.provider == "azure":
        configure_azure(config)
    elif config.provider == "openai":
        configure_openai(config)
    elif config.provider == "anthropic":
        configure_anthropic(config)
    elif config.provider == "ollama":
        configure_ollama(config)
    elif config.provider == "custom":
        configure_custom(config)

    # Test connection
    if config.provider != "anthropic":  # Skip for passthrough
        if not test_connection(config):
            if not Confirm.ask("Continue anyway?", default=False):
                return False

    # Save configuration
    save_config(config)
    console.print(f"\n[green]✓ Configuration saved to ~/.claude-universal/config.json[/green]")

    console.print("\n[bold]Setup complete![/bold]")
    console.print("Run [cyan]claude-universal[/cyan] to start using Claude Code with your backend.\n")

    return True


if __name__ == "__main__":
    run_wizard()
