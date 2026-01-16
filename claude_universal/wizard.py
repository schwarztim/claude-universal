"""Interactive setup wizard for Claude Universal."""

import getpass
import sys

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from .config import Config, config_exists, load_config, save_config

console = Console()


def masked_input(prompt_text: str, default: str = "") -> str:
    """Get password input with asterisk masking."""

    console.print(f"{prompt_text}", end="")
    if default:
        console.print(f" [dim]\\[{'*' * min(len(default), 8)}...][/dim]", end="")
    console.print(": ", end="")

    password = ""

    # Windows implementation
    if sys.platform == "win32":
        try:
            import msvcrt
            while True:
                ch = msvcrt.getwch()
                if ch in ('\r', '\n'):
                    print()  # New line
                    break
                elif ch == '\x03':  # Ctrl+C
                    raise KeyboardInterrupt
                elif ch in ('\b', '\x7f'):  # Backspace
                    if password:
                        password = password[:-1]
                        sys.stdout.write('\b \b')
                        sys.stdout.flush()
                elif ch >= ' ':  # Printable character
                    password += ch
                    sys.stdout.write('*')
                    sys.stdout.flush()
            return password if password else default
        except Exception:
            # Fallback to getpass
            value = getpass.getpass("")
            return value if value else default

    # Unix implementation
    else:
        try:
            import termios
            import tty

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                while True:
                    ch = sys.stdin.read(1)
                    if ch in ('\r', '\n'):
                        console.print()  # New line
                        break
                    elif ch == '\x7f' or ch == '\x08':  # Backspace
                        if password:
                            password = password[:-1]
                            sys.stdout.write('\b \b')
                            sys.stdout.flush()
                    elif ch == '\x03':  # Ctrl+C
                        raise KeyboardInterrupt
                    elif ch >= ' ':  # Printable character
                        password += ch
                        sys.stdout.write('*')
                        sys.stdout.flush()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

            return password if password else default
        except Exception:
            # Fallback to getpass
            value = getpass.getpass("")
            return value if value else default


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


def test_model(endpoint: str, api_key: str, model: str) -> tuple[bool, str]:
    """Test if a model/deployment works."""
    try:
        import httpx

        # Build URL for Azure OpenAI v1 endpoint
        base = endpoint.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/openai/v1"

        url = f"{base}/chat/completions"
        headers = {
            "api-key": api_key,
            "Content-Type": "application/json"
        }

        response = httpx.post(
            url,
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Say OK"}],
                "max_completion_tokens": 5
            },
            timeout=30.0
        )

        if response.status_code == 200:
            return True, "Model validated successfully"
        else:
            error = response.json().get("error", {}).get("message", response.text)
            return False, f"Error: {error}"
    except Exception as e:
        return False, f"Connection error: {e}"


def configure_azure(config: Config) -> None:
    """Configure Azure OpenAI settings."""
    console.print("\n[bold blue]Azure OpenAI Configuration[/bold blue]\n")

    console.print("[dim]Enter your Azure OpenAI resource endpoint[/dim]")
    console.print("[dim]Example: https://your-resource.openai.azure.com[/dim]\n")

    config.azure.endpoint = Prompt.ask(
        "Azure endpoint",
        default=config.azure.endpoint or "https://your-resource.openai.azure.com"
    )

    # API Key with asterisk masking
    config.azure.api_key = masked_input(
        "API Key",
        default=config.azure.api_key
    )

    # Model/deployment name
    console.print("\n[bold]Model Configuration[/bold]")
    console.print("[dim]Enter your Azure deployment name (e.g., gpt-4, gpt-4o)[/dim]\n")

    model_name = Prompt.ask(
        "Deployment name",
        default=config.models.sonnet or "gpt-4o"
    )

    # Validate the model
    console.print("\n[dim]Validating model...[/dim]")
    success, message = test_model(config.azure.endpoint, config.azure.api_key, model_name)

    if success:
        console.print(f"[green]✓ {message}[/green]")
        # Use same model for all tiers (can be customized later)
        config.models.opus = model_name
        config.models.sonnet = model_name
        config.models.haiku = model_name
    else:
        console.print(f"[red]✗ {message}[/red]")
        if Confirm.ask("Use this model anyway?", default=False):
            config.models.opus = model_name
            config.models.sonnet = model_name
            config.models.haiku = model_name
        else:
            return configure_azure(config)  # Retry

    # Ask if they want different models for different tiers
    if Confirm.ask("\nUse different models for Opus/Sonnet/Haiku?", default=False):
        console.print("\n[dim]Claude Code uses three model tiers:[/dim]")
        console.print("[dim]  Opus = most capable, Sonnet = balanced, Haiku = fast[/dim]\n")

        config.models.opus = Prompt.ask("Opus deployment", default=model_name)
        config.models.sonnet = Prompt.ask("Sonnet deployment", default=model_name)
        config.models.haiku = Prompt.ask("Haiku deployment", default=model_name)


def configure_openai(config: Config) -> None:
    """Configure OpenAI settings."""
    console.print("\n[bold green]OpenAI Configuration[/bold green]\n")

    config.openai.api_key = masked_input(
        "API Key",
        default=config.openai.api_key
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

    config.anthropic.api_key = masked_input(
        "API Key",
        default=config.anthropic.api_key
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

    config.custom_api_key = masked_input(
        "API Key (or press Enter for none)",
        default=config.custom_api_key or ""
    ) or "none"

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
            # Already tested during model validation
            console.print("[green]✓ Azure connection verified[/green]")
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

    # Test connection (skip for Azure since we already validated)
    if config.provider not in ("anthropic", "azure"):
        if not test_connection(config):
            if not Confirm.ask("Continue anyway?", default=False):
                return False

    # Save configuration
    save_config(config)
    console.print("\n[green]✓ Configuration saved to ~/.claude-azure/config.json[/green]")

    console.print("\n[bold]Setup complete![/bold]")
    console.print(
        "Run [cyan]claude-azure[/cyan] to start using Claude Code "
        "with your backend.\n"
    )

    return True


if __name__ == "__main__":
    run_wizard()
