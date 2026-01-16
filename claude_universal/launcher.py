"""Main launcher for Claude Universal."""

import atexit
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from .config import config_exists, get_config_dir, load_config


def check_for_updates() -> Optional[str]:
    """Check if updates are available. Returns update message or None. Cached for 24h."""
    import json

    cache_file = get_config_dir() / ".update_cache"
    cache_duration = 86400  # 24 hours

    # Check cache
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                cache = json.load(f)
                if time.time() - cache.get("timestamp", 0) < cache_duration:
                    return cache.get("message")
        except Exception:
            pass

    # Quick GitHub API check (non-blocking with short timeout)
    try:
        import httpx

        from . import __version__

        # Get latest commit SHA from GitHub
        response = httpx.get(
            "https://api.github.com/repos/schwarztim/claude-azure/commits/main",
            timeout=2.0  # Fast timeout
        )

        if response.status_code == 200:
            latest_sha = response.json()["sha"][:7]

            # Compare with current version
            current_version = __version__
            message = None

            # If version doesn't contain the latest SHA, update available
            if latest_sha not in current_version:
                message = "\033[38;5;11m● Update available! Run: claude-azure --update\033[0m"

            # Cache result
            get_config_dir().mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump({"timestamp": time.time(), "message": message}, f)

            return message
    except Exception:
        # Silently fail - don't block startup
        pass

    return None


def find_free_port() -> int:
    """Find a free port to use for the proxy."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def find_claude_binary() -> Optional[str]:
    """Find the claude binary in PATH."""
    import shutil
    return shutil.which("claude")


def start_proxy(port: int, verbose: bool = False) -> subprocess.Popen:
    """Start the embedded proxy server."""
    # Get the path to our proxy module
    proxy_module = Path(__file__).parent / "proxy.py"

    # Start proxy as a subprocess
    env = os.environ.copy()

    # Load config and set environment for proxy
    config = load_config()

    if config.provider == "azure":
        env["OPENAI_API_KEY"] = config.azure.api_key
        env["OPENAI_BASE_URL"] = config.azure.endpoint
        env["AZURE_API_VERSION"] = config.azure.api_version
    elif config.provider == "openai":
        env["OPENAI_API_KEY"] = config.openai.api_key
        env["OPENAI_BASE_URL"] = config.openai.base_url
    elif config.provider == "ollama":
        env["OPENAI_API_KEY"] = "ollama"
        env["OPENAI_BASE_URL"] = f"{config.ollama.endpoint}/v1"
    elif config.provider == "custom":
        env["OPENAI_API_KEY"] = config.custom_api_key
        env["OPENAI_BASE_URL"] = config.custom_base_url

    # Model mapping
    env["BIG_MODEL"] = config.models.opus
    env["MIDDLE_MODEL"] = config.models.sonnet
    env["SMALL_MODEL"] = config.models.haiku

    # Proxy settings
    env["PORT"] = str(port)
    env["HOST"] = "127.0.0.1"
    env["LOG_LEVEL"] = "DEBUG" if verbose else config.proxy.log_level

    # Output handling - show logs if verbose, otherwise suppress
    if verbose:
        stdout = None  # Inherit from parent (shows in terminal)
        stderr = None
    else:
        stdout = subprocess.PIPE
        stderr = subprocess.PIPE

    # Start proxy
    if sys.platform == "win32":
        # Windows: use CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen(
            [sys.executable, str(proxy_module)],
            env=env,
            stdout=stdout,
            stderr=stderr,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    else:
        # Unix: use start_new_session
        proc = subprocess.Popen(
            [sys.executable, str(proxy_module)],
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )

    return proc


def wait_for_proxy(port: int, timeout: float = 10.0) -> bool:
    """Wait for the proxy to be ready."""
    import httpx

    start_time = time.time()
    url = f"http://127.0.0.1:{port}/health"

    while time.time() - start_time < timeout:
        try:
            response = httpx.get(url, timeout=1.0)
            if response.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.1)

    return False


def launch_claude(args: list[str], port: int) -> int:
    """Launch Claude Code with the proxy configured."""
    claude_binary = find_claude_binary()
    if not claude_binary:
        print("Error: 'claude' command not found. Please install Claude Code first.")
        print("  Visit: https://claude.ai/code")
        return 1

    # Set up environment - only set BASE_URL, let Claude use existing auth
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"
    # Remove any existing API key to avoid conflict - proxy doesn't validate anyway
    env.pop("ANTHROPIC_API_KEY", None)

    # Launch Claude Code
    try:
        result = subprocess.run([claude_binary] + args, env=env)
        return result.returncode
    except KeyboardInterrupt:
        return 0


def cleanup_proxy(proc: subprocess.Popen) -> None:
    """Clean up the proxy process."""
    if proc.poll() is None:  # Process still running
        if sys.platform == "win32":
            proc.terminate()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=5)


def main(args: Optional[list[str]] = None) -> int:
    """Main entry point for claude-azure."""
    if args is None:
        args = sys.argv[1:]

    # Handle special arguments
    if "--reconfigure" in args or "--setup" in args:
        from .wizard import run_wizard
        run_wizard(force=True)
        args = [a for a in args if a not in ("--reconfigure", "--setup")]
        if not args:
            return 0

    if "--version" in args:
        from . import __version__
        print(f"claude-azure {__version__}")
        return 0

    if "--update" in args:
        # Check for running sessions
        import psutil
        current_pid = os.getpid()
        running_count = 0
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['pid'] != current_pid and proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if 'claude-azure' in cmdline and '--update' not in cmdline:
                        running_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        if running_count > 0:
            print(
                f"\033[38;5;9m✗ Error: {running_count} "
                "claude-azure session(s) still running.\033[0m"
            )
            print("Please close all sessions before updating.")
            return 1

        print("Updating claude-azure from GitHub...")

        # Try pipx first, fall back to pip
        import shutil
        if shutil.which("pipx"):
            print("Using pipx...")
            result = subprocess.run(
                ["pipx", "install", "--force", "git+https://github.com/schwarztim/claude-azure.git"],
                capture_output=False
            )
        else:
            print("Using pip...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--force-reinstall", "git+https://github.com/schwarztim/claude-azure.git"],
                capture_output=False
            )

        # Update bundled MCPs after successful update
        if result.returncode == 0:
            print()
            print("Updating bundled MCPs...")
            from .mcp_setup import setup_bundled_mcps
            setup_bundled_mcps(verbose=True, force=True)

        return result.returncode

    if "--help" in args and len(args) == 1:
        print("Claude Universal - Use Claude Code with any AI backend")
        print()
        print("Usage: claude-azure [options] [claude-args...]")
        print()
        print("Options:")
        print("  --setup, --reconfigure  Run the setup wizard")
        print("  --update               Update to latest version from GitHub")
        print("  --verbose              Show proxy request logs")
        print("  --version              Show version")
        print("  --help                 Show this help")
        print()
        print("All other arguments are passed to Claude Code.")
        print()
        print("Configuration: ~/.claude-azure/config.json")
        return 0

    # Check for verbose flag
    verbose = "--verbose" in args
    if verbose:
        args = [a for a in args if a != "--verbose"]

    # Fast auto-update check (cached for 24h, 2s timeout)
    update_msg = check_for_updates()
    if update_msg:
        print(update_msg)
        print()

    # Check for configuration
    if not config_exists():
        print("Welcome to Claude Universal!")
        print("Let's set up your AI backend.\n")
        from .wizard import run_wizard
        if not run_wizard():
            return 1

        # Install bundled MCPs on first run
        print("\nSetting up web search capability...")
        from .mcp_setup import setup_bundled_mcps
        setup_bundled_mcps(verbose=True)

    # Load config and check provider
    config = load_config()
    if not config.provider:
        print("Error: No provider configured. Run 'claude-azure --setup'")
        return 1

    # Ensure bundled MCPs are installed (silent check, installs if missing)
    from .mcp_setup import is_mcp_installed, setup_bundled_mcps
    if not is_mcp_installed("web-search-mcp"):
        print("Installing web search MCP...")
        setup_bundled_mcps(verbose=False)

    # Check for Anthropic passthrough mode
    if config.provider == "anthropic":
        # Direct passthrough - just launch Claude with API key
        claude_binary = find_claude_binary()
        if not claude_binary:
            print("Error: 'claude' command not found.")
            return 1

        env = os.environ.copy()
        env["ANTHROPIC_API_KEY"] = config.anthropic.api_key

        result = subprocess.run([claude_binary] + args, env=env)
        return result.returncode

    # Find free port for proxy
    port = find_free_port()

    # Show compact Q logo
    print("\033[38;5;39m ◖Q◗\033[0m \033[38;5;250mClaude Universal\033[0m")

    # Start proxy
    print(f"Starting proxy on port {port}...")
    proxy_proc = start_proxy(port, verbose=verbose)

    # Register cleanup
    atexit.register(cleanup_proxy, proxy_proc)

    # Wait for proxy to be ready
    if not wait_for_proxy(port):
        print("Error: Proxy failed to start. Check logs.")
        cleanup_proxy(proxy_proc)
        return 1

    print("Proxy ready. Launching Claude Code...")
    print()

    # Launch Claude Code
    try:
        return launch_claude(args, port)
    finally:
        cleanup_proxy(proxy_proc)


if __name__ == "__main__":
    sys.exit(main())
