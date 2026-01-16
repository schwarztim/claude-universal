"""MCP setup for Claude Universal.

Installs bundled MCPs like web-search to provide functionality
that backend models (Azure, OpenAI, etc.) don't have natively.
"""

import json
import shutil
import subprocess
from pathlib import Path


def get_mcp_dir() -> Path:
    """Get the MCP installation directory."""
    return Path.home() / ".claude-universal" / "mcps"


def get_user_mcps_path() -> Path:
    """Get the path to user-mcps.json."""
    return Path.home() / ".claude" / "user-mcps.json"


def is_mcp_installed(name: str) -> bool:
    """Check if an MCP is already installed."""
    mcp_path = get_mcp_dir() / name
    return (mcp_path / "dist" / "index.js").exists()


def is_mcp_registered(name: str) -> bool:
    """Check if an MCP is registered in user-mcps.json."""
    user_mcps = get_user_mcps_path()
    if not user_mcps.exists():
        return False

    try:
        with open(user_mcps) as f:
            config = json.load(f)
            return name in config.get("mcpServers", {})
    except Exception:
        return False


def install_web_search_mcp(verbose: bool = False) -> bool:
    """Install the web-search MCP.

    Returns True if installed successfully, False otherwise.
    """
    mcp_dir = get_mcp_dir()
    mcp_path = mcp_dir / "web-search-mcp"

    try:
        # Create MCP directory
        mcp_dir.mkdir(parents=True, exist_ok=True)

        # Clone or update the repo
        if mcp_path.exists():
            if verbose:
                print("Updating web-search MCP...")
            result = subprocess.run(
                ["git", "pull"],
                cwd=mcp_path,
                capture_output=not verbose,
                text=True,
            )
            if result.returncode != 0:
                if verbose:
                    print(f"Git pull failed: {result.stderr}")
                # Try fresh clone
                shutil.rmtree(mcp_path, ignore_errors=True)
                return install_web_search_mcp(verbose)
        else:
            if verbose:
                print("Installing web-search MCP...")
            result = subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/schwarztim/web-search-mcp.git",
                    str(mcp_path),
                ],
                capture_output=not verbose,
                text=True,
            )
            if result.returncode != 0:
                if verbose:
                    print(f"Git clone failed: {result.stderr}")
                return False

        # Install dependencies
        if verbose:
            print("Installing dependencies...")
        result = subprocess.run(
            ["npm", "install"],
            cwd=mcp_path,
            capture_output=not verbose,
            text=True,
        )
        if result.returncode != 0:
            if verbose:
                print(f"npm install failed: {result.stderr}")
            return False

        # Build
        if verbose:
            print("Building...")
        result = subprocess.run(
            ["npm", "run", "build"],
            cwd=mcp_path,
            capture_output=not verbose,
            text=True,
        )
        if result.returncode != 0:
            if verbose:
                print(f"npm build failed: {result.stderr}")
            return False

        return True

    except Exception as e:
        if verbose:
            print(f"Error installing web-search MCP: {e}")
        return False


def register_web_search_mcp(verbose: bool = False) -> bool:
    """Register the web-search MCP in user-mcps.json.

    Returns True if registered successfully, False otherwise.
    """
    mcp_path = get_mcp_dir() / "web-search-mcp"
    index_path = mcp_path / "dist" / "index.js"

    if not index_path.exists():
        if verbose:
            print("web-search MCP not built, cannot register")
        return False

    user_mcps = get_user_mcps_path()

    try:
        # Ensure .claude directory exists
        user_mcps.parent.mkdir(parents=True, exist_ok=True)

        # Load existing config or create new
        if user_mcps.exists():
            with open(user_mcps) as f:
                config = json.load(f)
        else:
            config = {"mcpServers": {}}

        # Ensure mcpServers exists
        if "mcpServers" not in config:
            config["mcpServers"] = {}

        # Add web-search MCP
        config["mcpServers"]["web-search"] = {
            "command": "node",
            "args": [str(index_path)],
        }

        # Write back
        with open(user_mcps, "w") as f:
            json.dump(config, f, indent=2)

        if verbose:
            print(f"Registered web-search MCP in {user_mcps}")

        return True

    except Exception as e:
        if verbose:
            print(f"Error registering web-search MCP: {e}")
        return False


def setup_bundled_mcps(verbose: bool = False, force: bool = False) -> None:
    """Set up all bundled MCPs.

    Called on first run and on --update.
    """
    # Check if npm is available
    if not shutil.which("npm"):
        if verbose:
            print("npm not found, skipping MCP setup")
        return

    # Check if git is available
    if not shutil.which("git"):
        if verbose:
            print("git not found, skipping MCP setup")
        return

    # Install web-search MCP if not installed or force update
    if force or not is_mcp_installed("web-search-mcp"):
        success = install_web_search_mcp(verbose)
        if success:
            if verbose:
                print("web-search MCP installed successfully")
        else:
            if verbose:
                print("Failed to install web-search MCP")
            return

    # Register if not registered
    if not is_mcp_registered("web-search"):
        success = register_web_search_mcp(verbose)
        if success:
            if verbose:
                print("web-search MCP registered")
            print(
                "\033[38;5;39m●\033[0m Web search MCP installed. "
                "Your model can now search the web!"
            )
    elif verbose:
        print("web-search MCP already registered")
