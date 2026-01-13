"""Claude Universal - Use Claude Code with any AI backend."""

import subprocess
from pathlib import Path

__author__ = "Tim Schwarz"

# Get git commit SHA if available
def _get_version():
    base_version = "0.1.0"
    try:
        # Try to get git commit SHA
        git_dir = Path(__file__).parent.parent / ".git"
        if git_dir.exists():
            result = subprocess.run(
                ["git", "rev-parse", "--short=7", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
                timeout=1
            )
            if result.returncode == 0:
                sha = result.stdout.strip()
                return f"{base_version}+{sha}"
    except Exception:
        pass
    return base_version

__version__ = _get_version()
