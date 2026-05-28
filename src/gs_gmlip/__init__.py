"""Global minimum search with Machine Leaning Interatomic Potentials (GS_GMLIP)."""

from __future__ import annotations
import subprocess
from pathlib import Path

LINE_WIDTH = 120
PADDING_CHARACTER = "="
TERMINATE_CHARACTER = "|"


def get_git_revision() -> str:
    """Get the current git revision (commit hash)."""
    import gs_gmlip

    try:
        dir_path = Path(gs_gmlip.__file__).parent
        version_string = (
            subprocess.check_output(
                ["git", f"--git-dir={dir_path}/.git", "rev-parse", "--short", "HEAD"]
            )
            .decode("ascii")
            .strip()
        )
    except subprocess.CalledProcessError:
        version_string = "unknown"
    except BlockingIOError:
        version_string = "unknown"
    except FileNotFoundError:
        version_string = "unknown"

    return version_string
