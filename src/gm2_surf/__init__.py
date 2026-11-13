"""GM2-Surf: global minimum search with global machine-learning interatomic potentials, for surfaces."""

from __future__ import annotations

import subprocess
from pathlib import Path


def get_git_revision() -> str:
    """Get the current git revision (commit hash)."""
    import gm2_surf

    try:
        dir_path = Path(gm2_surf.__file__).parent
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
