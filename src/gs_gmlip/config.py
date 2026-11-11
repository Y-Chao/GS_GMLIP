"""User-level settings shared by all projects.

Everything lives in one directory, ``~/.gs_gmlip`` by default or the path in
the ``GS_GMLIP_HOME`` environment variable: the global molecule library and
the list of recently opened projects.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

HOME_ENV = "GS_GMLIP_HOME"
_RECENT_FILE = "recent_projects.json"
_RECENT_LIMIT = 12


def gs_home() -> Path:
    """The user-level GS-GMLIP directory (created on demand)."""
    home = Path(os.environ.get(HOME_ENV) or Path.home() / ".gs_gmlip").expanduser()
    home.mkdir(parents=True, exist_ok=True)
    return home


def recent_projects() -> list[str]:
    """Recently opened project directories, most recent first (existing ones only)."""
    path = gs_home() / _RECENT_FILE
    if not path.exists():
        return []
    try:
        entries = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    return [
        entry for entry in entries if isinstance(entry, str) and Path(entry).is_dir()
    ]


def remember_project(project: str | Path) -> None:
    """Move a project directory to the front of the recent list."""
    resolved = str(Path(project).expanduser().resolve())
    entries = [resolved] + [entry for entry in recent_projects() if entry != resolved]
    (gs_home() / _RECENT_FILE).write_text(json.dumps(entries[:_RECENT_LIMIT], indent=2))
