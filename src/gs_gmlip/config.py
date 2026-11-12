"""User-level settings shared by all projects.

Everything lives in one directory, ``~/.gs_gmlip`` by default or the path in
the ``GS_GMLIP_HOME`` environment variable: the global molecule library and
the list of recently opened projects.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
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


def folder_picker_available() -> bool:
    """Whether ``choose_folder`` can show a native dialog on this machine."""
    return sys.platform == "darwin"


def choose_folder(
    prompt: str = "Choose a project folder", start: str | Path | None = None
) -> str:
    """Ask the user for a folder with the native macOS dialog (it can also create one).

    The dialog belongs to the machine the app runs on, which is the user's
    own machine for a local app.

    Args:
        prompt: Text shown in the dialog.
        start: Folder the dialog opens in; the home directory by default.

    Returns:
        The chosen path, or an empty string if the user cancelled.

    Raises:
        RuntimeError: Where no native dialog is available.
    """
    if not folder_picker_available():
        raise RuntimeError(
            "the folder dialog is only available on macOS; type the path instead"
        )
    location = Path(start).expanduser() if start else Path.home()
    if not location.is_dir():
        location = Path.home()
    script = (
        "on run argv\n"
        "  activate\n"
        "  return POSIX path of (choose folder with prompt (item 1 of argv) "
        "default location (POSIX file (item 2 of argv)))\n"
        "end run"
    )
    done = subprocess.run(  # noqa: S603 - fixed program, arguments passed as argv, not interpolated
        ["/usr/bin/osascript", "-e", script, prompt, str(location)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if done.returncode != 0:  # -128: the user pressed Cancel
        return ""
    return done.stdout.strip().rstrip("/")
