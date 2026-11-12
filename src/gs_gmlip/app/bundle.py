"""Build a double-clickable macOS application that starts the app on this machine.

The bundle is a thin launcher, not a frozen copy: it runs ``gs_gmlip app``
with the Python environment it was built from, so it always uses the
installed package and its optional extras (MatGL, MACE, Packmol). Opening it
starts the server in the background and shows the app in the default
browser; opening it again while it runs only brings up a new browser tab.
The server is stopped with the Quit button inside the app.
"""

from __future__ import annotations

import plistlib
import shlex
import stat
import sys
from pathlib import Path

APP_NAME = "GS-GMLIP"
BUNDLE_ID = "ai.materialyze.gs-gmlip"
DEFAULT_PORT = 8080

_LAUNCHER = """#!/bin/bash
# Launcher written by `gs_gmlip make-app`; rebuild the app instead of editing.
PYTHON={python}
HOME_DIR="${{GS_GMLIP_HOME:-$HOME/.gs_gmlip}}"
mkdir -p "$HOME_DIR"
cd "$HOME_DIR" || exit 1
export PATH="$(dirname "$PYTHON"):/opt/homebrew/bin:/usr/local/bin:$PATH"
if [ ! -x "$PYTHON" ]; then
  osascript -e 'display alert "GS-GMLIP" message "The Python environment this app was built from is gone:\\n{python_text}\\n\\nRebuild the app with: gs_gmlip make-app"'
  exit 1
fi
exec "$PYTHON" -m gs_gmlip.cli app --port {port} --allow-quit >> "$HOME_DIR/app.log" 2>&1
"""


def build_macos_app(
    output: str | Path, python: str | Path | None = None, port: int = DEFAULT_PORT
) -> Path:
    """Write ``GS-GMLIP.app`` into ``output`` and return its path.

    Args:
        output: Directory that receives the bundle (e.g. ``~/Applications``).
        python: Interpreter the app runs with; the current one by default.
        port: Preferred port; the next free one is used if it is taken.

    Raises:
        RuntimeError: When not on macOS.
    """
    if sys.platform != "darwin":
        raise RuntimeError(
            "application bundles are only built on macOS; use `gs_gmlip app` elsewhere"
        )
    # Not resolved: a virtualenv's python is a symlink, and following it would lose the environment.
    interpreter = Path(python or sys.executable).expanduser().absolute()
    bundle = Path(output).expanduser().resolve() / f"{APP_NAME}.app"
    macos = bundle / "Contents" / "MacOS"
    macos.mkdir(parents=True, exist_ok=True)
    (bundle / "Contents" / "Resources").mkdir(exist_ok=True)

    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("gs-gmlip")
    except PackageNotFoundError:
        __version__ = "0"

    info = {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleExecutable": APP_NAME,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": str(__version__),
        "CFBundleVersion": str(__version__),
        "LSMinimumSystemVersion": "11.0",
        # A background launcher: no Dock icon that would look frozen. Quit from inside the app.
        "LSUIElement": True,
        "NSHighResolutionCapable": True,
    }
    (bundle / "Contents" / "Info.plist").write_bytes(plistlib.dumps(info))
    launcher = macos / APP_NAME
    launcher.write_text(
        _LAUNCHER.format(
            python=shlex.quote(str(interpreter)),
            python_text=str(interpreter),
            port=int(port),
        )
    )
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bundle
