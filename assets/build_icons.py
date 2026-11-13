"""Regenerate every icon file from ``gm2-surf_icon.svg`` (macOS only).

    uv run python assets/build_icons.py

Writes ``assets/png/gm2-surf_icon_<size>.png``, ``assets/gm2-surf.icns`` and
the copies the app ships with (``src/gm2_surf/app/static/icon.svg``,
``icon.png`` and ``icon.icns``). The SVG is rendered with Quick Look, which
handles its masks and clip paths; ImageMagick's built-in renderer does not.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

ASSETS = Path(__file__).resolve().parent
SOURCE = ASSETS / "gm2-surf_icon.svg"
STATIC = ASSETS.parent / "src" / "gm2_surf" / "app" / "static"
PNG_SIZES = (16, 32, 64, 128, 256, 512, 1024)
# Names required by iconutil: icon_<points>x<points>[@2x].png
ICONSET = {
    "icon_16x16.png": 16,
    "icon_16x16@2x.png": 32,
    "icon_32x32.png": 32,
    "icon_32x32@2x.png": 64,
    "icon_128x128.png": 128,
    "icon_128x128@2x.png": 256,
    "icon_256x256.png": 256,
    "icon_256x256@2x.png": 512,
    "icon_512x512.png": 512,
    "icon_512x512@2x.png": 1024,
}


def _run(*command: str) -> None:
    subprocess.run(command, check=True, capture_output=True)


def main() -> None:
    """Render the SVG and write all derived files."""
    png_dir = ASSETS / "png"
    png_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        _run("qlmanage", "-t", "-s", "1024", "-o", tmp, str(SOURCE))
        master = Path(tmp) / f"{SOURCE.name}.png"
        for size in PNG_SIZES:
            _run(
                "sips",
                "-z",
                str(size),
                str(size),
                str(master),
                "--out",
                str(png_dir / f"gm2-surf_icon_{size}.png"),
            )
        iconset = Path(tmp) / "gm2-surf.iconset"
        iconset.mkdir()
        for name, size in ICONSET.items():
            shutil.copy(png_dir / f"gm2-surf_icon_{size}.png", iconset / name)
        _run(
            "iconutil", "-c", "icns", str(iconset), "-o", str(ASSETS / "gm2-surf.icns")
        )
    shutil.copy(SOURCE, STATIC / "icon.svg")
    shutil.copy(png_dir / "gm2-surf_icon_256.png", STATIC / "icon.png")
    shutil.copy(ASSETS / "gm2-surf.icns", STATIC / "icon.icns")


if __name__ == "__main__":
    main()
