"""Turn an ``Interface`` into a JSON scene for the JavaScript viewer.

Plain Python, no web framework. The scene carries what a generic molecule
viewer does not know about a surface system: the cell, substrate versus
adsorbate roles, frozen atoms, bonds that do not cross periodic boundaries,
extra constraints, and adsorption-site markers. Periodic images, selection
halos, and all styling are the browser's job.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
from ase.data import atomic_numbers, covalent_radii
from ase.data.colors import jmol_colors
from ase.neighborlist import neighbor_list

from gm2_surf.interface.core import Interface
from gm2_surf.interface.operations import (
    AdsorptionSite,
    describe_constraints,
    list_items,
)

SITE_COLORS = {"ontop": "#ff5a5a", "bridge": "#4aa8ff", "hollow": "#46d37a"}

# Element colours: Jmol/CPK with the light carbon, hydrogen, and nitrogen of the
# PyMOL-like "paton"/"pmol" presets of xyzrender, which read well on black.
_COLOR_OVERRIDES = {"C": "#d9d9d9", "H": "#fafafa", "N": "#7f7fbf"}

# Frozen atoms are blended towards this colour so constraints are visible.
_FIXED_TINT = np.array([0.16, 0.17, 0.2])
_FIXED_BLEND = 0.5
# Two atoms are bonded below this multiple of the summed covalent radii.
_BOND_SCALE = 1.12


def _hex(rgb: np.ndarray) -> str:
    red, green, blue = (int(round(255 * float(np.clip(x, 0, 1)))) for x in rgb)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _rgb(color: str) -> np.ndarray:
    return np.array([int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)])


def element_color(symbol: str) -> str:
    """Display colour of an element as ``#rrggbb``."""
    if symbol in _COLOR_OVERRIDES:
        return _COLOR_OVERRIDES[symbol]
    return _hex(np.asarray(jmol_colors[atomic_numbers[symbol]], dtype=float))


def atom_role(interface: Interface, index: int) -> str:
    """``"substrate"``, ``"cluster"``, or ``"adsorbate"`` for an interface atom index."""
    if index < len(interface.substrate):
        return "substrate"
    if any(index in group for group in interface.clusterList):
        return "cluster"
    return "adsorbate"


def find_bonds(interface: Interface) -> list[list[int]]:
    """Bonded atom pairs ``[i, j]`` with ``i < j``, never across a periodic boundary."""
    atoms = interface.interface.copy()
    atoms.set_pbc(False)
    cutoffs = [_BOND_SCALE * covalent_radii[number] for number in atoms.numbers]
    first, second = neighbor_list("ij", atoms, cutoffs)
    return sorted([int(i), int(j)] for i, j in zip(first, second) if i < j)


def build_scene(
    interface: Interface,
    sites: Sequence[AdsorptionSite] = (),
    excluded: Iterable[str] = (),
) -> dict[str, Any]:
    """Build the viewer scene for a structure.

    Args:
        interface: The structure to draw.
        sites: Adsorption sites to draw as clickable markers.
        excluded: Item keys (see ``operations.list_items``) the user switched
            off: their atoms are flagged ``excluded`` (drawn faded) and their
            constraints are left out.

    Returns:
        A JSON-serializable dict with keys ``atoms``, ``bonds``, ``cell``,
        ``sites``, and ``constraints``. Coordinates are in Å.
    """
    atoms = interface.interface
    positions = atoms.get_positions()
    fixed = set(interface.fix)
    off = set(excluded)
    items = {item["key"]: item for item in list_items(interface)} if off else {}
    faded = {
        i
        for key in off
        if key.startswith(("ads:", "cluster:")) and key in items
        for i in items[key]["atoms"]
    }
    if "fixed" in off:
        fixed = set()

    atom_items = []
    for index, (symbol, number, position) in enumerate(
        zip(atoms.get_chemical_symbols(), atoms.numbers, positions)
    ):
        color = element_color(symbol)
        is_fixed = index in fixed
        if is_fixed:
            color = _hex((1 - _FIXED_BLEND) * _rgb(color) + _FIXED_BLEND * _FIXED_TINT)
        atom_items.append(
            {
                "index": index,
                "elem": symbol,
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]),
                "color": color,
                "radius": float(covalent_radii[number]),
                "role": atom_role(interface, index),
                "fixed": is_fixed,
                "excluded": index in faded,
            }
        )

    summary = describe_constraints(interface)
    masses = atoms.get_masses()
    heights = sorted({round(wall["height"], 6) for wall in summary["walls"]})
    off_heights = {
        heights[int(k.split(":")[1])]
        for k in off
        if k.startswith("wall:") and k in items
    }
    off_bonds = {int(k.split(":")[1]) for k in off if k.startswith("bond:")}
    off_com = {int(k.split(":")[1]) for k in off if k.startswith("com:")}
    summary = {
        "bonds": [
            pair for n, pair in enumerate(summary["bonds"]) if n not in off_bonds
        ],
        "com": [group for n, group in enumerate(summary["com"]) if n not in off_com],
        "walls": [
            w for w in summary["walls"] if round(w["height"], 6) not in off_heights
        ],
    }
    centers = []
    for group in summary["com"]:
        center = (positions[group] * masses[group, None]).sum(axis=0) / masses[
            group
        ].sum()
        centers.append({"indices": group, "center": [float(x) for x in center]})

    return {
        "atoms": atom_items,
        "bonds": find_bonds(interface),
        "cell": np.asarray(atoms.cell).tolist(),
        "sites": [
            {
                "index": index,
                "kind": site.kind,
                "x": site.coords[0],
                "y": site.coords[1],
                "z": site.coords[2],
                "color": SITE_COLORS.get(site.kind, "#888888"),
            }
            for index, site in enumerate(sites)
        ],
        "constraints": {
            "bonds": summary["bonds"],
            "com": centers,
            "walls": summary["walls"],
        },
    }
