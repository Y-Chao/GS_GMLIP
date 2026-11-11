"""Turn an ``Interface`` into a JSON scene for the 3Dmol surface viewer.

This module is plain Python (no NiceGUI import), so the scene logic is unit
testable. The scene encodes what a generic molecule viewer does not know
about a surface system: in-plane periodic images, the cell, substrate versus
adsorbate roles, frozen atoms, adsorbate bonds, and adsorption-site markers.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
from ase.data import covalent_radii
from ase.data.colors import jmol_colors

from gs_gmlip.interface.core import Interface
from gs_gmlip.interface.operations import AdsorptionSite

SITE_COLORS = {"ontop": "#d62728", "bridge": "#1f77b4", "hollow": "#2ca02c"}
# Fixed atoms are blended towards this colour so constraints are visible.
_FIXED_TINT = np.array([0.35, 0.35, 0.35])
_FIXED_BLEND = 0.55
# Two adsorbate atoms are drawn bonded below this multiple of the summed covalent radii.
_BOND_SCALE = 1.2


def _hex(rgb: np.ndarray) -> str:
    red, green, blue = (int(round(255 * float(np.clip(x, 0, 1)))) for x in rgb)
    return f"#{red:02x}{green:02x}{blue:02x}"


def atom_role(interface: Interface, index: int) -> str:
    """``"substrate"``, ``"cluster"``, or ``"adsorbate"`` for an interface atom index."""
    if index < len(interface.substrate):
        return "substrate"
    if any(index in group for group in interface.clusterList):
        return "cluster"
    return "adsorbate"


def build_scene(
    interface: Interface,
    selection: Iterable[int] = (),
    sites: Sequence[AdsorptionSite] = (),
    repeat: int = 0,
) -> dict[str, Any]:
    """Build the viewer scene for a surface system.

    Args:
        interface: The structure to draw.
        selection: Interface indices to highlight.
        sites: Adsorption sites to draw as clickable markers.
        repeat: Number of ghost periodic images on each side along a and b
            (0 draws the cell only). Images are never added along the normal.

    Returns:
        A JSON-serializable dict with keys ``atoms``, ``ghosts``, ``bonds``,
        ``cell``, ``sites``, and ``center``. Coordinates are in Å.
    """
    atoms = interface.interface
    n_substrate = len(interface.substrate)
    positions = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    symbols = atoms.get_chemical_symbols()
    cell = np.asarray(atoms.cell)
    fixed = set(interface.fix)
    selected = {int(i) for i in selection}

    atom_items = []
    for index, (symbol, number, position) in enumerate(
        zip(symbols, numbers, positions)
    ):
        color = np.asarray(jmol_colors[number], dtype=float)
        is_fixed = index in fixed
        if is_fixed:
            color = (1 - _FIXED_BLEND) * color + _FIXED_BLEND * _FIXED_TINT
        atom_items.append(
            {
                "index": index,
                "elem": symbol,
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]),
                "color": _hex(color),
                "radius": float(covalent_radii[number]),
                "role": atom_role(interface, index),
                "fixed": is_fixed,
                "selected": index in selected,
            }
        )

    ghosts = []
    if repeat > 0 and cell.any():
        for shift_a in range(-repeat, repeat + 1):
            for shift_b in range(-repeat, repeat + 1):
                if shift_a == 0 and shift_b == 0:
                    continue
                offset = shift_a * cell[0] + shift_b * cell[1]
                for item in atom_items:
                    ghosts.append(
                        {
                            "elem": item["elem"],
                            "x": item["x"] + float(offset[0]),
                            "y": item["y"] + float(offset[1]),
                            "z": item["z"] + float(offset[2]),
                            "color": item["color"],
                            "radius": item["radius"],
                        }
                    )

    bonds = []
    appended = range(n_substrate, len(atoms))
    for i in appended:
        for j in range(i + 1, len(atoms)):
            cutoff = _BOND_SCALE * (
                covalent_radii[numbers[i]] + covalent_radii[numbers[j]]
            )
            if atoms.get_distance(i, j, mic=False) <= cutoff:
                bonds.append([i, j])

    site_items = [
        {
            "index": index,
            "kind": site.kind,
            "x": site.coords[0],
            "y": site.coords[1],
            "z": site.coords[2],
            "color": SITE_COLORS.get(site.kind, "#888888"),
        }
        for index, site in enumerate(sites)
    ]

    center = cell.sum(axis=0) / 2.0 if cell.any() else positions.mean(axis=0)
    if len(positions):
        center[2] = float((positions[:, 2].min() + positions[:, 2].max()) / 2.0)

    return {
        "atoms": atom_items,
        "ghosts": ghosts,
        "bonds": bonds,
        "cell": cell.tolist(),
        "sites": site_items,
        "center": [float(x) for x in center],
    }
