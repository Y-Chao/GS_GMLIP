"""Prepare RuO2(110) slab for water adsorption studies.

Creates a RuO2(110) surface slab — the most common facet for
oxygen evolution reaction (OER) catalysis.
"""

import numpy as np
from ase import Atoms
from ase.build import surface
from ase.constraints import FixAtoms
from ase.io import write
from ase.spacegroup import crystal


def make_ruo2_bulk() -> Atoms:
    """Create bulk RuO2 in rutile structure.

    Lattice parameters: a = 4.4919 Å, c = 3.1066 Å (rutile P42/mnm).
    """
    a = 4.4919
    c = 3.1066
    ruo2 = crystal(
        ["Ru", "O"],
        basis=[(0, 0, 0), (0.3056, 0.3056, 0)],
        spacegroup=136,  # P42/mnm
        cellpar=[a, a, c, 90, 90, 90],
    )
    return ruo2


def make_ruo2_110_slab(
    size: tuple = (2, 1, 4),
    vacuum: float = 15.0,
    fix_bottom: int = 2,
) -> Atoms:
    """Create a RuO2(110) slab.

    Parameters
    ----------
    size : tuple
        (a, b, layers) for the slab.
    vacuum : float
        Vacuum in Å.
    fix_bottom : int
        Number of bottom layers to fix.

    Returns
    -------
    slab : Atoms
        RuO2(110) slab with tags.
    """
    bulk = make_ruo2_bulk()
    slab = surface(bulk, (1, 1, 0), layers=size[2], vacuum=vacuum, periodic=True)

    # Repeat in x and y
    slab = slab.repeat((size[0], size[1], 1))

    # Fix bottom layers
    z = slab.positions[:, 2]
    z_unique = sorted(set(z.round(2)))
    if fix_bottom < len(z_unique):
        z_cutoff = z_unique[fix_bottom] - 0.1
    else:
        z_cutoff = z.min()

    fix_mask = z < z_cutoff
    slab.set_constraint(FixAtoms(mask=fix_mask))

    # Tag: 0 = slab, 1 = surface layer
    tags = [0 if fix_mask[i] else 1 for i in range(len(slab))]
    slab.set_tags(tags)

    return slab


if __name__ == "__main__":
    slab = make_ruo2_110_slab()
    write("ruo2_110_slab.xyz", slab)
    write("ruo2_110_slab.vasp", slab, format="vasp")

    n_ru = sum(1 for s in slab.symbols if s == "Ru")
    n_o = sum(1 for s in slab.symbols if s == "O")
    print(f"RuO2(110): {len(slab)} atoms ({n_ru} Ru, {n_o} O)")
    print(f"Cell: {slab.cell.lengths()}")
