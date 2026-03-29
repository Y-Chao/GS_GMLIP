"""Prepare Cu slab models for Cu/CSHO system.

Creates Cu(111) and Cu(100) surface slabs for studying
CO2, H2O, SO2, and HCOOH adsorption and surface reconstruction
under working conditions.
"""

from ase.build import fcc111, fcc100, add_adsorbate, molecule
from ase.constraints import FixAtoms
from ase.io import write


def make_cu111_slab(
    size: tuple = (3, 3, 4),
    vacuum: float = 15.0,
    fix_bottom: int = 2,
) -> "Atoms":
    """Create a Cu(111) slab.

    Parameters
    ----------
    size : tuple
        (a, b, layers) repeat.
    vacuum : float
        Vacuum in Å.
    fix_bottom : int
        Number of bottom layers to fix.

    Returns
    -------
    slab : Atoms
        Cu(111) slab with tags: 0=fixed slab, 1=relaxable surface.
    """
    slab = fcc111("Cu", size=size, vacuum=vacuum, periodic=True)

    # Fix bottom layers
    z = slab.positions[:, 2]
    z_sorted = sorted(set(z.round(2)))
    z_cutoff = (
        z_sorted[fix_bottom - 1] + 0.1 if fix_bottom <= len(z_sorted) else z.min()
    )
    fix_mask = z <= z_cutoff
    slab.set_constraint(FixAtoms(mask=fix_mask))

    # Tag: 0 = slab (fixed), 1 = surface Cu (relaxable)
    tags = [0 if fix_mask[i] else 1 for i in range(len(slab))]
    slab.set_tags(tags)

    return slab


def make_cu100_slab(
    size: tuple = (3, 3, 4),
    vacuum: float = 15.0,
    fix_bottom: int = 2,
) -> "Atoms":
    """Create a Cu(100) slab."""
    slab = fcc100("Cu", size=size, vacuum=vacuum, periodic=True)

    z = slab.positions[:, 2]
    z_sorted = sorted(set(z.round(2)))
    z_cutoff = (
        z_sorted[fix_bottom - 1] + 0.1 if fix_bottom <= len(z_sorted) else z.min()
    )
    fix_mask = z <= z_cutoff
    slab.set_constraint(FixAtoms(mask=fix_mask))

    tags = [0 if fix_mask[i] else 1 for i in range(len(slab))]
    slab.set_tags(tags)

    return slab


if __name__ == "__main__":
    # Generate slabs
    cu111 = make_cu111_slab()
    cu100 = make_cu100_slab()

    write("cu111_slab.xyz", cu111)
    write("cu100_slab.xyz", cu100)
    write("cu111_slab.vasp", cu111, format="vasp")
    write("cu100_slab.vasp", cu100, format="vasp")

    print(f"Cu(111): {len(cu111)} atoms, cell = {cu111.cell.lengths()}")
    print(f"Cu(100): {len(cu100)} atoms, cell = {cu100.cell.lengths()}")
