"""Pure slab-construction functions backed by pymatgen's SlabGenerator."""

from __future__ import annotations

from ase import Atoms
from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor


def _to_structure(bulk_structure: Atoms | Structure) -> Structure:
    if isinstance(bulk_structure, Structure):
        return bulk_structure
    return AseAtomsAdaptor.get_structure(bulk_structure)


def _build_slab(
    bulk_structure: Atoms | Structure,
    miller_index: tuple[int, int, int],
    layer: int,
    vacuum: float,
    symmetry: bool,
    primitive: bool,
) -> Atoms:
    """Build a slab via pymatgen SlabGenerator and return the first termination as ASE Atoms.

    Only the first slab returned by get_slabs (the lowest-energy / fewest-broken-
    bonds termination) is used.
    """
    structure = _to_structure(bulk_structure)
    gen = SlabGenerator(
        structure,
        miller_index=miller_index,
        min_slab_size=layer,
        min_vacuum_size=vacuum,
        in_unit_planes=True,
        center_slab=True,
        primitive=primitive,
    )
    slabs = gen.get_slabs(symmetrize=symmetry)
    if not slabs:
        raise ValueError(
            f"No slab generated for miller_index={miller_index} "
            f"(symmetry={symmetry}); try symmetry=False or adjust layer/vacuum."
        )
    return AseAtomsAdaptor.get_atoms(slabs[0])


def slab_from_bulk(
    bulk_structure: Atoms | Structure,
    miller_index: tuple[int, int, int],
    layer: int = 4,
    vacuum: float = 15.0,
    symmetry: bool = False,
) -> Atoms:
    """Build a conventional-cell slab from a bulk structure and Miller index.

    Args:
        bulk_structure: Bulk cell (ASE Atoms or pymatgen Structure).
        miller_index: Miller index of the surface, e.g. (1, 1, 1).
        layer: Minimum number of atomic layers (slab thickness, in layers).
        vacuum: Vacuum thickness in Å added along the surface normal.
        symmetry: If True, require symmetric (both surfaces equivalent) slabs.

    Returns:
        The first (lowest-energy) termination as ASE Atoms.
    """
    return _build_slab(
        bulk_structure, miller_index, layer, vacuum, symmetry, primitive=False
    )


def primitive_slab_from_bulk(
    bulk_structure: Atoms | Structure,
    miller_index: tuple[int, int, int],
    layer: int = 4,
    vacuum: float = 15.0,
    symmetry: bool = False,
) -> Atoms:
    """Build a primitive-cell slab from a bulk structure and Miller index.

    Same arguments as slab_from_bulk, but the slab is reduced to its primitive
    surface cell. Returns the first (lowest-energy) termination as ASE Atoms.
    """
    return _build_slab(
        bulk_structure, miller_index, layer, vacuum, symmetry, primitive=True
    )
