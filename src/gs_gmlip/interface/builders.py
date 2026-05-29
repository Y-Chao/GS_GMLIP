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


def slab_from_bulk(
    bulk_structure: Atoms | Structure,
    miller_index: tuple[int, int, int],
    layer: int = 4,
    vacuum: float = 15.0,
    symmetry: bool = False,
) -> Atoms:
    """Build a slab from a bulk structure and Miller index, returned as ASE Atoms.

    Args:
        bulk_structure: Bulk cell (ASE Atoms or pymatgen Structure).
        miller_index: Miller index of the surface, e.g. (1, 1, 1).
        layer: Minimum number of atomic layers (slab thickness, in layers).
        vacuum: Vacuum thickness in Å added along the surface normal.
        symmetry: If True, require symmetric (both surfaces equivalent) slabs.
    """
    structure = _to_structure(bulk_structure)
    # pymatgen 2026.5.4 defaults SlabGenerator(primitive=True); pass
    # primitive=False here so this builds the conventional (non-primitive) slab.
    gen = SlabGenerator(
        structure,
        miller_index=miller_index,
        min_slab_size=layer,
        min_vacuum_size=vacuum,
        in_unit_planes=True,
        center_slab=True,
        primitive=False,
    )
    slabs = gen.get_slabs(symmetrize=symmetry)
    return AseAtomsAdaptor.get_atoms(slabs[0])


def primitive_slab_from_bulk(
    bulk_structure: Atoms | Structure,
    miller_index: tuple[int, int, int],
    layer: int = 4,
    vacuum: float = 15.0,
    symmetry: bool = False,
) -> Atoms:
    """Like slab_from_bulk but reduces the slab to its primitive surface cell."""
    structure = _to_structure(bulk_structure)
    gen = SlabGenerator(
        structure,
        miller_index=miller_index,
        min_slab_size=layer,
        min_vacuum_size=vacuum,
        in_unit_planes=True,
        center_slab=True,
        primitive=True,
    )
    slabs = gen.get_slabs(symmetrize=symmetry)
    return AseAtomsAdaptor.get_atoms(slabs[0])
