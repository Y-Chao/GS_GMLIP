"""Tests for gs_gmlip.interface.builders."""

from __future__ import annotations

from ase import Atoms
from ase.build import bulk
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from gs_gmlip.interface.builders import primitive_slab_from_bulk, slab_from_bulk


def test_slab_from_bulk_returns_atoms_with_vacuum():
    pt = bulk("Pt", "fcc", a=3.92)
    slab = slab_from_bulk(pt, (1, 1, 1), layer=4, vacuum=15.0)
    assert isinstance(slab, Atoms)
    assert len(slab) > 0
    zs = slab.get_positions()[:, 2]
    assert slab.cell[2, 2] > (zs.max() - zs.min()) + 10.0


def test_primitive_slab_not_larger_than_conventional():
    pt = bulk("Pt", "fcc", a=3.92)
    slab = slab_from_bulk(pt, (1, 1, 1), layer=4, vacuum=15.0)
    prim = primitive_slab_from_bulk(pt, (1, 1, 1), layer=4, vacuum=15.0)
    assert len(prim) <= len(slab)


def test_accepts_pymatgen_structure_directly():
    # exercises _to_structure pass-through branch
    pt_struct = AseAtomsAdaptor.get_structure(bulk("Pt", "fcc", a=3.92))
    assert isinstance(pt_struct, Structure)
    slab = slab_from_bulk(pt_struct, (1, 1, 1), layer=4, vacuum=15.0)
    assert isinstance(slab, Atoms)
    assert len(slab) > 0
