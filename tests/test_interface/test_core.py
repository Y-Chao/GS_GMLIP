"""Tests for the Interface class."""

from __future__ import annotations

import pytest
from ase import Atoms
from ase.constraints import FixAtoms

from gs_gmlip.interface import Interface


def _slab():
    # two-layer 2x2 H "slab" stand-in (cheap, no calculator needed)
    pos = [[x, y, z] for z in (0.0, 2.0) for x in (0.0, 2.0) for y in (0.0, 2.0)]
    return Atoms("H8", positions=pos, cell=[4, 4, 12], pbc=[True, True, False])


def test_init_bare_substrate():
    intf = Interface(substrate=_slab())
    assert len(intf.substrate) == 8
    assert len(intf.interface) == 8
    assert intf.adsList == []
    assert intf.clusterList == []
    assert intf.fix == []
    assert sorted(intf.relax) == list(range(8))


def test_init_with_adsorbate_classifies():
    slab = _slab()
    full = slab + Atoms("CO", positions=[[2.0, 2.0, 4.0], [2.0, 2.0, 5.13]])
    intf = Interface(substrate=slab, interface=full)
    assert intf.clusterList == []
    assert sorted(intf.adsList[0]) == [8, 9]


def test_init_fixlist_validation():
    with pytest.raises(ValueError):
        Interface(substrate=_slab(), fixlist=[99])


def test_init_reads_existing_fixatoms_constraint():
    slab = _slab()
    slab.set_constraint(FixAtoms(indices=[0, 1, 2, 3]))
    intf = Interface(substrate=slab)
    assert intf.fix == [0, 1, 2, 3]
    # relax is the complement
    assert intf.relax == [4, 5, 6, 7]


def test_repr_contains_interface():
    intf = Interface(substrate=_slab())
    assert "Interface" in repr(intf)


def test_fix_is_flat_int_list_from_fixlist():
    intf = Interface(substrate=_slab(), fixlist=[2, 0, 1])
    assert intf.fix == [0, 1, 2]  # sorted, deduped, plain ints
    assert all(isinstance(i, int) for i in intf.fix)


def test_fix_matches_between_construction_paths():
    # fixlist path and constraint path should yield identical flat fix lists
    from ase.constraints import FixAtoms as _Fix

    a = Interface(substrate=_slab(), fixlist=[0, 1, 2, 3])
    slab = _slab()
    slab.set_constraint(_Fix(indices=[0, 1, 2, 3]))
    b = Interface(substrate=slab)
    assert a.fix == b.fix == [0, 1, 2, 3]


def test_relaxlist_explicit():
    intf = Interface(substrate=_slab(), relaxlist=[4, 5])
    assert intf.relax == [4, 5]


def test_cached_properties_present_and_cached():
    slab = _slab()
    full = slab + Atoms("CO", positions=[[2.0, 2.0, 4.0], [2.0, 2.0, 5.13]])
    intf = Interface(substrate=slab, interface=full)
    bm1 = intf.bondmatrix
    assert bm1.shape == (10, 10)
    assert intf.bondmatrix is bm1  # cached: identical object on 2nd access
    assert isinstance(intf.possible_bond, dict)
    import numpy as np
    assert isinstance(intf.fingerprint, np.ndarray)


def test_reset_cache_clears_entries():
    intf = Interface(substrate=_slab())
    _ = intf.bondmatrix
    assert "bondmatrix" in intf.__dict__
    intf._reset_cache()
    assert "bondmatrix" not in intf.__dict__
