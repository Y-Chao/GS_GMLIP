"""Tests for the Interface class."""

from __future__ import annotations

import numpy as np
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
    assert sorted(np.concatenate(intf.fix).tolist()) == [0, 1, 2, 3]


def test_repr_contains_interface():
    intf = Interface(substrate=_slab())
    assert "Interface" in repr(intf)
