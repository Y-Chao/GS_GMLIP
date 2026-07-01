"""Shared test fixtures: disk structures + ASE-generated ones."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from ase import Atoms
from ase.build import fcc111
from ase.io import read

DATA = Path(__file__).parent / "data"


@pytest.fixture
def cu100_slab() -> Atoms:
    """Cu(100) 4-layer slab from disk (144 atoms, pbc=[T,T,F])."""
    return read(str(DATA / "Cu_fcc100.xyz"))


@pytest.fixture
def cocn_sheet() -> Atoms:
    """Co-N4 single-atom catalyst on graphene (159 atoms, pbc=[T,T,T])."""
    return read(str(DATA / "gn_Co_N4.xyz"))


@pytest.fixture
def pt111_slab() -> Atoms:
    """Pt(111) 4x4x6 slab built with ase.build (96 atoms, 6 layers)."""
    return fcc111("Pt", size=(4, 4, 6), a=3.92, vacuum=10.0)


@pytest.fixture
def co_molecule() -> Atoms:
    """A CO molecule oriented with C below O (zero-translated; consumers translate as needed)."""
    return Atoms("CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.13]])


@pytest.fixture
def cu100_with_co(cu100_slab, co_molecule) -> Atoms:
    """Cu(100) slab + one CO placed ~2 Å above the top Cu layer."""
    top = cu100_slab.get_positions()[:, 2].max()
    co = co_molecule.copy()
    co.translate([5.0, 5.0, top + 2.0])
    return cu100_slab + co


@pytest.fixture
def cu100_with_two_co(cu100_slab, co_molecule) -> Atoms:
    """Cu(100) slab + two CO molecules at different (x,y) above the top layer."""
    top = cu100_slab.get_positions()[:, 2].max()
    co1 = co_molecule.copy()
    co1.translate([0.0, 0.0, top + 2.0])
    co2 = co_molecule.copy()
    co2.translate([5.0, 5.0, top + 2.0])
    return cu100_slab + co1 + co2


@pytest.fixture
def cu100_with_ten_co(cu100_slab, co_molecule) -> Atoms:
    """Cu(100) slab + ten CO molecules at different (x,y) above the top layer."""
    top = cu100_slab.get_positions()[:, 2].max()
    atoms = cu100_slab.copy()
    top_site = [
        atoms[i] for i in range(len(atoms)) if atoms.positions[i, 2] > top - 0.2
    ]
    random.shuffle(top_site)
    for i in range(10):
        co = co_molecule.copy()
        co.translate([top_site[i][0], top_site[i][1], top + 2.0])
        atoms += co
    return atoms


@pytest.fixture
def cu100_with_ten_cu_cluster(cu100_slab) -> Atoms:
    """Cu(100) slab + ten Cu atoms in a cluster above the top layer."""
    top = cu100_slab.get_positions()[:, 2].max()
    atoms = cu100_slab.copy()
    from ase.cluster import Icosahedron

    cu_cluster = Icosahedron("Cu", 2)  # 13-atom Cu cluster
    cu_cluster.translate([5.0, 5.0, top + 2.0])
    atoms += cu_cluster
    return atoms
