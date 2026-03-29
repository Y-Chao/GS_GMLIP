"""Tests for gs_gmlip.structure module."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111

from gs_gmlip.structure.atoms import SlabAtoms
from gs_gmlip.structure.bonds import BondData, generate_blmin, get_bond_matrix
from gs_gmlip.structure.composition import (
    CompositionConstraint,
    MolecularBlock,
    co2_block,
    hydrogen_block,
    water_block,
)
from gs_gmlip.structure.region import BoxRegion, SphereRegion


class TestSlabAtoms:
    def test_from_ase_atoms(self, cu111_slab):
        sa = SlabAtoms.from_atoms(cu111_slab)
        assert isinstance(sa, SlabAtoms)
        assert len(sa) == len(cu111_slab)

    def test_get_active_indices(self, cu111_slab):
        sa = SlabAtoms.from_atoms(cu111_slab)
        active = sa.get_active_indices()
        slab_idx = sa.get_slab_indices()
        assert len(active) + len(slab_idx) == len(sa)
        assert all(sa.get_tags()[i] > 0 for i in active)

    def test_tag_slab_by_height(self):
        slab = fcc111("Cu", size=(2, 2, 4), vacuum=10.0)
        sa = SlabAtoms.from_atoms(slab)
        z_mid = (sa.positions[:, 2].min() + sa.positions[:, 2].max()) / 2
        sa.tag_slab_by_height(z_threshold=z_mid)
        tags = sa.get_tags()
        assert 0 in tags
        assert 1 in tags


class TestRegion:
    def test_box_region_random_position(self, cu111_slab, rng):
        z_top = cu111_slab.positions[:, 2].max()
        region = BoxRegion(
            cell=cu111_slab.cell,
            z_min=z_top + 1.0,
            z_max=z_top + 5.0,
            pbc=cu111_slab.pbc,
        )
        pos = region.random_position(rng)
        assert pos.shape == (3,)
        assert pos[2] >= z_top + 1.0
        assert pos[2] <= z_top + 5.0

    def test_box_region_contains(self, box_region, cu111_slab):
        z_top = cu111_slab.positions[:, 2].max()
        assert box_region.contains(np.array([1.0, 1.0, z_top + 3.0]))
        assert not box_region.contains(np.array([1.0, 1.0, z_top - 1.0]))

    def test_sphere_region(self, rng):
        region = SphereRegion(center=np.array([5.0, 5.0, 10.0]), radius=3.0)
        pos = region.random_position(rng)
        dist = np.linalg.norm(pos - region.center)
        assert dist <= 3.0


class TestBonds:
    def test_bond_data(self):
        from ase.data import atomic_numbers

        bd = BondData()
        z_cu = atomic_numbers["Cu"]
        r = bd.get_max_bond_length(z_cu, z_cu, "Cu", "Cu")
        assert r > 0
        assert r < 5.0

    def test_generate_blmin(self):
        blmin = generate_blmin(["Cu", "H", "O"])
        # Keys are atomic number tuples
        assert (29, 1) in blmin or (1, 29) in blmin
        assert all(v > 0 for v in blmin.values())

    def test_bond_matrix(self, cu111_slab):
        bm = get_bond_matrix(cu111_slab)
        assert bm.shape == (len(cu111_slab), len(cu111_slab))


class TestComposition:
    def test_molecular_block(self):
        block = water_block(mu=-14.0)
        assert block.name == "H2O"
        assert block.n_atoms == 3
        assert "H" in block.symbols
        assert "O" in block.symbols

    def test_hydrogen_block(self):
        block = hydrogen_block()
        assert block.name == "H"
        assert block.n_atoms == 1

    def test_co2_block(self):
        block = co2_block()
        assert block.n_atoms == 3

    def test_composition_constraint(self):
        blocks = [water_block(), hydrogen_block()]
        cc = CompositionConstraint(
            blocks=blocks,
            min_count=1,
            max_count=4,
            element_pool={"H": (0, 8), "O": (0, 4)},
        )
        assert cc.min_count == 1
        assert cc.max_count == 4
