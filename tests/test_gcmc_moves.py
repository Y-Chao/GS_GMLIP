"""Tests for GCMC move classes."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from gs_gmlip.interface.core import Interface
from gs_gmlip.search.gcmc.moves import DeleteMove, DisplaceMove, InsertMove, SwapMove
from gs_gmlip.search.region import BoxRegion


class TestInsertMove:
    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    @pytest.fixture
    def region(self):
        return BoxRegion(
            cell=np.array([[10, 0, 0], [0, 10, 0], [0, 0, 20]]),
            z_min=15.0,
            z_max=18.0,
        )

    @pytest.fixture
    def move(self, region):
        return InsertMove(
            blocks=[Atoms("O", positions=[[0, 0, 0]])],
            block_names=["O"],
            chemical_potentials={"O": -5.0},
            region=region,
        )

    def test_insert_adds_atom(self, cu100_slab, move, rng):
        iface = Interface(substrate=cu100_slab)
        iface.fix_substrate(layers=2)
        n_before = len(iface.interface)

        new_iface, log_ratio, info = move.propose(iface, rng)
        assert new_iface is not None
        assert len(new_iface.interface) == n_before + 1
        assert info["move"] == "insert"
        assert info["species"] == "O"

    def test_insert_log_ratio_is_finite(self, cu100_slab, move, rng):
        iface = Interface(substrate=cu100_slab)
        _, log_ratio, _ = move.propose(iface, rng)
        assert np.isfinite(log_ratio)


class TestDeleteMove:
    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    def test_delete_removes_group(self, cu100_slab, rng):
        iface = Interface(substrate=cu100_slab)
        # Add a single O atom
        o_atom = Atoms("O", positions=[[5.0, 5.0, 18.0]])
        iface.add_adsorbate(o_atom)
        n_before = len(iface.interface)
        assert len(iface.adsList) == 1

        move = DeleteMove(chemical_potentials={"O": -5.0})
        new_iface, log_ratio, info = move.propose(iface, rng)

        assert new_iface is not None
        assert len(new_iface.interface) == n_before - 1
        assert info["move"] == "delete"

    def test_delete_empty_returns_none(self, cu100_slab, rng):
        iface = Interface(substrate=cu100_slab)
        move = DeleteMove()
        new_iface, _, info = move.propose(iface, rng)
        assert new_iface is None
        assert info["reason"] == "no_adsorbates"


class TestDisplaceMove:
    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    def test_displace_moves_group(self, cu100_slab, rng):
        iface = Interface(substrate=cu100_slab)
        o_atom = Atoms("O", positions=[[5.0, 5.0, 18.0]])
        iface.add_adsorbate(o_atom)
        orig_pos = iface.interface.positions.copy()

        move = DisplaceMove(max_displacement=2.0)
        new_iface, log_ratio, info = move.propose(iface, rng)

        assert new_iface is not None
        assert log_ratio == 0.0  # symmetric
        # Appended atom should have moved
        n_sub = len(cu100_slab)
        assert not np.allclose(
            orig_pos[n_sub:], new_iface.interface.positions[n_sub:]
        )

    def test_displace_empty_returns_none(self, cu100_slab, rng):
        iface = Interface(substrate=cu100_slab)
        move = DisplaceMove()
        new_iface, _, info = move.propose(iface, rng)
        assert new_iface is None


class TestSwapMove:
    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    def test_swap_changes_element(self, cu100_slab, rng):
        iface = Interface(substrate=cu100_slab)
        o_atom = Atoms("O", positions=[[5.0, 5.0, 18.0]])
        iface.add_adsorbate(o_atom)

        move = SwapMove(
            allowed_elements=["O", "C", "N"],
            chemical_potentials={"O": -5.0, "C": -7.0, "N": -8.0},
        )
        new_iface, log_ratio, info = move.propose(iface, rng)

        assert new_iface is not None
        assert info["move"] == "swap"
        assert info["old_species"] == "O"
        assert info["new_species"] in ("C", "N")

    def test_swap_needs_alternatives(self, cu100_slab, rng):
        iface = Interface(substrate=cu100_slab)
        o_atom = Atoms("O", positions=[[5.0, 5.0, 18.0]])
        iface.add_adsorbate(o_atom)

        move = SwapMove(allowed_elements=["O"])  # only O available
        new_iface, _, info = move.propose(iface, rng)
        assert new_iface is None
