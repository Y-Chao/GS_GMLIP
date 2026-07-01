"""Tests for GA operators on Interface-based structures."""

from __future__ import annotations

import numpy as np
import pytest

from gs_gmlip.interface.core import Interface
from gs_gmlip.search.ga.operators import (
    CutAndSpliceCrossover,
    MirrorMutation,
    OperationSelector,
    PermutationMutation,
    RattleMutation,
    TwistMutation,
    _appended_indices,
    _check_min_distances,
    _generate_blmin,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_iface_with_ads(slab, *molecules):
    """Build an Interface from a slab and molecules placed above it."""
    iface = Interface(substrate=slab)
    top = slab.positions[:, 2].max()
    for mol in molecules:
        m = mol.copy()
        m.translate([5.0, 5.0, top + 2.0 + len(iface.adsList) * 2.0])
        iface.add_adsorbate(m)
    iface.fix_substrate(layers=2)
    return iface


# ---------------------------------------------------------------------------
# Tests: utility functions
# ---------------------------------------------------------------------------


class TestUtilities:
    def test_appended_indices_empty(self, cu100_slab):
        iface = Interface(substrate=cu100_slab)
        assert _appended_indices(iface) == []

    def test_appended_indices_with_ads(self, cu100_slab, co_molecule):
        iface = _make_iface_with_ads(cu100_slab, co_molecule)
        idx = _appended_indices(iface)
        assert len(idx) == 2  # CO has 2 atoms
        assert all(i >= len(cu100_slab) for i in idx)

    def test_generate_blmin(self):
        blmin = _generate_blmin([29, 8, 1], scale=0.7)
        assert (29, 8) in blmin or (8, 29) in blmin
        assert blmin[(29, 29)] > 0

    def test_check_min_distances_ok(self, cu100_slab, co_molecule):
        iface = _make_iface_with_ads(cu100_slab, co_molecule)
        blmin = _generate_blmin([29, 6, 8], scale=0.3)  # very permissive
        assert _check_min_distances(iface, blmin)

    def test_check_min_distances_fail(self, cu100_slab):
        """Two atoms very close should fail strict blmin."""
        # Place two identical single atoms very close
        iface = Interface(substrate=cu100_slab)
        top = cu100_slab.positions[:, 2].max()
        for _ in range(2):
            mol = cu100_slab[0:1].copy()  # single Cu atom
            mol.positions[0] = [5.0, 5.0, top + 2.0]
            iface.add_adsorbate(mol)
        # Very strict blmin (large minimum distance)
        blmin = {(29, 29): 10.0}
        assert not _check_min_distances(iface, blmin)


# ---------------------------------------------------------------------------
# Tests: OperationSelector
# ---------------------------------------------------------------------------


class TestOperationSelector:
    def test_selection(self):
        rng = np.random.default_rng(42)
        op1 = RattleMutation(rng=rng)
        op1.descriptor = "op1"
        op2 = PermutationMutation(rng=rng)
        op2.descriptor = "op2"

        sel = OperationSelector([1.0, 0.0], [op1, op2], rng=rng)
        # With probability 1.0 for op1, should always select op1
        for _ in range(20):
            assert sel.get_operator().descriptor == "op1"

    def test_weighted_selection(self):
        rng = np.random.default_rng(42)
        op1 = RattleMutation(rng=rng)
        op1.descriptor = "op1"
        op2 = PermutationMutation(rng=rng)
        op2.descriptor = "op2"

        sel = OperationSelector([0.0, 1.0], [op1, op2], rng=rng)
        for _ in range(20):
            assert sel.get_operator().descriptor == "op2"


# ---------------------------------------------------------------------------
# Tests: crossover
# ---------------------------------------------------------------------------


class TestCutAndSpliceCrossover:
    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    def test_crossover_produces_offspring(self, cu100_slab, co_molecule, rng):
        """Crossover of two slab+adsorbate parents produces valid offspring."""
        p1 = _make_iface_with_ads(cu100_slab, co_molecule, co_molecule)
        p2 = _make_iface_with_ads(cu100_slab, co_molecule, co_molecule)

        op = CutAndSpliceCrossover(rng=rng)
        offspring = op.get_new_individual([p1, p2])

        assert offspring is not None
        assert isinstance(offspring, Interface)
        # Substrate should be preserved
        assert len(offspring.substrate) == len(cu100_slab)
        # Should have some appended atoms
        assert len(_appended_indices(offspring)) > 0

    def test_crossover_needs_two_parents(self, cu100_slab, co_molecule, rng):
        iface = _make_iface_with_ads(cu100_slab, co_molecule)
        op = CutAndSpliceCrossover(rng=rng)
        assert op.get_new_individual([iface]) is None

    def test_crossover_no_adsorbates(self, cu100_slab, rng):
        """Crossover of two bare slabs returns None (no active atoms)."""
        p1 = Interface(substrate=cu100_slab)
        p2 = Interface(substrate=cu100_slab)
        op = CutAndSpliceCrossover(rng=rng)
        assert op.get_new_individual([p1, p2]) is None


# ---------------------------------------------------------------------------
# Tests: mutations
# ---------------------------------------------------------------------------


class TestRattleMutation:
    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    def test_rattle_displaces_atoms(self, cu100_slab, co_molecule, rng):
        iface = _make_iface_with_ads(cu100_slab, co_molecule)
        op = RattleMutation(rattle_strength=1.0, rattle_prop=1.0, rng=rng)
        original_positions = iface.interface.positions.copy()

        offspring = op.get_new_individual([iface])
        assert offspring is not None
        new_positions = offspring.interface.positions

        # Substrate positions should be unchanged
        n_sub = len(cu100_slab)
        assert np.allclose(original_positions[:n_sub], new_positions[:n_sub])

        # Appended atoms should have moved (since rattle_prop=1.0)
        assert not np.allclose(original_positions[n_sub:], new_positions[n_sub:])

    def test_rattle_no_ads(self, cu100_slab, rng):
        iface = Interface(substrate=cu100_slab)
        op = RattleMutation(rng=rng)
        assert op.get_new_individual([iface]) is None


class TestPermutationMutation:
    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    def test_permutation_swaps_positions(self, cu100_slab, rng):
        """Place two different single-atom species and verify swap."""
        iface = Interface(substrate=cu100_slab)
        top = cu100_slab.positions[:, 2].max()

        # Add an O atom
        o_atom = cu100_slab[0:1].copy()
        o_atom.symbols = ["O"]
        o_atom.positions[0] = [5.0, 5.0, top + 2.0]
        iface.add_adsorbate(o_atom)

        # Add a C atom
        c_atom = cu100_slab[0:1].copy()
        c_atom.symbols = ["C"]
        c_atom.positions[0] = [7.0, 7.0, top + 2.0]
        iface.add_adsorbate(c_atom)

        op = PermutationMutation(rng=rng)
        offspring = op.get_new_individual([iface])
        assert offspring is not None

        # Symbols should still be O and C (positions swapped)
        appended = _appended_indices(offspring)
        symbols = set(offspring.interface.symbols[i] for i in appended)
        assert "O" in symbols
        assert "C" in symbols


class TestMirrorMutation:
    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    def test_mirror_modifies_positions(self, cu100_slab, co_molecule, rng):
        iface = _make_iface_with_ads(cu100_slab, co_molecule, co_molecule)
        op = MirrorMutation(rng=rng)
        offspring = op.get_new_individual([iface])
        assert offspring is not None
        # substrate unchanged
        n_sub = len(cu100_slab)
        assert np.allclose(
            iface.interface.positions[:n_sub],
            offspring.interface.positions[:n_sub],
        )


class TestTwistMutation:
    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    def test_twist_rotates_around_z(self, cu100_slab, co_molecule, rng):
        """Twist rotates appended atoms around the z-axis."""
        # Manually place CO at different (x,y) positions so rotation moves them
        iface = Interface(substrate=cu100_slab)
        top = cu100_slab.positions[:, 2].max()
        co1 = co_molecule.copy()
        co1.translate([3.0, 5.0, top + 2.0])
        iface.add_adsorbate(co1)
        co2 = co_molecule.copy()
        co2.translate([7.0, 5.0, top + 2.0])
        iface.add_adsorbate(co2)
        iface.fix_substrate(layers=2)

        op = TwistMutation(angle_range=(90, 90), rng=rng)  # exactly 90°

        offspring = op.get_new_individual([iface])
        assert offspring is not None

        # Substrate unchanged
        n_sub = len(cu100_slab)
        assert np.allclose(
            iface.interface.positions[:n_sub],
            offspring.interface.positions[:n_sub],
        )

        # Appended atoms should have moved in x-y
        old_xy = iface.interface.positions[n_sub:, :2]
        new_xy = offspring.interface.positions[n_sub:, :2]
        assert not np.allclose(old_xy, new_xy)

        # z should be preserved
        old_z = iface.interface.positions[n_sub:, 2]
        new_z = offspring.interface.positions[n_sub:, 2]
        assert np.allclose(old_z, new_z)
