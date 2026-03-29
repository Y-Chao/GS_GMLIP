"""Tests for gs_gmlip.search.gcmc module."""

import numpy as np
import pytest
from ase import Atoms

from gs_gmlip.search.gcmc.ensemble import GCEnsemble
from gs_gmlip.search.gcmc.moves import DeleteMove, DisplaceMove, InsertMove, SwapMove


class TestGCEnsemble:
    def test_init(self):
        ens = GCEnsemble(
            temperature=300.0,
            chemical_potentials={"H": -3.0, "O": -4.5},
        )
        assert ens.beta > 0
        assert ens.acceptance_rate == 0.0

    def test_accept_lower_energy(self):
        ens = GCEnsemble(temperature=300.0, chemical_potentials={"H": -3.0})
        rng = np.random.default_rng(42)
        # Lower energy should always be accepted
        accepted = ens.accept(
            energy_old=0.0,
            energy_new=-1.0,
            move_info={"move": "displace"},
            log_proposal_ratio=0.0,
            rng=rng,
        )
        assert accepted

    def test_stats(self):
        ens = GCEnsemble(temperature=1000.0, chemical_potentials={})
        rng = np.random.default_rng(0)
        for _ in range(10):
            ens.accept(0.0, 0.1, {"move": "displace"}, 0.0, rng)
        total = ens.n_accepted + ens.n_rejected
        assert total == 10

    def test_summary(self):
        ens = GCEnsemble(temperature=300.0, chemical_potentials={"H": -3.0})
        summary = ens.get_summary()
        assert "temperature" in summary
        assert "acceptance_rate" in summary


class TestMoves:
    def test_insert_move(self, simple_blocks, box_region, cu111_slab, rng):
        move = InsertMove(simple_blocks, box_region)
        new_atoms, log_ratio, info = move.propose(cu111_slab, rng)
        assert new_atoms is not None
        assert len(new_atoms) > len(cu111_slab)
        assert info["move"] == "insert"

    def test_delete_move_empty(self, simple_blocks, cu111_slab, rng):
        move = DeleteMove(simple_blocks)
        # Slab with no adsorbates (all tags are 0 or 1 for surface)
        atoms = cu111_slab.copy()
        atoms.set_tags([0] * len(atoms))
        new_atoms, _, info = move.propose(atoms, rng)
        assert new_atoms is None
        assert info["reason"] == "no_adsorbates"

    def test_delete_move_with_adsorbate(self, simple_blocks, slab_with_adsorbate, rng):
        move = DeleteMove(simple_blocks)
        new_atoms, _, info = move.propose(slab_with_adsorbate, rng)
        assert new_atoms is not None
        assert len(new_atoms) < len(slab_with_adsorbate)

    def test_displace_move(self, slab_with_adsorbate, rng):
        move = DisplaceMove(max_displacement=0.5)
        new_atoms, log_ratio, info = move.propose(slab_with_adsorbate, rng)
        assert new_atoms is not None
        assert log_ratio == 0.0
        assert info["move"] == "displace"

    def test_swap_needs_two_blocks(self, cu111_slab, rng):
        move = SwapMove([])
        new_atoms, _, info = move.propose(cu111_slab, rng)
        assert new_atoms is None
