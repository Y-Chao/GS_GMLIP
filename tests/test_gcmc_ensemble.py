"""Tests for GCEnsemble (Metropolis acceptance)."""

from __future__ import annotations

import numpy as np
import pytest

from gs_gmlip.search.gcmc.ensemble import GCEnsemble


class TestGCEnsemble:
    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    @pytest.fixture
    def ensemble(self):
        return GCEnsemble(
            temperature=300.0,
            chemical_potentials={"O": -5.0, "H": -3.5},
        )

    def test_beta_positive(self, ensemble):
        assert ensemble.beta > 0

    def test_acceptance_rate_initial_zero(self, ensemble):
        assert ensemble.acceptance_rate == 0.0

    def test_always_accept_lower_energy_insert(self, ensemble, rng):
        """Insert that lowers energy should always be accepted."""
        accepted = ensemble.accept(
            energy_old=-10.0,
            energy_new=-15.0,  # lower is better
            move_info={"move": "insert", "chemical_potential": -5.0},
            log_proposal_ratio=0.0,
            rng=rng,
        )
        assert accepted
        assert ensemble.n_accepted == 1

    def test_always_accept_lower_energy_delete(self, ensemble, rng):
        """Delete that lowers effective energy should always be accepted."""
        accepted = ensemble.accept(
            energy_old=-5.0,
            energy_new=-10.0,
            move_info={"move": "delete", "chemical_potential": -5.0},
            log_proposal_ratio=0.0,
            rng=rng,
        )
        assert accepted

    def test_reject_high_energy_barrier(self, ensemble, rng):
        """A very unfavorable move should be rejected with high probability."""
        # Use extreme energy difference
        rejected_count = 0
        for _ in range(50):
            accepted = ensemble.accept(
                energy_old=-10.0,
                energy_new=100.0,  # very unfavorable
                move_info={"move": "insert", "chemical_potential": 0.0},
                log_proposal_ratio=0.0,
                rng=np.random.default_rng(),
            )
            if not accepted:
                rejected_count += 1
        assert rejected_count > 40  # overwhelmingly rejected

    def test_swap_uses_difference(self, ensemble, rng):
        """Swap acceptance uses chemical potential difference."""
        accepted = ensemble.accept(
            energy_old=-10.0,
            energy_new=-12.0,  # lower energy
            move_info={
                "move": "swap",
                "old_species": "O",
                "new_species": "H",
                "chemical_potential": -3.5 - (-5.0),  # mu_H - mu_O = +1.5
            },
            log_proposal_ratio=0.0,
            rng=rng,
        )
        assert accepted

    def test_summary(self, ensemble, rng):
        ensemble.accept(-10, -15, {"move": "insert", "chemical_potential": -5.0}, 0.0, rng)
        ensemble.accept(-15, -10, {"move": "delete", "chemical_potential": -5.0}, 0.0, rng)

        summary = ensemble.get_summary()
        assert summary["total_accepted"] + summary["total_rejected"] == 2
        assert "insert" in summary["move_stats"]
        assert "delete" in summary["move_stats"]

    def test_reset_stats(self, ensemble, rng):
        ensemble.accept(-10, -15, {"move": "insert", "chemical_potential": -5.0}, 0.0, rng)
        ensemble.reset_stats()
        assert ensemble.n_accepted == 0
        assert ensemble.acceptance_rate == 0.0
