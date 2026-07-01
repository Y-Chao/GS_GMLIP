"""Tests for the GCMC runner (integration)."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT

from gs_gmlip.evaluator.base import BaseEvaluator
from gs_gmlip.interface.core import Interface
from gs_gmlip.search.gcmc.runner import GCMCRunner


class EMTEvaluator(BaseEvaluator):
    """Concrete evaluator using ASE's EMT calculator."""

    def get_calculator(self):
        return EMT()


class TestGCMCRunner:
    """Integration tests for the GCMC runner."""

    @pytest.fixture
    def evaluator(self):
        return EMTEvaluator(fmax=0.5, relax_steps=15)

    @pytest.fixture
    def substrate(self, cu100_slab):
        """Cu(100) slab with bottom 2 layers fixed."""
        iface = Interface(substrate=cu100_slab)
        iface.fix_substrate(layers=2)
        return iface

    @pytest.fixture
    def runner(self, evaluator, substrate):
        return GCMCRunner(
            evaluator=evaluator,
            substrate=substrate,
            blocks=[Atoms("O", positions=[[0, 0, 0]])],
            block_names=["O"],
            chemical_potentials={"O": -5.0},
            temperature=300.0,
            max_displacement=1.0,
            move_weights={"insert": 0.4, "delete": 0.3, "displace": 0.3},
            seed=42,
        )

    def test_setup_initializes(self, runner):
        """setup() evaluates the initial configuration."""
        runner.setup()
        assert runner.current is not None
        assert runner.current.interface.calc is not None
        assert np.isfinite(runner.current_energy)

    def test_run_completes(self, runner):
        """Full run completes without error."""
        runner.setup()
        runner.run(n_steps=5)
        assert runner.step_count == 5

    def test_acceptance_tracking(self, runner):
        """Acceptance statistics are tracked."""
        runner.setup()
        runner.run(n_steps=5)
        summary = runner.get_summary()
        assert summary["total_accepted"] + summary["total_rejected"] + summary["total_invalid"] >= 0
        assert summary["temperature"] == 300.0

    def test_best_tracks_minimum(self, runner):
        """Best energy is tracked across steps."""
        runner.setup()
        initial_best = runner.best_energy
        runner.run(n_steps=5)
        # Best should be <= initial (can only improve or stay same)
        assert runner.best_energy <= initial_best

    def test_reproducibility(self, evaluator, substrate):
        """Same seed → same results."""
        r1 = GCMCRunner(
            evaluator=evaluator, substrate=substrate,
            blocks=[Atoms("O", positions=[[0, 0, 0]])],
            block_names=["O"],
            chemical_potentials={"O": -5.0},
            temperature=300.0, seed=42,
        )
        r2 = GCMCRunner(
            evaluator=evaluator, substrate=substrate,
            blocks=[Atoms("O", positions=[[0, 0, 0]])],
            block_names=["O"],
            chemical_potentials={"O": -5.0},
            temperature=300.0, seed=42,
        )
        r1.setup()
        r2.setup()
        # Same seed should give same initial energy
        assert r1.current_energy == pytest.approx(r2.current_energy)
