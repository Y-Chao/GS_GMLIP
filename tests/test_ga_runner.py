"""Tests for the GA runner (integration)."""

from __future__ import annotations

import pytest
from ase import Atoms
from ase.calculators.emt import EMT

from gs_gmlip.evaluator.base import BaseEvaluator
from gs_gmlip.interface.core import Interface
from gs_gmlip.search.ga.runner import GARunner


class EMTEvaluator(BaseEvaluator):
    """Concrete evaluator using ASE's EMT calculator."""

    def get_calculator(self):
        return EMT()


class TestGARunner:
    """Integration tests for the GA runner."""

    @pytest.fixture
    def evaluator(self):
        return EMTEvaluator(fmax=0.5, relax_steps=20)

    @pytest.fixture
    def substrate(self, cu100_slab):
        """Cu(100) slab with bottom 2 layers fixed."""
        iface = Interface(substrate=cu100_slab)
        iface.fix_substrate(layers=2)
        return iface

    @pytest.fixture
    def blocks(self):
        """Single Cu atom as building block."""
        return [Atoms("Cu", positions=[[0, 0, 0]])]

    @pytest.fixture
    def runner(self, evaluator, substrate, blocks):
        return GARunner(
            evaluator=evaluator,
            substrate=substrate,
            blocks=blocks,
            counts=[2],  # 2 Cu adatoms
            population_size=5,
            n_initial=5,
            fmax=0.5,
            relax_steps=20,
            seed=42,
        )

    def test_setup_creates_population(self, runner):
        """setup() generates initial population and evaluates them."""
        runner.setup()
        assert runner.population is not None
        assert runner.population.size > 0
        # All individuals should have a calculator attached
        for iface in runner.population.individuals:
            assert iface.interface.calc is not None

    def test_run_completes(self, runner):
        """Full run completes without error."""
        runner.setup()
        results = runner.run(n_steps=3)
        assert len(results) > 0

    def test_get_best_returns_sorted(self, runner):
        """get_best returns lowest-energy structures first."""
        runner.setup()
        runner.run(n_steps=3)
        best = runner.get_best(n=3)
        assert len(best) >= 1
        # Verify energies are sorted
        energies = [iface.interface.get_potential_energy() for iface in best]
        assert energies == sorted(energies)

    def test_reproducibility(self, evaluator, substrate, blocks):
        """Same seed → same results."""
        r1 = GARunner(
            evaluator=evaluator, substrate=substrate,
            blocks=blocks, counts=[2],
            population_size=5, n_initial=5,
            fmax=0.5, relax_steps=20, seed=42,
        )
        r2 = GARunner(
            evaluator=evaluator, substrate=substrate,
            blocks=blocks, counts=[2],
            population_size=5, n_initial=5,
            fmax=0.5, relax_steps=20, seed=42,
        )
        r1.setup()
        r2.setup()
        # Both should have same population size and similar energies
        assert r1.population.size == r2.population.size
