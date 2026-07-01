"""Tests for GA Population management."""

from __future__ import annotations

import numpy as np
import pytest
from ase.calculators.calculator import Calculator

from gs_gmlip.interface.core import Interface
from gs_gmlip.search.ga.population import Population, _tanh_fitness


class _DummyCalculator(Calculator):
    """A calculator that always returns a fixed energy."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, energy: float = 0.0):
        super().__init__()
        self._energy = energy

    def calculate(self, atoms=None, properties=None, system_changes=None):
        n = len(atoms) if atoms is not None else len(self.atoms)
        self.results = {
            "energy": self._energy,
            "forces": np.zeros((n, 3)),
        }


class TestTanhFitness:
    def test_uniform_energies(self):
        """All equal energies → uniform fitness."""
        e = np.array([-10.0, -10.0, -10.0])
        f = _tanh_fitness(e)
        assert np.allclose(f, 0.5)

    def test_lower_energy_higher_fitness(self):
        """Lower energy → higher fitness."""
        e = np.array([-10.0, -5.0, -1.0])
        f = _tanh_fitness(e)
        assert f[0] > f[1] > f[2]

    def test_range_0_to_1(self):
        """Fitness values are in (0, 1)."""
        e = np.array([-100.0, 0.0, 50.0, 200.0])
        f = _tanh_fitness(e)
        assert np.all(f > 0)
        assert np.all(f < 1)


class TestPopulation:
    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    @staticmethod
    def _make_iface(slab, energy):
        """Make a mock Interface with a dummy calculator returning the given energy."""
        iface = Interface(substrate=slab)
        iface.interface.calc = _DummyCalculator(energy=energy)
        return iface

    def test_empty_population(self, cu100_slab):
        pop = Population(population_size=5)
        assert pop.size == 0
        assert pop.get_one_candidate() is None
        assert pop.get_two_candidates() is None
        assert pop.best() is None

    def test_add_individuals(self, cu100_slab, rng):
        pop = Population(population_size=3, rng=rng)
        for e in [-10.0, -8.0, -12.0]:
            iface = self._make_iface(cu100_slab, e)
            assert pop.add(iface)

        assert pop.size == 3
        assert pop.best_energy() == pytest.approx(-12.0)

    def test_replaces_worst(self, cu100_slab, rng):
        pop = Population(population_size=3, rng=rng)
        for e in [-10.0, -8.0, -6.0]:
            pop.add(self._make_iface(cu100_slab, e))
        assert pop.size == 3

        # Try to add a better one
        assert pop.add(self._make_iface(cu100_slab, -12.0))
        assert pop.size == 3
        assert pop.best_energy() == pytest.approx(-12.0)

        # A worse one should be rejected
        assert not pop.add(self._make_iface(cu100_slab, -4.0))
        assert pop.size == 3

    def test_selection_is_weighted(self, cu100_slab, rng):
        """Best individuals are selected more often."""
        pop = Population(population_size=5, rng=rng)
        pop.add(self._make_iface(cu100_slab, -100.0))  # very good
        pop.add(self._make_iface(cu100_slab, -1.0))
        pop.add(self._make_iface(cu100_slab, -1.0))
        pop.add(self._make_iface(cu100_slab, -1.0))
        pop.add(self._make_iface(cu100_slab, -1.0))

        # The best candidate should dominate selections
        best_count = 0
        for _ in range(50):
            cand = pop.get_one_candidate()
            if cand is not None and cand.interface.get_potential_energy() == pytest.approx(-100.0):
                best_count += 1
        # Should be selected >20% (much more than uniform 20%)
        assert best_count > 10

    def test_get_two_candidates_distinct(self, cu100_slab, rng):
        pop = Population(population_size=5, rng=rng)
        for e in [-10.0, -9.0, -8.0, -7.0, -6.0]:
            pop.add(self._make_iface(cu100_slab, e))

        pair = pop.get_two_candidates()
        assert pair is not None
        a, b = pair
        assert a is not b  # different objects

    def test_comparator_dedup(self, cu100_slab, rng):
        """With a comparator that marks everything as duplicate, only first is kept."""
        pop = Population(population_size=5, comparator=lambda a, b: True, rng=rng)
        assert pop.add(self._make_iface(cu100_slab, -10.0))
        assert not pop.add(self._make_iface(cu100_slab, -12.0))  # rejected as duplicate
        assert pop.size == 1
