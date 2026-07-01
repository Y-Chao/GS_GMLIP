"""GA Population management.

Maintains the current population of candidate structures and provides
fitness-based parent selection (roulette wheel).

Reference:
    L.B. Vilhelmsen et al., JACS, 2012, 134 (30), pp 12807-12816
"""

from __future__ import annotations

import logging

import numpy as np

from gs_gmlip.interface.core import Interface

logger = logging.getLogger(__name__)


def _tanh_fitness(energies: np.ndarray) -> np.ndarray:
    """Compute fitness via tanh scaling (Vilhelmsen et al.).

    F_i = 0.5 * (1 - tanh(2 * rho_i - 1))
    where rho_i = (E_i - E_min) / (E_max - E_min)

    Lower energy → higher fitness.  Returns fitness in (0, 1).
    """
    e_min, e_max = energies.min(), energies.max()
    if e_max - e_min < 1e-12:
        return np.ones_like(energies) * 0.5
    rho = (energies - e_min) / (e_max - e_min)
    return 0.5 * (1.0 - np.tanh(2.0 * rho - 1.0))


class Population:
    """Population for genetic algorithm structure optimization.

    Maintains a fixed-size population, computes fitness, and provides
    parent selection via roulette-wheel sampling.

    Parameters
    ----------
    population_size : int
        Maximum number of candidates kept in the population.
    comparator : callable, optional
        Function ``looks_like(a: Interface, b: Interface) -> bool`` for
        deduplication.  If None, duplicates are allowed.
    rng : np.random.Generator, optional
    """

    def __init__(
        self,
        population_size: int = 20,
        comparator=None,
        rng: np.random.Generator | None = None,
    ):
        self.pop_size = population_size
        self.comparator = comparator
        self.rng = rng or np.random.default_rng()
        self._individuals: list[Interface] = []
        self._energies: list[float] = []
        self._fitness: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def individuals(self) -> list[Interface]:
        """Current population, sorted best (lowest energy) first."""
        return list(self._individuals)

    @property
    def size(self) -> int:
        return len(self._individuals)

    @property
    def energies(self) -> list[float]:
        return list(self._energies)

    # ------------------------------------------------------------------
    # Population management
    # ------------------------------------------------------------------

    def add(self, interface: Interface) -> bool:
        """Add an individual to the population.

        Returns True if the individual was accepted (not a duplicate /
        not worse than the worst when population is full).
        """
        energy = interface.interface.get_potential_energy()

        # Deduplicate against existing population
        if self.comparator is not None:
            for existing in self._individuals:
                if self.comparator(existing, interface):
                    logger.debug("Duplicate rejected (energy=%.4f)", energy)
                    return False

        if len(self._individuals) < self.pop_size:
            self._individuals.append(interface)
            self._energies.append(energy)
            self._sort()
            return True

        # Population is full — replace worst if better
        worst_energy = self._energies[-1]  # sorted ascending
        if energy < worst_energy:
            self._individuals[-1] = interface
            self._energies[-1] = energy
            self._sort()
            logger.debug("Replaced worst (%.4f → %.4f)", worst_energy, energy)
            return True

        return False

    def update(self, interfaces: list[Interface]) -> int:
        """Try to add multiple individuals. Returns how many were accepted."""
        return sum(1 for iface in interfaces if self.add(iface))

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _sort(self) -> None:
        """Sort by energy ascending, recompute fitness."""
        order = np.argsort(self._energies)
        self._individuals = [self._individuals[i] for i in order]
        self._energies = [self._energies[i] for i in order]
        if self._energies:
            self._fitness = _tanh_fitness(np.array(self._energies))

    def _normalized_fitness(self) -> np.ndarray:
        """Return fitness normalized to sum to 1 (for roulette wheel)."""
        if self._fitness is None or len(self._fitness) == 0:
            return np.array([])
        total = self._fitness.sum()
        if total < 1e-12:
            return np.ones(len(self._fitness)) / len(self._fitness)
        return self._fitness / total

    def get_one_candidate(self) -> Interface | None:
        """Select one individual by roulette-wheel fitness."""
        if not self._individuals:
            return None
        probs = self._normalized_fitness()
        idx = self.rng.choice(len(self._individuals), p=probs)
        return self._individuals[idx]

    def get_two_candidates(self) -> tuple[Interface, Interface] | None:
        """Select two distinct individuals for crossover."""
        if len(self._individuals) < 2:
            return None
        probs = self._normalized_fitness()
        idx = self.rng.choice(len(self._individuals), size=2, replace=False, p=probs)
        return self._individuals[idx[0]], self._individuals[idx[1]]

    def best(self) -> Interface | None:
        """Return the best (lowest energy) individual."""
        return self._individuals[0] if self._individuals else None

    def best_energy(self) -> float:
        """Return the lowest energy in the population."""
        return self._energies[0] if self._energies else float("inf")
