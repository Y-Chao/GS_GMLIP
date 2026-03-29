"""Minima hopping strategy for SSW.

Combines SSW walking with Metropolis-like acceptance to explore
the PES by hopping between local minima.
"""

from __future__ import annotations

import logging

import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)


class MinimaHopping:
    """Minima hopping with adaptive temperature.

    Parameters
    ----------
    temperature : float
        Initial temperature (K).
    beta_decrease : float
        Factor to decrease temperature on acceptance (< 1).
    beta_increase : float
        Factor to increase temperature on rejection (> 1).
    T_min : float
        Minimum temperature (K).
    T_max : float
        Maximum temperature (K).
    """

    kB = 8.617333262e-5  # eV/K

    def __init__(
        self,
        temperature: float = 300.0,
        beta_decrease: float = 0.95,
        beta_increase: float = 1.05,
        T_min: float = 100.0,
        T_max: float = 3000.0,
    ):
        self.temperature = temperature
        self.beta_decrease = beta_decrease
        self.beta_increase = beta_increase
        self.T_min = T_min
        self.T_max = T_max

        # History
        self.visited_energies: list[float] = []
        self.n_accepted = 0
        self.n_rejected = 0

    @property
    def acceptance_rate(self) -> float:
        total = self.n_accepted + self.n_rejected
        return self.n_accepted / total if total > 0 else 0.0

    def accept(
        self,
        energy_old: float,
        energy_new: float,
        rng: np.random.Generator,
    ) -> bool:
        """Metropolis acceptance with adaptive temperature.

        Parameters
        ----------
        energy_old : float
            Current minimum energy.
        energy_new : float
            New minimum energy.
        rng : Generator
            Random number generator.

        Returns
        -------
        accepted : bool
        """
        dE = energy_new - energy_old
        beta = 1.0 / (self.kB * self.temperature)

        if dE < 0:
            accepted = True
        else:
            p = np.exp(-beta * dE)
            accepted = rng.random() < p

        # Adaptive temperature
        if accepted:
            self.temperature *= self.beta_decrease
            self.n_accepted += 1
            self.visited_energies.append(energy_new)
        else:
            self.temperature *= self.beta_increase
            self.n_rejected += 1

        # Clamp temperature
        self.temperature = np.clip(self.temperature, self.T_min, self.T_max)

        return accepted

    def is_new_minimum(self, energy: float, tolerance: float = 0.01) -> bool:
        """Check if energy represents a new (unvisited) minimum."""
        for e in self.visited_energies:
            if abs(e - energy) < tolerance:
                return False
        return True
