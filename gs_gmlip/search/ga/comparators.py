"""Structure comparators for duplicate detection in GA population.

Refactored from ase_ga_official/standard_comparators.py.
Primary comparator: InteratomicDistanceComparator based on pair correlation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)


class BaseComparator(ABC):
    """Abstract base for structure comparators."""

    @abstractmethod
    def looks_like(self, a1: Atoms, a2: Atoms) -> bool:
        """Check if two structures are similar.

        Returns True if structures are considered equivalent.
        """


class InteratomicDistanceComparator(BaseComparator):
    """Compare structures using pair correlation functions.

    Based on the method described in:
    L.B. Vilhelmsen and B. Hammer, PRL 2012.

    Parameters
    ----------
    n_top : int
        Number of atoms in the active region.
    pair_cor_cum_diff : float
        Maximum cumulative difference in pair correlation.
    pair_cor_max : float
        Maximum single-point difference in pair correlation.
    dE : float
        Maximum energy difference (eV) for structures to be similar.
    mic : bool
        Minimum image convention for periodic systems.
    """

    def __init__(
        self,
        n_top: int = 0,
        pair_cor_cum_diff: float = 0.015,
        pair_cor_max: float = 0.7,
        dE: float = 0.02,
        mic: bool = True,
    ):
        self.n_top = n_top
        self.pair_cor_cum_diff = pair_cor_cum_diff
        self.pair_cor_max = pair_cor_max
        self.dE = dE
        self.mic = mic

    def looks_like(self, a1: Atoms, a2: Atoms) -> bool:
        # Energy check
        try:
            e1 = a1.info["key_value_pairs"]["raw_score"]
            e2 = a2.info["key_value_pairs"]["raw_score"]
            if abs(e1 - e2) > self.dE:
                return False
        except (KeyError, TypeError):
            pass

        # Pair correlation comparison
        n1 = len(a1)
        n2 = len(a2)
        if n1 != n2:
            return False

        n_slab = n1 - self.n_top if self.n_top > 0 else 0
        top1 = a1[n_slab:]
        top2 = a2[n_slab:]

        if len(top1) < 2:
            return True

        d1 = sorted(top1.get_all_distances(mic=self.mic).flatten())
        d2 = sorted(top2.get_all_distances(mic=self.mic).flatten())

        if len(d1) != len(d2):
            return False

        d1 = np.array(d1)
        d2 = np.array(d2)

        diff = np.abs(d1 - d2)

        if np.max(diff) > self.pair_cor_max:
            return False
        if np.mean(diff) > self.pair_cor_cum_diff:
            return False

        return True


class EnergyComparator(BaseComparator):
    """Compare structures based on energy difference only.

    Parameters
    ----------
    dE : float
        Maximum energy difference in eV.
    """

    def __init__(self, dE: float = 0.02):
        self.dE = dE

    def looks_like(self, a1: Atoms, a2: Atoms) -> bool:
        try:
            e1 = a1.info["key_value_pairs"]["raw_score"]
            e2 = a2.info["key_value_pairs"]["raw_score"]
            return abs(e1 - e2) < self.dE
        except (KeyError, TypeError):
            return False


class CompositionComparator(BaseComparator):
    """Compare structures based on chemical composition."""

    def looks_like(self, a1: Atoms, a2: Atoms) -> bool:
        return a1.get_chemical_formula(mode="hill") == a2.get_chemical_formula(
            mode="hill"
        )


class AtomsComparator(BaseComparator):
    """Direct comparison of Atoms objects (positions and numbers)."""

    def looks_like(self, a1: Atoms, a2: Atoms) -> bool:
        if len(a1) != len(a2):
            return False
        if not np.array_equal(a1.get_atomic_numbers(), a2.get_atomic_numbers()):
            return False
        return np.allclose(a1.get_positions(), a2.get_positions(), atol=1e-5)


class SequentialComparator(BaseComparator):
    """Combine multiple comparators with AND/OR logic.

    Parameters
    ----------
    comparators : list of BaseComparator
    logic : str
        "and" or "or".
    """

    def __init__(self, comparators: list[BaseComparator], logic: str = "and"):
        self.comparators = comparators
        self.logic = logic

    def looks_like(self, a1: Atoms, a2: Atoms) -> bool:
        results = [c.looks_like(a1, a2) for c in self.comparators]
        if self.logic == "and":
            return all(results)
        else:
            return any(results)
