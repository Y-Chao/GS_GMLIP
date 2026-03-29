"""GA Population management.

Maintains the current population of candidate structures and proposes
which candidates to pair together, using fitness-based roulette wheel
selection. Refactored from ase_ga_official/population.py.

Reference:
    L.B. Vilhelmsen et al., JACS, 2012, 134 (30), pp 12807-12816
"""

from __future__ import annotations

import logging
from math import sqrt, tanh

import numpy as np
from ase import Atoms

from gs_gmlip.data.database import CandidateDatabase

logger = logging.getLogger(__name__)


def _get_raw_score(atoms: Atoms) -> float:
    return atoms.info["key_value_pairs"]["raw_score"]


def _count_looks_like(a: Atoms, all_cand: list[Atoms], comparator) -> int:
    """Count how many candidates in all_cand look like a."""
    n = 0
    for b in all_cand:
        if a.info["confid"] == b.info["confid"]:
            continue
        if comparator.looks_like(a, b):
            n += 1
    return n


class Population:
    """Population for genetic algorithm structure optimization.

    Maintains the current population, computes fitness, and proposes
    candidates for pairing using roulette wheel selection with
    fitness criteria from Vilhelmsen et al.

    Parameters
    ----------
    db : CandidateDatabase
        Database connection.
    population_size : int
        Number of candidates in the population.
    comparator : object, optional
        Comparator with looks_like(a, b) method.
    logfile : str, optional
        Path to population log file.
    use_extinct : bool
        Whether to respect the extinct flag.
    rng : numpy.random.Generator, optional
        Random number generator.
    """

    def __init__(
        self,
        db: CandidateDatabase,
        population_size: int,
        comparator=None,
        logfile: str | None = None,
        use_extinct: bool = False,
        rng: np.random.Generator | None = None,
    ):
        self.db = db
        self.pop_size = population_size
        if comparator is None:
            from gs_gmlip.search.ga.comparators import AtomsComparator

            comparator = AtomsComparator()
        self.comparator = comparator
        self.logfile = logfile
        self.use_extinct = use_extinct
        self.rng = rng or np.random.default_rng()
        self.pop: list[Atoms] = []
        self.pairs: list[tuple[int, int]] = []
        self.all_cand: list[Atoms] = []
        self._initialize_pop()

    def _initialize_pop(self):
        """Initialize population from database."""
        all_cand = self.db.get_all_relaxed(use_extinct=self.use_extinct)
        all_cand.sort(key=_get_raw_score, reverse=True)

        i = 0
        while i < len(all_cand) and len(self.pop) < self.pop_size:
            c = all_cand[i]
            i += 1
            if not any(self.comparator.looks_like(a, c) for a in self.pop):
                self.pop.append(c)

        for a in self.pop:
            a.info["looks_like"] = _count_looks_like(a, all_cand, self.comparator)

        self.all_cand = all_cand
        self._calc_participation()

    def _calc_participation(self):
        """Update pairing participation counts."""
        participation, pairs = self.db.get_participation()
        for a in self.pop:
            confid = a.info["confid"]
            a.info["n_paired"] = participation.get(confid, 0)
        self.pairs = pairs

    def update(self, new_cand: list[Atoms] | None = None):
        """Update population with new candidates."""
        if len(self.pop) == 0:
            self._initialize_pop()

        if new_cand is None:
            new_cand = self.db.get_all_relaxed(
                only_new=True, use_extinct=self.use_extinct
            )

        for a in new_cand:
            self._add_candidate(a)
            self.all_cand.append(a)
        self._calc_participation()

    def _add_candidate(self, a: Atoms):
        """Add a single candidate to the population."""
        raw_score_a = _get_raw_score(a)
        if len(self.pop) == self.pop_size:
            raw_score_worst = _get_raw_score(self.pop[-1])
            if raw_score_a < raw_score_worst:
                return

        # Check if it replaces a similar structure
        for i, b in enumerate(self.pop):
            if self.comparator.looks_like(a, b):
                if _get_raw_score(b) < raw_score_a:
                    del self.pop[i]
                    a.info["looks_like"] = _count_looks_like(
                        a, self.all_cand, self.comparator
                    )
                    self.pop.append(a)
                    self.pop.sort(key=_get_raw_score, reverse=True)
                return

        if len(self.pop) == self.pop_size:
            del self.pop[-1]

        a.info["looks_like"] = _count_looks_like(a, self.all_cand, self.comparator)
        self.pop.append(a)
        self.pop.sort(key=_get_raw_score, reverse=True)

    def _get_fitness(
        self, indices: list[int], with_history: bool = True
    ) -> list[float]:
        """Calculate fitness using Vilhelmsen et al. formula."""
        scores = [_get_raw_score(x) for x in self.pop]
        min_s = min(scores)
        max_s = max(scores)
        T = min_s - max_s
        if T == 0:
            return [1.0] * len(indices)

        f = [0.5 * (1.0 - tanh(2.0 * (scores[i] - max_s) / T - 1.0)) for i in indices]

        if with_history:
            M = [float(self.pop[i].info.get("n_paired", 0)) for i in indices]
            L = [float(self.pop[i].info.get("looks_like", 0)) for i in indices]
            f = [f[j] / sqrt(1.0 + M[j]) / sqrt(1.0 + L[j]) for j in range(len(f))]
        return f

    def get_two_candidates(
        self, with_history: bool = True
    ) -> tuple[Atoms, Atoms] | None:
        """Select two candidates for pairing via roulette wheel."""
        if len(self.pop) < 2:
            self.update()
        if len(self.pop) < 2:
            return None

        fit = self._get_fitness(list(range(len(self.pop))), with_history)
        fmax = max(fit)

        c1 = c2 = self.pop[0]
        used_before = False

        while c1.info["confid"] == c2.info["confid"] and not used_before:
            # Select c1
            while True:
                t = self.rng.integers(len(self.pop))
                if fit[t] > self.rng.random() * fmax:
                    c1 = self.pop[t]
                    break
            # Select c2
            while True:
                t = self.rng.integers(len(self.pop))
                if fit[t] > self.rng.random() * fmax:
                    c2 = self.pop[t]
                    break

            c1id, c2id = c1.info["confid"], c2.info["confid"]
            used_before = tuple(sorted([c1id, c2id])) in self.pairs

        return (c1.copy(), c2.copy())

    def get_one_candidate(self, with_history: bool = True) -> Atoms | None:
        """Select one candidate for mutation via roulette wheel."""
        if len(self.pop) < 1:
            self.update()
        if len(self.pop) < 1:
            return None

        fit = self._get_fitness(list(range(len(self.pop))), with_history)
        fmax = max(fit)

        while True:
            t = self.rng.integers(len(self.pop))
            if fit[t] > self.rng.random() * fmax:
                return self.pop[t].copy()

    def get_current_population(self) -> list[Atoms]:
        """Return a copy of the current population."""
        self.update()
        return [a.copy() for a in self.pop]

    def mass_extinction(self, ids: list[int]):
        """Kill candidates by their confids."""
        for confid in ids:
            self.db.kill_candidate(confid)
        self.pop = []
