"""Abstract base class for all search methods.

All search methods (GA, GCMC, GOFEE, SSW, Metadynamics) inherit from
BaseSearcher and implement setup/step/run/get_results.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from gs_gmlip.evaluator.base import BaseEvaluator
from gs_gmlip.interface.core import Interface

logger = logging.getLogger(__name__)


class BaseSearcher(ABC):
    """Abstract base class for global structure search methods.

    Parameters
    ----------
    evaluator : BaseEvaluator
        Energy/force evaluator.
    workdir : str
        Working directory for output files.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        evaluator: BaseEvaluator,
        workdir: str = ".",
        seed: int | None = None,
    ):
        self.evaluator = evaluator
        self.workdir = workdir
        self.seed = seed
        self._step = 0
        self._results: list[Interface] = []
        self._is_setup = False

    @abstractmethod
    def setup(self) -> None:
        """Initialize the search. Must be called before ``run()``."""

    @abstractmethod
    def step(self) -> dict:
        """Perform one search step.

        Returns
        -------
        info : dict
            Information about the step (energy, acceptance, etc.).
        """

    def run(self, n_steps: int = 100) -> list[Interface]:
        """Run the search for *n_steps*.

        Parameters
        ----------
        n_steps : int
            Number of search steps to perform.

        Returns
        -------
        results : list of Interface
            All structures found during the search.
        """
        if not self._is_setup:
            self.setup()

        logger.info("Starting %s for %d steps", type(self).__name__, n_steps)

        for i in range(n_steps):
            try:
                info = self.step()
                self._step += 1
                if i % max(1, n_steps // 10) == 0:
                    logger.info("Step %d: %s", self._step, info)
            except Exception:
                logger.warning("Step %d failed", self._step, exc_info=True)
                continue

        logger.info("Search completed. %d structures collected.", len(self._results))
        return self.get_results()

    def get_results(self) -> list[Interface]:
        """Return all structures found during the search."""
        return list(self._results)

    def get_best(self, n: int = 1) -> list[Interface]:
        """Return the *n* lowest-energy structures found."""
        with_energy = [
            iface
            for iface in self._results
            if iface.interface.calc is not None
        ]
        sorted_results = sorted(
            with_energy,
            key=lambda iface: iface.interface.get_potential_energy(),
        )
        return sorted_results[:n]
