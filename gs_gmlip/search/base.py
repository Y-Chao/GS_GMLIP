"""Abstract base class for all search methods.

All search methods (GA, GCMC, GOFEE, SSW, Metadynamics) inherit from
BaseSearcher and implement the setup/step/run/get_results interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from ase import Atoms

from gs_gmlip.evaluate.base import BaseEvaluator

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
        self._results: list[Atoms] = []
        self._is_setup = False

    @abstractmethod
    def setup(self) -> None:
        """Initialize the search. Must be called before run()."""

    @abstractmethod
    def step(self) -> dict:
        """Perform one search step.

        Returns
        -------
        info : dict
            Information about the step (energy, acceptance, etc.).
        """

    def run(self, n_steps: int = 100) -> list[Atoms]:
        """Run the search for n_steps.

        Parameters
        ----------
        n_steps : int
            Number of search steps to perform.

        Returns
        -------
        results : list of Atoms
            All structures found during the search.
        """
        if not self._is_setup:
            self.setup()

        logger.info(f"Starting {type(self).__name__} for {n_steps} steps")

        for i in range(n_steps):
            try:
                info = self.step()
                self._step += 1

                if i % max(1, n_steps // 10) == 0:
                    logger.info(f"Step {self._step}: {info}")
            except Exception as e:
                logger.warning(f"Step {self._step} failed: {e}")
                continue

        logger.info(f"Search completed. {len(self._results)} structures collected.")
        return self.get_results()

    def get_results(self) -> list[Atoms]:
        """Return all structures found during the search."""
        return list(self._results)

    def get_best(self, n: int = 1) -> list[Atoms]:
        """Return the n lowest energy structures found.

        Parameters
        ----------
        n : int
            Number of structures to return.
        """
        sorted_results = sorted(
            [a for a in self._results if a.calc is not None],
            key=lambda a: a.get_potential_energy(),
        )
        return sorted_results[:n]

    @classmethod
    def from_config(cls, config: dict, evaluator: BaseEvaluator, workdir: str):
        """Create a searcher from a configuration dictionary.

        Subclasses should override this to handle their specific config.
        """
        raise NotImplementedError(f"{cls.__name__} does not implement from_config")
