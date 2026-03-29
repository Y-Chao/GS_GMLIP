"""Abstract evaluator interface.

Wraps ASE Calculator protocol to provide a unified interface for
energy/force evaluation across different backends (MLIP, DFT).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize import BFGS

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """Abstract base class for structure evaluation.

    All evaluators must implement get_calculator() which returns
    an ASE Calculator instance.
    """

    @abstractmethod
    def get_calculator(self) -> Calculator:
        """Return an ASE Calculator instance."""

    def evaluate(self, atoms: Atoms, relax: bool = False, **kwargs) -> Atoms:
        """Evaluate energy and forces for a structure.

        Parameters
        ----------
        atoms : Atoms
            Structure to evaluate.
        relax : bool
            If True, perform local relaxation before evaluation.

        Returns
        -------
        atoms : Atoms
            Structure with calculated energy and forces attached.
        """
        atoms = atoms.copy()
        calc = self.get_calculator()
        atoms.calc = calc

        if relax:
            atoms = self.relax(atoms, **kwargs)
        else:
            atoms.get_potential_energy()
            atoms.get_forces()

        return atoms

    def relax(
        self,
        atoms: Atoms,
        fmax: float = 0.05,
        steps: int = 200,
        optimizer_cls=None,
        logfile: str | None = None,
    ) -> Atoms:
        """Locally relax a structure.

        Parameters
        ----------
        atoms : Atoms
            Structure to relax (must have calculator attached).
        fmax : float
            Force convergence criterion in eV/Angstrom.
        steps : int
            Maximum optimization steps.
        optimizer_cls : class, optional
            ASE Optimizer class. Default is BFGS.
        logfile : str, optional
            Path to write optimization log. None for no output.
        """
        if atoms.calc is None:
            atoms.calc = self.get_calculator()

        if optimizer_cls is None:
            optimizer_cls = BFGS

        opt = optimizer_cls(atoms, logfile=logfile)
        opt.run(fmax=fmax, steps=steps)
        return atoms

    def batch_evaluate(
        self, structures: list[Atoms], relax: bool = False, **kwargs
    ) -> list[Atoms]:
        """Evaluate a list of structures sequentially.

        Parameters
        ----------
        structures : list of Atoms
        relax : bool
        """
        results = []
        for atoms in structures:
            try:
                result = self.evaluate(atoms, relax=relax, **kwargs)
                results.append(result)
            except Exception as e:
                logger.warning(f"Evaluation failed for structure: {e}")
                results.append(None)
        return results
