"""Abstract base for energy/force evaluators.

Every evaluator wraps an ASE Calculator behind a uniform interface so
search methods are decoupled from the energy backend (EMT, MACE, VASP, …).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from ase.calculators.calculator import Calculator
from ase.optimize import BFGS

from gs_gmlip.interface.core import Interface

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """Abstract evaluator: wraps an ASE Calculator.

    Subclasses must implement ``get_calculator()``.  The default
    ``evaluate`` / ``relax`` methods attach that calculator and
    optionally relax the structure via BFGS.

    Parameters
    ----------
    fmax : float
        Force convergence criterion for relaxation (eV/Å).
    relax_steps : int
        Maximum number of BFGS steps.
    """

    def __init__(self, fmax: float = 0.05, relax_steps: int = 200):
        self.fmax = fmax
        self.relax_steps = relax_steps

    @abstractmethod
    def get_calculator(self) -> Calculator:
        """Return a freshly configured ASE Calculator."""

    def evaluate(
        self,
        interface: Interface,
        relax: bool = False,
        fmax: float | None = None,
        steps: int | None = None,
    ) -> Interface:
        """Attach calculator, optionally relax, and return the interface.

        The calculator is attached to ``interface.interface`` (the full
        Atoms).  If *relax* is True, a BFGS relaxation is run first;
        the substrate portion is kept fixed via the FixAtoms constraint
        already present on the Interface.

        Returns the same Interface with its ``interface`` Atoms updated
        (positions and energy).  The caller should re-derive cached
        properties via ``_reset_cache()`` if needed.
        """
        calc = self.get_calculator()
        atoms = interface.interface
        atoms.calc = calc

        if relax:
            fmax = fmax if fmax is not None else self.fmax
            steps = steps if steps is not None else self.relax_steps
            self._relax(atoms, fmax=fmax, steps=steps)

        # Ensure the substrate copy is also up to date
        interface.substrate = atoms[: len(interface.substrate)]
        return interface

    def relax(
        self,
        interface: Interface,
        fmax: float | None = None,
        steps: int | None = None,
    ) -> Interface:
        """Relax the interface (same as evaluate with relax=True)."""
        return self.evaluate(interface, relax=True, fmax=fmax, steps=steps)

    def _relax(self, atoms, fmax: float, steps: int) -> None:
        """Run BFGS relaxation on *atoms* (mutated in-place)."""
        dyn = BFGS(atoms)
        try:
            dyn.run(fmax=fmax, steps=steps)
        except Exception:
            logger.warning("BFGS did not converge within %d steps (fmax=%.4f)", steps, fmax)
