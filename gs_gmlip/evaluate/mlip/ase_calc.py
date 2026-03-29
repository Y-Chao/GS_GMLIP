"""Generic ASE Calculator adapter.

Allows any ASE-compatible Calculator to be used as an evaluator.
"""

from __future__ import annotations

from ase.calculators.calculator import Calculator

from gs_gmlip.evaluate.base import BaseEvaluator


class ASECalculatorEvaluator(BaseEvaluator):
    """Evaluator wrapping any existing ASE Calculator instance.

    Parameters
    ----------
    calculator : Calculator
        An ASE Calculator instance (e.g., EMT, VASP, etc.).
    """

    def __init__(self, calculator: Calculator):
        self._calc = calculator

    def get_calculator(self) -> Calculator:
        return self._calc
