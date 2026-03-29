"""VASP DFT interface (stub for v1.0).

Full VASP integration planned for future versions.
For now, users can use ASE's built-in Vasp calculator directly
via ASECalculatorEvaluator.
"""

from __future__ import annotations

from ase.calculators.calculator import Calculator

from gs_gmlip.evaluate.base import BaseEvaluator


class VASPEvaluator(BaseEvaluator):
    """VASP DFT evaluator (stub).

    For v1.0, this is a thin wrapper. Users should prefer using
    ASECalculatorEvaluator with ASE's Vasp calculator for full control.

    Parameters
    ----------
    command : str
        VASP execution command.
    kpts : tuple
        k-point grid.
    xc : str
        Exchange-correlation functional.
    kwargs : dict
        Additional VASP parameters.
    """

    def __init__(
        self,
        command: str = "vasp_std",
        kpts: tuple = (1, 1, 1),
        xc: str = "PBE",
        **kwargs,
    ):
        self.command = command
        self.kpts = kpts
        self.xc = xc
        self.kwargs = kwargs

    def get_calculator(self) -> Calculator:
        from ase.calculators.vasp import Vasp

        return Vasp(
            command=self.command,
            kpts=self.kpts,
            xc=self.xc,
            **self.kwargs,
        )
