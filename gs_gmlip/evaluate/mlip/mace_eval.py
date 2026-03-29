"""MACE machine learning interatomic potential evaluator."""

from __future__ import annotations

import logging

from ase.calculators.calculator import Calculator

from gs_gmlip.evaluate.base import BaseEvaluator

logger = logging.getLogger(__name__)


class MACEEvaluator(BaseEvaluator):
    """Evaluator using MACE machine learning potential.

    Parameters
    ----------
    model : str
        Path to MACE model file, or one of the foundation models:
        "small", "medium", "large" for MACE-MP-0.
    device : str
        Device to run on: "cpu", "cuda", "mps".
    default_dtype : str
        Default dtype: "float32" or "float64".
    dispersion : bool
        Whether to include D3 dispersion correction.
    """

    def __init__(
        self,
        model: str = "medium",
        device: str = "cpu",
        default_dtype: str = "float64",
        dispersion: bool = False,
    ):
        self.model = model
        self.device = device
        self.default_dtype = default_dtype
        self.dispersion = dispersion
        self._calc = None

    def get_calculator(self) -> Calculator:
        if self._calc is None:
            try:
                from mace.calculators import mace_mp

                self._calc = mace_mp(
                    model=self.model,
                    device=self.device,
                    default_dtype=self.default_dtype,
                    dispersion=self.dispersion,
                )
                logger.info(
                    f"Loaded MACE-MP model={self.model} on device={self.device}"
                )
            except ImportError:
                raise ImportError(
                    "MACE is not installed. Install with: pip install gs-gmlip[mlip]"
                )
        return self._calc
