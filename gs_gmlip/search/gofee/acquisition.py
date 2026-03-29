"""Acquisition functions for GOFEE.

Determine which candidate structures to evaluate next based on
the surrogate model predictions (mean, uncertainty).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def lower_confidence_bound(
    mean: np.ndarray, std: np.ndarray, kappa: float = 1.96
) -> np.ndarray:
    """Lower Confidence Bound (LCB) acquisition function.

    Selects candidates with low predicted energy and high uncertainty.
    LCB = mean - kappa * std  (minimize this for next evaluation).

    Parameters
    ----------
    mean : ndarray
        Predicted energies.
    std : ndarray
        Prediction uncertainties.
    kappa : float
        Exploration-exploitation trade-off. Higher = more exploration.

    Returns
    -------
    lcb : ndarray
        Acquisition values (lower = better candidate).
    """
    return mean - kappa * std


def expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    best_energy: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Expected Improvement (EI) acquisition function.

    Parameters
    ----------
    mean : ndarray
        Predicted energies.
    std : ndarray
        Prediction uncertainties.
    best_energy : float
        Best energy observed so far.
    xi : float
        Exploration parameter.

    Returns
    -------
    ei : ndarray
        Expected improvement values (higher = better candidate).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        improvement = best_energy - mean - xi
        Z = improvement / std
        ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
        # Where std is nearly zero, EI is zero
        ei = np.where(std > 1e-10, ei, 0.0)
    return ei


def probability_of_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    best_energy: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Probability of Improvement (PI) acquisition function.

    Parameters
    ----------
    mean, std : ndarray
        GPR predictions.
    best_energy : float
        Best energy so far.
    xi : float
        Exploration parameter.

    Returns
    -------
    pi : ndarray
        PI values (higher = better candidate).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = (best_energy - mean - xi) / std
        pi = norm.cdf(Z)
        pi = np.where(std > 1e-10, pi, 0.0)
    return pi


ACQUISITION_FUNCTIONS = {
    "lcb": lower_confidence_bound,
    "ei": expected_improvement,
    "pi": probability_of_improvement,
}
