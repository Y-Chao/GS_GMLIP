"""Trajectory analysis: convergence, energy evolution, and statistics.

Tools for analyzing search trajectories from GA, GCMC, SSW,
GOFEE, and Metadynamics runs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

logger = logging.getLogger(__name__)


class TrajectoryAnalyzer:
    """Analyze a structure search trajectory.

    Parameters
    ----------
    trajectory : list of dict
        Step-by-step records from a search runner.
    structures : list of Atoms, optional
        Corresponding structures at each step or collected minima.
    """

    def __init__(
        self,
        trajectory: list[dict],
        structures: list[Atoms] | None = None,
    ):
        self.trajectory = trajectory
        self.structures = structures or []

    @property
    def energies(self) -> np.ndarray:
        """Extract energy array from trajectory."""
        return np.array([s.get("energy", np.nan) for s in self.trajectory])

    @property
    def best_energies(self) -> np.ndarray:
        """Extract running best energy."""
        return np.array([s.get("best_energy", np.nan) for s in self.trajectory])

    @property
    def steps(self) -> np.ndarray:
        return np.array([s.get("step", i) for i, s in enumerate(self.trajectory)])

    def convergence_check(self, window: int = 50, threshold: float = 0.01) -> dict:
        """Check if the search has converged.

        Parameters
        ----------
        window : int
            Number of recent steps to consider.
        threshold : float
            Energy change threshold (eV) for convergence.

        Returns
        -------
        result : dict
            'converged': bool, 'best_energy': float,
            'energy_change': float over the window.
        """
        best = self.best_energies
        valid = best[~np.isnan(best)]
        if len(valid) < window:
            return {
                "converged": False,
                "best_energy": float(valid[-1]) if len(valid) > 0 else np.nan,
                "energy_change": np.nan,
                "reason": "insufficient_data",
            }

        recent = valid[-window:]
        change = abs(recent[-1] - recent[0])

        return {
            "converged": bool(change < threshold),
            "best_energy": float(valid[-1]),
            "energy_change": float(change),
            "n_steps": len(valid),
        }

    def acceptance_history(self) -> np.ndarray:
        """Extract acceptance boolean array."""

        def _to_bool(v):
            if isinstance(v, str):
                return v.lower() == "true"
            return bool(v)

        return np.array([_to_bool(s.get("accepted", False)) for s in self.trajectory])

    def rolling_acceptance_rate(self, window: int = 100) -> np.ndarray:
        """Compute rolling acceptance rate."""
        acc = self.acceptance_history().astype(float)
        if len(acc) < window:
            return np.array([acc.mean()])
        kernel = np.ones(window) / window
        return np.convolve(acc, kernel, mode="valid")

    def energy_distribution(self, n_bins: int = 50) -> tuple[np.ndarray, np.ndarray]:
        """Histogram of visited energies.

        Returns
        -------
        bin_centers : ndarray
        counts : ndarray
        """
        e = self.energies
        valid = e[~np.isnan(e)]
        if len(valid) == 0:
            return np.array([]), np.array([])
        counts, edges = np.histogram(valid, bins=n_bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, counts

    def summary(self) -> dict:
        """Generate a summary of the search trajectory."""
        e = self.energies
        valid_e = e[~np.isnan(e)]
        acc = self.acceptance_history()

        info = {
            "n_steps": len(self.trajectory),
            "n_structures": len(self.structures),
        }

        if len(valid_e) > 0:
            info.update(
                {
                    "energy_min": float(valid_e.min()),
                    "energy_max": float(valid_e.max()),
                    "energy_mean": float(valid_e.mean()),
                    "energy_std": float(valid_e.std()),
                }
            )

        if len(acc) > 0:
            info["acceptance_rate"] = float(acc.mean())

        conv = self.convergence_check()
        info["converged"] = conv["converged"]
        info["best_energy"] = conv.get("best_energy", np.nan)

        return info

    def save_trajectory(self, path: str | Path, format: str = "extxyz") -> None:
        """Save structures to file."""
        if self.structures:
            write(str(path), self.structures, format=format)
            logger.info("Saved %d structures to %s", len(self.structures), path)

    @classmethod
    def from_runner(cls, runner) -> "TrajectoryAnalyzer":
        """Create analyzer from a search runner object."""
        trajectory = getattr(runner, "trajectory", [])
        structures = []
        if hasattr(runner, "get_results"):
            structures = runner.get_results()
        return cls(trajectory=trajectory, structures=structures)
