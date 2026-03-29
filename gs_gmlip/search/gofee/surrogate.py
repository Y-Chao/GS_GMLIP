"""Gaussian Process surrogate model for GOFEE.

Uses fingerprint-based descriptors with sklearn GPR to model the
potential energy surface and provide uncertainty estimates.
"""

from __future__ import annotations

import logging

import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)


def get_fingerprint(
    atoms: Atoms, active_indices: list[int] | None = None
) -> np.ndarray:
    """Compute a radial distribution fingerprint for a structure.

    Uses pair distances between active atoms and all atoms as features.

    Parameters
    ----------
    atoms : Atoms
        Input structure.
    active_indices : list[int], optional
        Indices of active (non-slab) atoms. If None, all atoms are used.

    Returns
    -------
    fp : ndarray, shape (n_bins,)
        Radial distribution fingerprint.
    """
    if active_indices is None:
        active_indices = list(range(len(atoms)))

    if len(active_indices) == 0:
        return np.zeros(64)

    # Pair distances from active atoms to all atoms
    from ase.geometry import get_distances

    pos_active = atoms.positions[active_indices]
    pos_all = atoms.positions
    _, D = get_distances(pos_active, pos_all, cell=atoms.cell, pbc=atoms.pbc)

    # Flatten and remove self-distances (zeros)
    dists = D.ravel()
    dists = dists[dists > 0.1]

    # Histogram as fingerprint
    bins = np.linspace(0.5, 8.0, 65)
    fp, _ = np.histogram(dists, bins=bins, density=True)
    return fp


class SurrogateModel:
    """Gaussian Process Regression surrogate for the PES.

    Parameters
    ----------
    kernel : str
        Kernel type: 'rbf' (default) or 'matern'.
    length_scale : float
        Length scale for the kernel.
    alpha : float
        Noise level for GPR.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        length_scale: float = 1.0,
        alpha: float = 0.1,
    ):
        self.kernel_type = kernel
        self.length_scale = length_scale
        self.alpha = alpha
        self.gpr = None
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self._fitted = False

    def _build_gpr(self):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel

        if self.kernel_type == "matern":
            kernel = Matern(length_scale=self.length_scale, nu=2.5) + WhiteKernel(
                noise_level=self.alpha
            )
        else:
            kernel = RBF(length_scale=self.length_scale) + WhiteKernel(
                noise_level=self.alpha
            )

        self.gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            normalize_y=True,
        )

    def update(self, structures: list[Atoms], energies: list[float]) -> None:
        """Update the surrogate with new training data.

        Parameters
        ----------
        structures : list of Atoms
            Training structures.
        energies : list of float
            Corresponding energies.
        """
        X_new = np.array([get_fingerprint(s) for s in structures])
        y_new = np.array(energies)

        if self.X_train is not None:
            self.X_train = np.vstack([self.X_train, X_new])
            self.y_train = np.concatenate([self.y_train, y_new])
        else:
            self.X_train = X_new
            self.y_train = y_new

        self._fit()

    def _fit(self) -> None:
        """Fit the GPR model."""
        if self.gpr is None:
            self._build_gpr()

        self.gpr.fit(self.X_train, self.y_train)
        self._fitted = True
        logger.debug("GPR fitted with %d training points", len(self.y_train))

    def predict(self, structures: list[Atoms]) -> tuple[np.ndarray, np.ndarray]:
        """Predict energy and uncertainty for structures.

        Returns
        -------
        mean : ndarray, shape (n,)
            Predicted energies.
        std : ndarray, shape (n,)
            Prediction uncertainties.
        """
        if not self._fitted:
            raise RuntimeError("Surrogate not fitted yet.")

        X = np.array([get_fingerprint(s) for s in structures])
        mean, std = self.gpr.predict(X, return_std=True)
        return mean, std

    def predict_single(self, atoms: Atoms) -> tuple[float, float]:
        """Predict for a single structure."""
        mean, std = self.predict([atoms])
        return float(mean[0]), float(std[0])
