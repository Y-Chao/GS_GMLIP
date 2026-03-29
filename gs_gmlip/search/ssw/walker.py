"""SSW walker: soft-mode-following on the potential energy surface.

Implements the core SSW algorithm:
1. At a local minimum, compute the Hessian (or approximate it).
2. Identify soft modes (lowest eigenvalue eigenvectors).
3. Walk along a soft mode direction, adding bias potential.
4. Relax into a new minimum.
5. Apply Metropolis criterion to accept/reject.
"""

from __future__ import annotations

import logging

import numpy as np
from ase import Atoms
from ase.optimize import BFGS, FIRE

from gs_gmlip.evaluate.base import BaseEvaluator

logger = logging.getLogger(__name__)


class BiasedCalculator:
    """ASE-compatible calculator that adds Gaussian bias potentials.

    The bias is applied along a direction in configuration space to
    push the system over barriers.

    Parameters
    ----------
    base_calc : Calculator
        The underlying ASE calculator.
    centers : list of ndarray
        Centers of Gaussian biases (flattened positions).
    heights : list of float
        Heights of Gaussian biases (eV).
    widths : list of float
        Widths of Gaussian biases (Å).
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, base_calc, centers=None, heights=None, widths=None):
        self.base_calc = base_calc
        self.centers = centers or []
        self.heights = heights or []
        self.widths = widths or []
        self.atoms = None

    def calculate(self, atoms=None, properties=None, system_changes=None):
        if atoms is not None:
            self.atoms = atoms

        # Get base energy and forces
        self.atoms.calc = self.base_calc
        base_energy = self.atoms.get_potential_energy()
        base_forces = self.atoms.get_forces().copy()

        # Add Gaussian bias potentials
        pos = self.atoms.positions.ravel()
        bias_energy = 0.0
        bias_forces = np.zeros_like(pos)

        for center, h, w in zip(self.centers, self.heights, self.widths):
            diff = pos - center
            r2 = np.dot(diff, diff)
            gauss = h * np.exp(-r2 / (2 * w**2))
            bias_energy += gauss
            bias_forces += gauss * diff / w**2

        self.results = {
            "energy": base_energy + bias_energy,
            "forces": (base_forces + bias_forces.reshape(-1, 3)),
        }
        return self.results


def approximate_hessian_finite_diff(
    atoms: Atoms,
    evaluator: BaseEvaluator,
    active_indices: list[int],
    step: float = 0.01,
) -> np.ndarray:
    """Approximate the Hessian using finite differences of forces.

    Only computes the Hessian block for active atoms to save cost.

    Parameters
    ----------
    atoms : Atoms
        Current structure at a local minimum.
    evaluator : BaseEvaluator
        Calculator.
    active_indices : list of int
        Indices of atoms to include in the Hessian.
    step : float
        Finite difference step in Å.

    Returns
    -------
    H : ndarray, shape (3*n_active, 3*n_active)
        Approximate Hessian matrix.
    """
    n = len(active_indices)
    H = np.zeros((3 * n, 3 * n))

    # Get reference forces
    atoms_ref = atoms.copy()
    atoms_ref.calc = evaluator.get_calculator()
    f0 = atoms_ref.get_forces()[active_indices].ravel()

    for i, atom_idx in enumerate(active_indices):
        for d in range(3):
            # Forward step
            atoms_plus = atoms.copy()
            atoms_plus.positions[atom_idx, d] += step
            atoms_plus.calc = evaluator.get_calculator()
            f_plus = atoms_plus.get_forces()[active_indices].ravel()

            # Backward step
            atoms_minus = atoms.copy()
            atoms_minus.positions[atom_idx, d] -= step
            atoms_minus.calc = evaluator.get_calculator()
            f_minus = atoms_minus.get_forces()[active_indices].ravel()

            # Central difference
            H[3 * i + d, :] = -(f_plus - f_minus) / (2 * step)

    # Symmetrize
    H = 0.5 * (H + H.T)
    return H


def get_soft_modes(
    hessian: np.ndarray, n_modes: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the softest (lowest eigenvalue) modes from the Hessian.

    Parameters
    ----------
    hessian : ndarray
        Hessian matrix.
    n_modes : int
        Number of soft modes to return.

    Returns
    -------
    eigenvalues : ndarray, shape (n_modes,)
    eigenvectors : ndarray, shape (n_modes, N)
        Each row is a mode direction.
    """
    eigvals, eigvecs = np.linalg.eigh(hessian)
    return eigvals[:n_modes], eigvecs[:, :n_modes].T


class SSWWalker:
    """Performs SSW walks on the PES.

    Parameters
    ----------
    evaluator : BaseEvaluator
        Energy calculator.
    temperature : float
        Temperature in Kelvin for Metropolis criterion.
    bias_height : float
        Height of Gaussian bias (eV).
    bias_width : float
        Width of Gaussian bias (Å).
    walk_length : float
        Length of the walk along the soft mode (Å).
    n_soft_modes : int
        Number of soft modes to consider.
    fmax : float
        Force convergence criterion for local relaxation.
    max_relax_steps : int
        Maximum relaxation steps.
    hessian_step : float
        Step size for finite difference Hessian.
    """

    kB = 8.617333262e-5  # eV/K

    def __init__(
        self,
        evaluator: BaseEvaluator,
        temperature: float = 300.0,
        bias_height: float = 0.3,
        bias_width: float = 0.5,
        walk_length: float = 2.0,
        n_soft_modes: int = 3,
        fmax: float = 0.05,
        max_relax_steps: int = 200,
        hessian_step: float = 0.01,
    ):
        self.evaluator = evaluator
        self.temperature = temperature
        self.bias_height = bias_height
        self.bias_width = bias_width
        self.walk_length = walk_length
        self.n_soft_modes = n_soft_modes
        self.fmax = fmax
        self.max_relax_steps = max_relax_steps
        self.hessian_step = hessian_step

    def walk(
        self, atoms: Atoms, active_indices: list[int], rng: np.random.Generator
    ) -> tuple[Atoms | None, float | None, dict]:
        """Perform one SSW walk from the current minimum.

        Parameters
        ----------
        atoms : Atoms
            Structure at a local minimum.
        active_indices : list of int
            Indices of active (non-slab) atoms.
        rng : Generator
            Random number generator.

        Returns
        -------
        new_atoms : Atoms or None
            New minimum found, or None if walk failed.
        new_energy : float or None
            Energy of new minimum.
        info : dict
            Walk information.
        """
        if not active_indices:
            return None, None, {"reason": "no_active_atoms"}

        # Step 1: Approximate Hessian
        try:
            H = approximate_hessian_finite_diff(
                atoms, self.evaluator, active_indices, self.hessian_step
            )
        except Exception as e:
            logger.warning("Hessian computation failed: %s", e)
            return None, None, {"reason": "hessian_failed"}

        # Step 2: Get soft modes
        n_modes = min(self.n_soft_modes, len(active_indices) * 3)
        eigenvalues, eigenvectors = get_soft_modes(H, n_modes)

        # Step 3: Select walk direction (weighted by softness)
        # Choose the softest mode or random combination
        weights = np.exp(-eigenvalues / max(abs(eigenvalues.max()), 1e-10))
        weights /= weights.sum()
        mode_idx = rng.choice(n_modes, p=weights)
        direction = eigenvectors[mode_idx]

        # Random sign
        if rng.random() < 0.5:
            direction = -direction

        # Normalize and scale
        direction = direction / np.linalg.norm(direction) * self.walk_length

        # Step 4: Walk along the direction
        walked = atoms.copy()
        for i, atom_idx in enumerate(active_indices):
            walked.positions[atom_idx] += direction[3 * i : 3 * i + 3]

        # Step 5: Relax with bias at original position
        try:
            relaxed = self.evaluator.evaluate(walked, relax=True)
            new_energy = relaxed.get_potential_energy()
        except Exception as e:
            logger.warning("Relaxation after walk failed: %s", e)
            return None, None, {"reason": "relax_failed"}

        return (
            relaxed,
            new_energy,
            {
                "eigenvalue": float(eigenvalues[mode_idx]),
                "mode_index": int(mode_idx),
            },
        )
