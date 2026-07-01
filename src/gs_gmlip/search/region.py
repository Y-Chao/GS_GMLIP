"""Regions for random atom placement during structure generation.

Regions define where new atoms / molecules can be placed relative to
the slab surface.  Used by GA start generators and GCMC insert moves.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Region(ABC):
    """Abstract region for random position sampling."""

    @abstractmethod
    def random_position(self, rng: np.random.Generator) -> np.ndarray:
        """Return a random (3,) position inside the region."""


class BoxRegion(Region):
    """Axis-aligned box region above a slab.

    Parameters
    ----------
    cell : np.ndarray
        (3, 3) cell matrix defining the in-plane bounds.
    z_min : float
        Minimum z coordinate (above slab top).
    z_max : float
        Maximum z coordinate.
    pbc : list[bool]
        Periodic boundary conditions (defaults to [True, True, False]).
    margin : float
        In-plane margin inside the cell edges (Å).
    """

    def __init__(
        self,
        cell: np.ndarray,
        z_min: float = 0.0,
        z_max: float = 10.0,
        pbc: list[bool] | None = None,
        margin: float = 1.0,
    ):
        self.cell = np.asarray(cell)
        self.z_min = z_min
        self.z_max = z_max
        self.pbc = pbc if pbc is not None else [True, True, False]
        self.margin = margin

    def random_position(self, rng: np.random.Generator) -> np.ndarray:
        """Pick a random (x, y, z) within the box."""
        a_vec = self.cell[0]
        b_vec = self.cell[1]
        x = rng.uniform(self.margin, np.linalg.norm(a_vec) - self.margin)
        y = rng.uniform(self.margin, np.linalg.norm(b_vec) - self.margin)
        z = rng.uniform(self.z_min, self.z_max)
        return np.array([x, y, z])


class SphereRegion(Region):
    """Spherical region around a center point.

    Parameters
    ----------
    center : np.ndarray
        (3,) center of the sphere.
    radius : float
        Radius of the sphere (Å).
    """

    def __init__(self, center: np.ndarray, radius: float = 5.0):
        self.center = np.asarray(center)
        self.radius = radius

    def random_position(self, rng: np.random.Generator) -> np.ndarray:
        """Pick a random point uniformly inside the sphere."""
        # Rejection-free: sample in spherical coordinates
        u = rng.random()
        r = self.radius * np.cbrt(u)  # uniform in volume
        theta = rng.uniform(0, 2 * np.pi)
        phi = np.arccos(2 * rng.random() - 1)
        offset = np.array([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi),
        ])
        return self.center + offset
