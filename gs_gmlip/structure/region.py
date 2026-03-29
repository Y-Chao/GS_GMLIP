"""Region definition for atom insertion/deletion during global search.

Defines spatial regions (box, sphere) above the slab surface where
atoms/molecules can be added or removed during search operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from ase import Atoms


class Region(ABC):
    """Abstract base class for spatial regions.

    A Region defines where new atoms/molecules can be placed during
    search operations (GA, GCMC, etc.).
    """

    @abstractmethod
    def random_position(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Generate a random position within this region.

        Returns
        -------
        position : ndarray, shape (3,)
        """

    @abstractmethod
    def contains(self, position: np.ndarray) -> bool:
        """Check if a position is within the region."""

    def random_positions(
        self, n: int, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Generate n random positions within the region.

        Returns
        -------
        positions : ndarray, shape (n, 3)
        """
        return np.array([self.random_position(rng) for _ in range(n)])


class BoxRegion(Region):
    """Rectangular box region, typically above a slab surface.

    Parameters
    ----------
    cell : ndarray, shape (3, 3)
        The cell vectors defining the lateral extent.
    z_min : float
        Minimum height (above slab top).
    z_max : float
        Maximum height.
    origin : ndarray, shape (3,), optional
        Origin of the box. Default is [0, 0, 0].
    """

    def __init__(
        self,
        cell: np.ndarray,
        z_min: float,
        z_max: float,
        origin: np.ndarray | None = None,
        pbc: np.ndarray | None = None,
    ):
        self.cell = np.asarray(cell, dtype=float)
        self.z_min = z_min
        self.z_max = z_max
        self.origin = np.zeros(3) if origin is None else np.asarray(origin, dtype=float)
        self.pbc = pbc

    @classmethod
    def from_atoms(
        cls, atoms: Atoms, z_min: float, z_max: float, padding: float = 0.0
    ) -> BoxRegion:
        """Create BoxRegion from atoms object.

        The lateral extent is taken from the cell, and z_min/z_max
        are absolute heights.
        """
        cell = atoms.get_cell()
        return cls(cell=cell, z_min=z_min, z_max=z_max)

    def random_position(self, rng: np.random.Generator | None = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        # Random fractional coordinates for x, y
        s1, s2 = rng.random(2)
        xy = s1 * self.cell[0, :2] + s2 * self.cell[1, :2]
        z = rng.uniform(self.z_min, self.z_max)
        return self.origin + np.array([xy[0], xy[1], z])

    def contains(self, position: np.ndarray) -> bool:
        pos = np.asarray(position) - self.origin
        z = pos[2]
        if z < self.z_min or z > self.z_max:
            return False
        # Check if xy is within the cell (using fractional coordinates)
        cell_2d = self.cell[:2, :2]
        try:
            frac = np.linalg.solve(cell_2d.T, pos[:2])
        except np.linalg.LinAlgError:
            return False
        return np.all(frac >= 0) and np.all(frac <= 1)


class SphereRegion(Region):
    """Spherical region.

    Parameters
    ----------
    center : ndarray, shape (3,)
        Center of the sphere.
    radius : float
        Radius of the sphere.
    """

    def __init__(self, center: np.ndarray, radius: float):
        self.center = np.asarray(center, dtype=float)
        self.radius = radius

    def random_position(self, rng: np.random.Generator | None = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        # Uniform random point in sphere
        u = rng.random()
        r = self.radius * u ** (1.0 / 3.0)
        theta = rng.uniform(0, 2 * np.pi)
        phi = np.arccos(2 * rng.random() - 1)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return self.center + np.array([x, y, z])

    def contains(self, position: np.ndarray) -> bool:
        return np.linalg.norm(np.asarray(position) - self.center) <= self.radius
