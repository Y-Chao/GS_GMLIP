"""GA operators: crossover and mutation for Interface-based structures.

All operators work on ``Interface`` objects and only modify the appended
(adsorbate/cluster) region — the substrate is always preserved unchanged.

Refactored from the original ase_ga_official operators to use the new
Interface API (adsList / clusterList / add_adsorbate / remove_group).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
from ase.data import covalent_radii

from gs_gmlip.interface.core import Interface

logger = logging.getLogger(__name__)


def _appended_indices(interface: Interface) -> list[int]:
    """Return flat list of all appended (non-substrate) atom indices."""
    n_sub = len(interface.substrate)
    n_int = len(interface.interface)
    return list(range(n_sub, n_int))


def _check_min_distances(
    interface: Interface, blmin: dict[tuple[int, int], float]
) -> bool:
    """Check that all appended atoms satisfy *blmin* distance constraints."""
    atoms = interface.interface
    indices = _appended_indices(interface)
    if len(indices) < 2:
        return True

    numbers = atoms.get_atomic_numbers()
    for i_idx, i in enumerate(indices):
        for j in indices[i_idx + 1 :]:
            d = atoms.get_distance(i, j, mic=True)
            pair = (int(numbers[i]), int(numbers[j]))
            min_d = blmin.get(pair) or blmin.get((pair[1], pair[0]))
            if min_d is not None and d < min_d:
                return False
    return True


def _generate_blmin(
    atomic_numbers: list[int], scale: float = 0.7
) -> dict[tuple[int, int], float]:
    """Auto-generate minimum interatomic distances from covalent radii."""
    unique = list(set(atomic_numbers))
    blmin: dict[tuple[int, int], float] = {}
    for i, z1 in enumerate(unique):
        for z2 in unique[i:]:
            d = (covalent_radii[z1] + covalent_radii[z2]) * scale
            blmin[(z1, z2)] = d
            blmin[(z2, z1)] = d
    return blmin


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class OffspringCreator(ABC):
    """Base class for all genetic operators (crossover and mutation).

    Parameters
    ----------
    rng : np.random.Generator, optional
    """

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()
        self.descriptor: str = "unknown"
        self.min_inputs: int = 1

    @abstractmethod
    def get_new_individual(self, parents: list[Interface]) -> Interface | None:
        """Create a new individual from parent(s).

        Returns None if the operation could not produce a valid offspring.
        """

    def _copy_parent_metadata(self, parent: Interface, offspring: Interface) -> Interface:
        """Copy relevant metadata from parent to offspring."""
        return offspring


# ---------------------------------------------------------------------------
# Operation selector (weighted random choice)
# ---------------------------------------------------------------------------


class OperationSelector:
    """Randomly select genetic operators with weighted probabilities.

    Parameters
    ----------
    probabilities : list[float]
        Selection weights (normalized internally).
    operators : list[OffspringCreator]
        Corresponding operators.
    rng : np.random.Generator, optional
    """

    def __init__(
        self,
        probabilities: list[float],
        operators: list[OffspringCreator],
        rng: np.random.Generator | None = None,
    ):
        if len(probabilities) != len(operators):
            raise ValueError("probabilities and operators must have the same length")
        self.rng = rng or np.random.default_rng()
        total = sum(probabilities)
        self._cumulative = []
        s = 0.0
        for p in probabilities:
            s += p / total
            self._cumulative.append(s)
        self.operators = operators

    def get_operator(self) -> OffspringCreator:
        """Select an operator randomly according to weights."""
        r = self.rng.random()
        for i, cp in enumerate(self._cumulative):
            if r < cp:
                return self.operators[i]
        return self.operators[-1]


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------


class CutAndSpliceCrossover(OffspringCreator):
    """Deaven-Ho style cut-and-splice crossover for slab+adsorbate systems.

    Cuts two parent structures with a random plane and combines the halves.
    Only the appended (adsorbate/cluster) atoms participate; the substrate
    is taken from parent 1 unchanged.

    Parameters
    ----------
    blmin : dict
        Minimum interatomic distances {(Z1, Z2): distance}.
    rng : np.random.Generator, optional
    """

    def __init__(
        self,
        blmin: dict[tuple[int, int], float] | None = None,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(rng=rng)
        self.blmin = blmin or {}
        self.descriptor = "CutAndSpliceCrossover"
        self.min_inputs = 2

    def get_new_individual(self, parents: list[Interface]) -> Interface | None:
        if len(parents) < 2:
            return None
        return self._cross(parents[0], parents[1])

    def _cross(
        self, p1: Interface, p2: Interface, maxiter: int = 1000
    ) -> Interface | None:
        n1_sub = len(p1.substrate)
        n2_sub = len(p2.substrate)
        idx1 = _appended_indices(p1)
        idx2 = _appended_indices(p2)

        if not idx1 or not idx2:
            return None

        a1 = p1.interface
        a2 = p2.interface

        # Positions of appended atoms only
        pos1 = a1.positions[idx1]
        pos2 = a2.positions[idx2]
        com1 = pos1.mean(axis=0)
        com2 = pos2.mean(axis=0)

        for _ in range(maxiter):
            # Random cutting plane normal
            theta = self.rng.uniform(0, 2 * np.pi)
            phi = np.arccos(2 * self.rng.random() - 1)
            normal = np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi),
            ])

            # Cut point midway between the two centers, plus small random offset
            cut_point = (com1 + com2) / 2.0 + self.rng.normal(0, 0.5, 3)

            # Select atoms from each parent relative to the cutting plane
            above1 = np.dot(pos1 - cut_point, normal) > 0
            below2 = np.dot(pos2 - cut_point, normal) <= 0

            sel1 = [idx1[i] for i in np.where(above1)[0]]
            sel2 = [idx2[i] for i in np.where(below2)[0]]

            if not sel1 or not sel2:
                continue

            # Build offspring: start from p1, remove all appended, re-add selections
            offspring = p1.copy()
            # Remove all appended atoms
            all_appended = _appended_indices(offspring)
            if all_appended:
                offspring.remove_group(all_appended)

            # Add selected atoms from parent 1
            for idx in sorted(sel1):
                atom = a1[idx]
                offspring.interface += atom  # append single atom

            # Add selected atoms from parent 2
            for idx in sorted(sel2):
                atom = a2[idx]
                offspring.interface += atom

            # Re-classify appended region
            offspring._init_ads_cluster(None, None)
            offspring._reset_cache()

            if not self.blmin or _check_min_distances(offspring, self.blmin):
                return offspring

        return None


# ---------------------------------------------------------------------------
# Mutation operators
# ---------------------------------------------------------------------------


class RattleMutation(OffspringCreator):
    """Random Gaussian displacement of appended atoms.

    Each appended atom is displaced with probability *rattle_prop* by a
    vector drawn from N(0, *rattle_strength*).

    Parameters
    ----------
    blmin : dict
        Minimum interatomic distances for validation.
    rattle_strength : float
        Standard deviation of displacement (Å).
    rattle_prop : float
        Probability of rattling each appended atom.
    rng : np.random.Generator, optional
    """

    def __init__(
        self,
        blmin: dict[tuple[int, int], float] | None = None,
        rattle_strength: float = 0.8,
        rattle_prop: float = 0.4,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(rng=rng)
        self.blmin = blmin or {}
        self.rattle_strength = rattle_strength
        self.rattle_prop = rattle_prop
        self.descriptor = "RattleMutation"

    def get_new_individual(self, parents: list[Interface]) -> Interface | None:
        return self._mutate(parents[0])

    def _mutate(self, parent: Interface, maxiter: int = 1000) -> Interface | None:
        indices = _appended_indices(parent)
        if not indices:
            return None

        for _ in range(maxiter):
            offspring = parent.copy()
            for i in _appended_indices(offspring):
                if self.rng.random() < self.rattle_prop:
                    displacement = self.rng.normal(0, self.rattle_strength, 3)
                    offspring.interface.positions[i] += displacement

            offspring._reset_cache()
            if not self.blmin or _check_min_distances(offspring, self.blmin):
                return offspring

        return None


class PermutationMutation(OffspringCreator):
    """Swap positions of two appended atoms of different elements.

    Parameters
    ----------
    rng : np.random.Generator, optional
    """

    def __init__(self, rng: np.random.Generator | None = None):
        super().__init__(rng=rng)
        self.descriptor = "PermutationMutation"

    def get_new_individual(self, parents: list[Interface]) -> Interface | None:
        parent = parents[0]
        indices = _appended_indices(parent)
        if len(indices) < 2:
            return None

        numbers = parent.interface.get_atomic_numbers()
        # Find pairs of different-element atoms
        pairs = [
            (i, j)
            for i_idx, i in enumerate(indices)
            for j in indices[i_idx + 1 :]
            if numbers[i] != numbers[j]
        ]
        if not pairs:
            return None

        offspring = parent.copy()
        i, j = pairs[self.rng.integers(len(pairs))]
        pos = offspring.interface.positions
        pos[[i, j]] = pos[[j, i]]
        offspring._reset_cache()
        return offspring


class MirrorMutation(OffspringCreator):
    """Mirror appended atoms on one side of a random plane through their COM.

    Parameters
    ----------
    blmin : dict
        Minimum interatomic distances for validation.
    rng : np.random.Generator, optional
    """

    def __init__(
        self,
        blmin: dict[tuple[int, int], float] | None = None,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(rng=rng)
        self.blmin = blmin or {}
        self.descriptor = "MirrorMutation"

    def get_new_individual(self, parents: list[Interface]) -> Interface | None:
        return self._mutate(parents[0])

    def _mutate(self, parent: Interface, maxiter: int = 1000) -> Interface | None:
        indices = _appended_indices(parent)
        if len(indices) < 2:
            return None

        com = parent.interface.positions[indices].mean(axis=0)

        for _ in range(maxiter):
            offspring = parent.copy()
            # Random mirror plane normal
            normal = self.rng.normal(0, 1, 3)
            normal /= np.linalg.norm(normal)

            for i in _appended_indices(offspring):
                d = np.dot(offspring.interface.positions[i] - com, normal)
                if d > 0:
                    offspring.interface.positions[i] -= 2 * d * normal

            offspring._reset_cache()
            if not self.blmin or _check_min_distances(offspring, self.blmin):
                return offspring

        return None


class TwistMutation(OffspringCreator):
    """Rotate appended atoms around the surface normal (z-axis) by a random angle.

    This operator is motivated by the paper's GCGA method, where adsorbed
    clusters are rotated by 1°–180° around the surface normal.  Useful for
    exploring rotational degrees of freedom of clusters on surfaces.

    Parameters
    ----------
    blmin : dict
        Minimum interatomic distances for validation.
    angle_range : tuple[float, float]
        (min, max) rotation angle in degrees. Default (1, 180).
    rng : np.random.Generator, optional
    """

    def __init__(
        self,
        blmin: dict[tuple[int, int], float] | None = None,
        angle_range: tuple[float, float] = (1.0, 180.0),
        rng: np.random.Generator | None = None,
    ):
        super().__init__(rng=rng)
        self.blmin = blmin or {}
        self.angle_range = angle_range
        self.descriptor = "TwistMutation"

    def get_new_individual(self, parents: list[Interface]) -> Interface | None:
        return self._mutate(parents[0])

    def _mutate(self, parent: Interface, maxiter: int = 1000) -> Interface | None:
        indices = _appended_indices(parent)
        if not indices:
            return None

        for _ in range(maxiter):
            offspring = parent.copy()
            # Random rotation angle
            angle = self.rng.uniform(*self.angle_range)

            # Rotate around z-axis through the center of mass of appended atoms
            com = offspring.interface.positions[indices].mean(axis=0)
            # z-axis is [0, 0, 1]; rotate in-place
            for i in _appended_indices(offspring):
                pos = offspring.interface.positions[i] - com
                rad = np.radians(angle)
                cos_a, sin_a = np.cos(rad), np.sin(rad)
                new_x = cos_a * pos[0] - sin_a * pos[1]
                new_y = sin_a * pos[0] + cos_a * pos[1]
                offspring.interface.positions[i] = com + np.array([new_x, new_y, pos[2]])

            offspring._reset_cache()
            if not self.blmin or _check_min_distances(offspring, self.blmin):
                return offspring

        return None
