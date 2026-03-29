"""GA operators: crossover, mutation, and operation selection.

Refactored from ase_ga_official: offspring_creator.py, standardmutations.py,
cutandsplicepairing.py. Provides the core genetic operators for structure
optimization on surfaces.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
from ase import Atoms
from ase.data import covalent_radii

logger = logging.getLogger(__name__)


class OffspringCreator(ABC):
    """Base class for all genetic operators.

    All crossover and mutation operators inherit from this class
    and implement get_new_individual().
    """

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()
        self.descriptor = "unknown"
        self.min_inputs = 1

    @abstractmethod
    def get_new_individual(self, parents: list[Atoms]) -> Atoms | None:
        """Create a new individual from parent(s).

        Parameters
        ----------
        parents : list of Atoms
            Parent structures.

        Returns
        -------
        offspring : Atoms or None
            New structure, or None if operation failed.
        """

    def initialize_individual(self, parent: Atoms, indi: Atoms | None = None) -> Atoms:
        """Initialize offspring info from parent."""
        if indi is None:
            indi = parent.copy()
        indi.info.setdefault("key_value_pairs", {})
        indi.info.setdefault("data", {})
        return indi

    def finalize_individual(self, indi: Atoms) -> Atoms:
        """Finalize offspring metadata."""
        indi.info["key_value_pairs"]["origin"] = self.descriptor
        return indi


class OperationSelector:
    """Randomly select genetic operators with weighted probabilities.

    Parameters
    ----------
    probabilities : list of float
        Selection weights (will be normalized to sum to 1).
    operators : list of OffspringCreator
        Corresponding operators.
    rng : Generator, optional
    """

    def __init__(
        self,
        probabilities: list[float],
        operators: list[OffspringCreator],
        rng: np.random.Generator | None = None,
    ):
        assert len(probabilities) == len(operators)
        self.rng = rng or np.random.default_rng()
        total = sum(probabilities)
        self.cumulative_probs = []
        s = 0.0
        for p in probabilities:
            s += p / total
            self.cumulative_probs.append(s)
        self.operators = operators

    def get_operator(self) -> OffspringCreator:
        """Select an operator randomly according to weights."""
        r = self.rng.random()
        for i, cp in enumerate(self.cumulative_probs):
            if r < cp:
                return self.operators[i]
        return self.operators[-1]


def _check_min_distances(atoms: Atoms, blmin: dict) -> bool:
    """Check if all interatomic distances satisfy blmin constraints.

    Parameters
    ----------
    atoms : Atoms
    blmin : dict
        {(Z1, Z2): min_distance}

    Returns
    -------
    valid : bool
        True if all distances satisfy constraints.
    """
    numbers = atoms.get_atomic_numbers()
    n = len(atoms)
    if n < 2:
        return True

    distances = atoms.get_all_distances(mic=True)
    for i in range(n):
        for j in range(i + 1, n):
            pair = (numbers[i], numbers[j])
            min_d = blmin.get(pair) or blmin.get((pair[1], pair[0]))
            if min_d is not None and distances[i, j] < min_d:
                return False
    return True


# --- Crossover Operators ---


class CutAndSplicePairing(OffspringCreator):
    """Cut-and-splice crossover for slab systems.

    Cuts two parent structures with a random plane and combines
    the halves to create offspring. Adapted from Deaven-Ho method.

    Parameters
    ----------
    slab : Atoms
        The fixed slab (tag=0 atoms).
    n_top : int
        Number of atoms in the active region.
    blmin : dict
        Minimum interatomic distances {(Z1, Z2): distance}.
    use_tags : bool
        Preserve molecular identity via tags.
    rng : Generator, optional
    """

    def __init__(
        self,
        slab: Atoms,
        n_top: int,
        blmin: dict,
        use_tags: bool = False,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(rng=rng)
        self.slab = slab
        self.n_top = n_top
        self.blmin = blmin
        self.use_tags = use_tags
        self.descriptor = "CutAndSplicePairing"
        self.min_inputs = 2

    def get_new_individual(self, parents: list[Atoms]) -> Atoms | None:
        if len(parents) < 2:
            return None
        a1, a2 = parents[0], parents[1]
        return self._cross(a1, a2)

    def _cross(self, a1: Atoms, a2: Atoms, maxiter: int = 1000) -> Atoms | None:
        """Perform the cut-and-splice operation."""
        for _ in range(maxiter):
            # Get active atoms from both parents
            n = len(a1)
            n_slab = n - self.n_top
            top1 = a1[n_slab:]
            top2 = a2[n_slab:]

            if len(top1) == 0 or len(top2) == 0:
                return None

            # Random cutting plane (normal vector)
            theta = self.rng.uniform(0, 2 * np.pi)
            phi = np.arccos(2 * self.rng.random() - 1)
            normal = np.array(
                [
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi),
                ]
            )

            # Cut point = center of mass of combined top atoms
            com = (top1.get_center_of_mass() + top2.get_center_of_mass()) / 2
            # Shift slightly
            com += self.rng.normal(0, 0.5, 3)

            # Select atoms from each parent
            pos1 = top1.get_positions()
            pos2 = top2.get_positions()
            above1 = np.dot(pos1 - com, normal) > 0
            below2 = np.dot(pos2 - com, normal) <= 0

            # Build offspring: slab + selected from parent1 + selected from parent2
            offspring = self.slab.copy()
            sel1_indices = np.where(above1)[0]
            sel2_indices = np.where(below2)[0]

            if len(sel1_indices) == 0 or len(sel2_indices) == 0:
                continue

            for idx in sel1_indices:
                offspring.append(top1[idx])
            for idx in sel2_indices:
                offspring.append(top2[idx])

            # Check distance constraints
            if _check_min_distances(offspring, self.blmin):
                offspring = self.initialize_individual(a1, offspring)
                offspring.info["data"]["parents"] = [
                    a1.info.get("confid", -1),
                    a2.info.get("confid", -1),
                ]
                return self.finalize_individual(offspring)

        return None


# --- Mutation Operators ---


class RattleMutation(OffspringCreator):
    """Random displacement of atoms in the active region.

    Parameters
    ----------
    n_top : int
        Number of active atoms.
    blmin : dict
        Minimum interatomic distances.
    rattle_strength : float
        Standard deviation of Gaussian displacement (Angstrom).
    rattle_prop : float
        Probability of rattling each atom.
    rng : Generator, optional
    """

    def __init__(
        self,
        n_top: int,
        blmin: dict,
        rattle_strength: float = 0.8,
        rattle_prop: float = 0.4,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(rng=rng)
        self.n_top = n_top
        self.blmin = blmin
        self.rattle_strength = rattle_strength
        self.rattle_prop = rattle_prop
        self.descriptor = "RattleMutation"

    def get_new_individual(self, parents: list[Atoms]) -> Atoms | None:
        parent = parents[0]
        return self._mutate(parent)

    def _mutate(self, atoms: Atoms, maxiter: int = 1000) -> Atoms | None:
        for _ in range(maxiter):
            indi = atoms.copy()
            n = len(indi)
            n_slab = n - self.n_top

            for i in range(n_slab, n):
                if self.rng.random() < self.rattle_prop:
                    displacement = self.rng.normal(0, self.rattle_strength, 3)
                    indi.positions[i] += displacement

            if _check_min_distances(indi, self.blmin):
                indi = self.initialize_individual(atoms, indi)
                indi.info["data"]["parents"] = [atoms.info.get("confid", -1)]
                return self.finalize_individual(indi)

        return None


class PermutationMutation(OffspringCreator):
    """Swap positions of different element atoms in the active region.

    Parameters
    ----------
    n_top : int
        Number of active atoms.
    rng : Generator, optional
    """

    def __init__(self, n_top: int, rng: np.random.Generator | None = None):
        super().__init__(rng=rng)
        self.n_top = n_top
        self.descriptor = "PermutationMutation"

    def get_new_individual(self, parents: list[Atoms]) -> Atoms | None:
        parent = parents[0]
        indi = parent.copy()
        n = len(indi)
        n_slab = n - self.n_top
        top_indices = list(range(n_slab, n))

        if len(top_indices) < 2:
            return None

        # Find pairs of atoms with different elements
        numbers = indi.get_atomic_numbers()
        pairs = []
        for i in top_indices:
            for j in top_indices:
                if i < j and numbers[i] != numbers[j]:
                    pairs.append((i, j))

        if not pairs:
            return None

        # Swap a random pair
        i, j = pairs[self.rng.integers(len(pairs))]
        indi.positions[[i, j]] = indi.positions[[j, i]]

        indi = self.initialize_individual(parent, indi)
        indi.info["data"]["parents"] = [parent.info.get("confid", -1)]
        return self.finalize_individual(indi)


class MirrorMutation(OffspringCreator):
    """Mirror half of the active atoms across a random plane.

    Parameters
    ----------
    n_top : int
        Number of active atoms.
    blmin : dict
        Minimum interatomic distances.
    rng : Generator, optional
    """

    def __init__(
        self,
        n_top: int,
        blmin: dict,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(rng=rng)
        self.n_top = n_top
        self.blmin = blmin
        self.descriptor = "MirrorMutation"

    def get_new_individual(self, parents: list[Atoms]) -> Atoms | None:
        parent = parents[0]
        return self._mutate(parent)

    def _mutate(self, atoms: Atoms, maxiter: int = 1000) -> Atoms | None:
        for _ in range(maxiter):
            indi = atoms.copy()
            n = len(indi)
            n_slab = n - self.n_top
            top_indices = list(range(n_slab, n))

            if len(top_indices) < 2:
                return None

            # Random mirror plane through center of mass
            top = indi[top_indices]
            com = top.get_center_of_mass()

            # Random normal
            normal = self.rng.normal(0, 1, 3)
            normal /= np.linalg.norm(normal)

            # Mirror atoms on one side
            for idx in top_indices:
                d = np.dot(indi.positions[idx] - com, normal)
                if d > 0:
                    indi.positions[idx] -= 2 * d * normal

            if _check_min_distances(indi, self.blmin):
                indi = self.initialize_individual(atoms, indi)
                indi.info["data"]["parents"] = [atoms.info.get("confid", -1)]
                return self.finalize_individual(indi)

        return None
