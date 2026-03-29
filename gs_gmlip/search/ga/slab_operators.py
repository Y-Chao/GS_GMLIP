"""Slab-specific GA operators for surface structure optimization.

Specialized crossover and mutation operators for slab+adsorbate systems.
Handles composition constraints, element pools, and surface-specific
operations. Refactored from ase_ga_official/slab_operators.py.
"""

from __future__ import annotations

import logging

import numpy as np
from ase import Atoms

from gs_gmlip.search.ga.operators import OffspringCreator, _check_min_distances

logger = logging.getLogger(__name__)


class CutSpliceSlabCrossover(OffspringCreator):
    """Crossover for slab systems via random cutting plane.

    Slices two parent surfaces and combines halves, keeping
    the slab fixed and only modifying the active region.

    Parameters
    ----------
    slab : Atoms
        Fixed slab atoms (tag=0).
    n_top : int
        Number of active atoms.
    blmin : dict
        Minimum interatomic distances.
    allowed_compositions : list of dict, optional
        List of allowed compositions, e.g. [{"H": 4, "O": 2}, {"H": 2, "O": 1}].
    rng : Generator, optional
    """

    def __init__(
        self,
        slab: Atoms,
        n_top: int,
        blmin: dict,
        allowed_compositions: list[dict[str, int]] | None = None,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(rng=rng)
        self.slab = slab
        self.n_top = n_top
        self.blmin = blmin
        self.allowed_compositions = allowed_compositions
        self.descriptor = "CutSpliceSlabCrossover"
        self.min_inputs = 2

    def get_new_individual(self, parents: list[Atoms]) -> Atoms | None:
        if len(parents) < 2:
            return None
        return self._cross(parents[0], parents[1])

    def _cross(self, a1: Atoms, a2: Atoms, maxiter: int = 1000) -> Atoms | None:
        n = len(a1)
        n_slab = n - self.n_top

        for _ in range(maxiter):
            top1 = a1[n_slab:]
            top2 = a2[n_slab:]

            if len(top1) == 0 or len(top2) == 0:
                return None

            # Random cutting direction (mostly in-plane for slabs)
            theta = self.rng.uniform(0, 2 * np.pi)
            normal = np.array([np.cos(theta), np.sin(theta), 0.0])

            com = 0.5 * (top1.get_center_of_mass() + top2.get_center_of_mass())

            # Select from each parent
            pos1 = top1.get_positions()
            pos2 = top2.get_positions()
            above1 = np.dot(pos1 - com, normal) > 0
            below2 = np.dot(pos2 - com, normal) <= 0

            offspring = self.slab.copy()
            for idx in np.where(above1)[0]:
                offspring.append(top1[idx])
            for idx in np.where(below2)[0]:
                offspring.append(top2[idx])

            # Check composition constraints
            if self.allowed_compositions is not None:
                active = offspring[len(self.slab) :]
                comp = {}
                for s in active.get_chemical_symbols():
                    comp[s] = comp.get(s, 0) + 1
                if comp not in self.allowed_compositions:
                    continue

            if _check_min_distances(offspring, self.blmin):
                offspring = self.initialize_individual(a1, offspring)
                offspring.info["data"]["parents"] = [
                    a1.info.get("confid", -1),
                    a2.info.get("confid", -1),
                ]
                return self.finalize_individual(offspring)

        return None


class RandomCompositionMutation(OffspringCreator):
    """Mutate to a random allowed composition.

    Changes the active region to a randomly chosen composition
    from the allowed set, keeping positions as similar as possible.

    Parameters
    ----------
    slab : Atoms
        Fixed slab.
    n_top : int
        Number of active atoms.
    blmin : dict
        Minimum interatomic distances.
    allowed_compositions : list of dict
        Allowed compositions.
    element_pool : list of str
        Allowed elements.
    rng : Generator, optional
    """

    def __init__(
        self,
        slab: Atoms,
        n_top: int,
        blmin: dict,
        allowed_compositions: list[dict[str, int]],
        element_pool: list[str],
        rng: np.random.Generator | None = None,
    ):
        super().__init__(rng=rng)
        self.slab = slab
        self.n_top = n_top
        self.blmin = blmin
        self.allowed_compositions = allowed_compositions
        self.element_pool = element_pool
        self.descriptor = "RandomCompositionMutation"

    def get_new_individual(self, parents: list[Atoms]) -> Atoms | None:
        parent = parents[0]
        n = len(parent)
        n_slab = n - self.n_top

        # Pick a random composition
        new_comp = self.allowed_compositions[
            self.rng.integers(len(self.allowed_compositions))
        ]

        # Build new active region with the new composition
        indi = self.slab.copy()
        active_pos = parent.positions[n_slab:]

        new_symbols = []
        for elem, count in new_comp.items():
            new_symbols.extend([elem] * count)

        n_new = len(new_symbols)
        if n_new == 0:
            return None

        # Reuse positions if possible, generate random ones otherwise
        if n_new <= len(active_pos):
            positions = active_pos[:n_new]
        else:
            positions = np.vstack(
                [
                    active_pos,
                    active_pos[
                        self.rng.integers(len(active_pos), size=n_new - len(active_pos))
                    ],
                ]
            )
            # Add random displacement to duplicated positions
            positions[len(active_pos) :] += self.rng.normal(
                0, 0.5, (n_new - len(active_pos), 3)
            )

        self.rng.shuffle(new_symbols)
        new_atoms = Atoms(new_symbols, positions=positions)
        new_atoms.set_tags([1] * n_new)
        indi.extend(new_atoms)

        if _check_min_distances(indi, self.blmin):
            indi = self.initialize_individual(parent, indi)
            indi.info["data"]["parents"] = [parent.info.get("confid", -1)]
            return self.finalize_individual(indi)

        return None


class RandomElementMutation(OffspringCreator):
    """Swap a single atom's element with another from the element pool.

    Parameters
    ----------
    n_top : int
        Number of active atoms.
    element_pool : list of str
        Allowed elements.
    rng : Generator, optional
    """

    def __init__(
        self,
        n_top: int,
        element_pool: list[str],
        rng: np.random.Generator | None = None,
    ):
        super().__init__(rng=rng)
        self.n_top = n_top
        self.element_pool = element_pool
        self.descriptor = "RandomElementMutation"

    def get_new_individual(self, parents: list[Atoms]) -> Atoms | None:
        parent = parents[0]
        indi = parent.copy()
        n = len(indi)
        n_slab = n - self.n_top

        if self.n_top < 1:
            return None

        # Pick a random active atom
        idx = self.rng.integers(n_slab, n)
        current_sym = indi.symbols[idx]

        # Pick a different element
        candidates = [e for e in self.element_pool if e != current_sym]
        if not candidates:
            return None

        new_sym = candidates[self.rng.integers(len(candidates))]
        indi.symbols[idx] = new_sym

        indi = self.initialize_individual(parent, indi)
        indi.info["data"]["parents"] = [parent.info.get("confid", -1)]
        return self.finalize_individual(indi)
