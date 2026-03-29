"""Initial structure generator for GA.

Generates random initial candidate structures for the genetic algorithm,
placing atoms/molecules in the active region while respecting distance
constraints. Refactored from ase_ga_official/startgenerator.py.
"""

from __future__ import annotations

import logging

import numpy as np
from ase import Atoms

from gs_gmlip.search.ga.operators import _check_min_distances
from gs_gmlip.structure.composition import CompositionConstraint, MolecularBlock
from gs_gmlip.structure.region import Region

logger = logging.getLogger(__name__)


class StartGenerator:
    """Generate random initial candidate structures.

    Parameters
    ----------
    slab : Atoms
        The fixed slab structure.
    blocks : list of MolecularBlock
        Building blocks to place.
    counts : list of int
        Number of each block to place.
    blmin : dict
        Minimum interatomic distances {(Z1, Z2): distance}.
    region : Region
        Spatial region where blocks can be placed.
    rng : Generator, optional
    """

    def __init__(
        self,
        slab: Atoms,
        blocks: list[MolecularBlock],
        counts: list[int],
        blmin: dict,
        region: Region,
        rng: np.random.Generator | None = None,
    ):
        self.slab = slab
        self.blocks = blocks
        self.counts = counts
        self.blmin = blmin
        self.region = region
        self.rng = rng or np.random.default_rng()

    def get_new_candidate(self, maxiter: int = 1000) -> Atoms | None:
        """Generate a new random candidate structure.

        Returns
        -------
        candidate : Atoms or None
            New candidate, or None if generation failed after maxiter attempts.
        """
        for _ in range(maxiter):
            candidate = self.slab.copy()
            tag = max(candidate.get_tags(), default=0) + 1
            success = True

            for block, count in zip(self.blocks, self.counts):
                for _ in range(count):
                    placed = self._place_block(candidate, block, tag)
                    if not placed:
                        success = False
                        break
                    tag += 1
                if not success:
                    break

            if success and _check_min_distances(candidate, self.blmin):
                candidate.info["key_value_pairs"] = {}
                candidate.info["data"] = {}
                return candidate

        logger.warning(f"Failed to generate candidate after {maxiter} attempts")
        return None

    def _place_block(
        self, candidate: Atoms, block: MolecularBlock, tag: int, maxiter: int = 500
    ) -> bool:
        """Try to place a single block in the candidate structure."""
        for _ in range(maxiter):
            pos = self.region.random_position(self.rng)

            # Create block atoms at the target position
            block_atoms = block.atoms.copy()
            # Center the block and move to target position
            block_atoms.translate(pos - block_atoms.get_center_of_mass())

            # Add random rotation for multi-atom blocks
            if len(block_atoms) > 1:
                angle = self.rng.uniform(0, 360)
                axis = self.rng.normal(0, 1, 3)
                axis /= np.linalg.norm(axis)
                block_atoms.rotate(angle, axis, center=pos)

            block_atoms.set_tags([tag] * len(block_atoms))

            # Test if placement is valid
            test = candidate.copy()
            test.extend(block_atoms)

            if _check_min_distances(test, self.blmin):
                candidate.extend(block_atoms)
                return True

        return False

    def get_candidates(self, n: int, maxiter_per: int = 1000) -> list[Atoms]:
        """Generate multiple candidates.

        Parameters
        ----------
        n : int
            Number of candidates to generate.
        maxiter_per : int
            Max iterations per candidate.
        """
        candidates = []
        for i in range(n):
            c = self.get_new_candidate(maxiter=maxiter_per)
            if c is not None:
                candidates.append(c)
            else:
                logger.warning(f"Could not generate candidate {i + 1}/{n}")
        return candidates
