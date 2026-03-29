"""Monte Carlo move classes for GCMC.

Implements Insert, Delete, Swap, and Displace moves for the
grand canonical ensemble. Each move proposes a new configuration
and returns the log proposal ratio for detailed balance.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
from ase import Atoms

from gs_gmlip.structure.composition import CompositionConstraint, MolecularBlock
from gs_gmlip.structure.region import Region

logger = logging.getLogger(__name__)


class MCMove(ABC):
    """Abstract base class for Monte Carlo moves."""

    @abstractmethod
    def propose(
        self, atoms: Atoms, rng: np.random.Generator
    ) -> tuple[Atoms | None, float, dict]:
        """Propose a new configuration.

        Parameters
        ----------
        atoms : Atoms
            Current configuration.
        rng : Generator
            Random number generator.

        Returns
        -------
        new_atoms : Atoms or None
            Proposed configuration, or None if move is invalid.
        log_proposal_ratio : float
            Log of the proposal ratio for detailed balance.
            For symmetric proposals, this is 0.
        info : dict
            Additional information about the move.
        """


class InsertMove(MCMove):
    """Insert a random molecular block into the active region.

    Parameters
    ----------
    blocks : list of MolecularBlock
        Possible species to insert.
    region : Region
        Region where insertion is allowed.
    constraints : CompositionConstraint, optional
        Composition constraints.
    """

    def __init__(
        self,
        blocks: list[MolecularBlock],
        region: Region,
        constraints: CompositionConstraint | None = None,
    ):
        self.blocks = blocks
        self.region = region
        self.constraints = constraints

    def propose(
        self, atoms: Atoms, rng: np.random.Generator
    ) -> tuple[Atoms | None, float, dict]:
        # Select a random block
        block = self.blocks[rng.integers(len(self.blocks))]

        # Check constraints
        if self.constraints is not None:
            current_counts = self._count_blocks(atoms)
            if not self.constraints.can_add(block.name, current_counts):
                return None, 0.0, {"reason": "composition_constraint"}

        # Generate random position
        pos = self.region.random_position(rng)

        # Place the block
        new_atoms = atoms.copy()
        block_atoms = block.atoms.copy()
        block_atoms.translate(pos - block_atoms.get_center_of_mass())

        # Random rotation for multi-atom blocks
        if len(block_atoms) > 1:
            angle = rng.uniform(0, 360)
            axis = rng.normal(0, 1, 3)
            axis /= np.linalg.norm(axis)
            block_atoms.rotate(angle, axis, center=pos)

        # Tag the new atoms
        max_tag = max(new_atoms.get_tags(), default=0)
        block_atoms.set_tags([max_tag + 1] * len(block_atoms))
        new_atoms.extend(block_atoms)

        # Log proposal ratio for GCMC: includes volume and N factors
        n_species = self._count_species(atoms, block.name)
        V = self.region.random_positions(1, rng).size  # Approximate volume
        log_ratio = np.log(V) - np.log(n_species + 1)

        return (
            new_atoms,
            log_ratio,
            {
                "move": "insert",
                "species": block.name,
                "chemical_potential": block.chemical_potential,
            },
        )

    def _count_blocks(self, atoms: Atoms) -> dict[str, int]:
        """Count blocks in the active region (rough estimate by tags)."""
        tags = atoms.get_tags()
        unique_tags = set(t for t in tags if t > 0)
        counts: dict[str, int] = {}
        for block in self.blocks:
            counts[block.name] = 0
        # Simplified: count unique adsorbate tags
        counts["total"] = len(unique_tags)
        return counts

    def _count_species(self, atoms: Atoms, species_name: str) -> int:
        """Count occurrences of a species."""
        tags = atoms.get_tags()
        return len(set(t for t in tags if t > 0))


class DeleteMove(MCMove):
    """Remove a random adsorbate (molecular unit) from the surface.

    Parameters
    ----------
    blocks : list of MolecularBlock
        Species that can be removed.
    """

    def __init__(self, blocks: list[MolecularBlock]):
        self.blocks = blocks

    def propose(
        self, atoms: Atoms, rng: np.random.Generator
    ) -> tuple[Atoms | None, float, dict]:
        # Find adsorbate tags
        tags = atoms.get_tags()
        adsorbate_tags = sorted(set(t for t in tags if t > 0))

        if not adsorbate_tags:
            return None, 0.0, {"reason": "no_adsorbates"}

        # Select random adsorbate to remove
        tag_to_remove = adsorbate_tags[rng.integers(len(adsorbate_tags))]

        # Identify the species being removed
        indices_to_remove = [i for i, t in enumerate(tags) if t == tag_to_remove]
        removed_symbols = [atoms.symbols[i] for i in indices_to_remove]

        # Find matching block for chemical potential
        mu = 0.0
        species_name = "unknown"
        for block in self.blocks:
            if sorted(block.symbols) == sorted(removed_symbols):
                mu = block.chemical_potential
                species_name = block.name
                break

        # Create new atoms without the removed adsorbate
        keep_indices = [i for i in range(len(atoms)) if tags[i] != tag_to_remove]
        new_atoms = atoms[keep_indices]

        n_before = len(adsorbate_tags)
        log_ratio = np.log(n_before) - np.log(1.0)  # Simplified

        return (
            new_atoms,
            log_ratio,
            {
                "move": "delete",
                "species": species_name,
                "chemical_potential": mu,
            },
        )


class SwapMove(MCMove):
    """Swap the species of a random adsorbate.

    Parameters
    ----------
    blocks : list of MolecularBlock
        Allowed species (must be single atoms for v1.0 swap).
    """

    def __init__(self, blocks: list[MolecularBlock]):
        self.blocks = [b for b in blocks if b.n_atoms == 1]

    def propose(
        self, atoms: Atoms, rng: np.random.Generator
    ) -> tuple[Atoms | None, float, dict]:
        if len(self.blocks) < 2:
            return None, 0.0, {"reason": "need_2_single_atom_blocks"}

        tags = atoms.get_tags()
        active_indices = [i for i, t in enumerate(tags) if t > 0]

        if not active_indices:
            return None, 0.0, {"reason": "no_active_atoms"}

        # Select random active atom
        idx = active_indices[rng.integers(len(active_indices))]
        current_sym = atoms.symbols[idx]

        # Select different species
        candidates = [b for b in self.blocks if b.name != current_sym]
        if not candidates:
            return None, 0.0, {"reason": "no_alternative"}

        new_block = candidates[rng.integers(len(candidates))]

        new_atoms = atoms.copy()
        new_atoms.symbols[idx] = new_block.name

        return (
            new_atoms,
            0.0,
            {
                "move": "swap",
                "old_species": current_sym,
                "new_species": new_block.name,
            },
        )


class DisplaceMove(MCMove):
    """Random displacement of adsorbate atoms.

    Parameters
    ----------
    max_displacement : float
        Maximum displacement in Angstrom.
    """

    def __init__(self, max_displacement: float = 1.0):
        self.max_displacement = max_displacement

    def propose(
        self, atoms: Atoms, rng: np.random.Generator
    ) -> tuple[Atoms | None, float, dict]:
        tags = atoms.get_tags()
        active_indices = [i for i, t in enumerate(tags) if t > 0]

        if not active_indices:
            return None, 0.0, {"reason": "no_active_atoms"}

        new_atoms = atoms.copy()

        # Select a random adsorbate tag to displace
        adsorbate_tags = sorted(set(t for t in tags if t > 0))
        tag = adsorbate_tags[rng.integers(len(adsorbate_tags))]

        # Displace all atoms with this tag
        indices = [i for i, t in enumerate(tags) if t == tag]
        displacement = rng.uniform(-self.max_displacement, self.max_displacement, 3)

        for idx in indices:
            new_atoms.positions[idx] += displacement

        return new_atoms, 0.0, {"move": "displace", "tag": int(tag)}
