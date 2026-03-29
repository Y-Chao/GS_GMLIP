"""Composition constraints and molecular building blocks.

Defines what chemical species can be added/removed during search,
allowed stoichiometry ranges, and molecular building blocks (H2O, CO2, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ase import Atoms
from ase.build import molecule as ase_molecule


@dataclass
class MolecularBlock:
    """A molecular building block for insertion during search.

    Parameters
    ----------
    name : str
        Human-readable name (e.g., "H2O", "CO2").
    atoms : Atoms
        ASE Atoms object representing the molecule.
    chemical_potential : float
        Chemical potential mu in eV for grand canonical methods.
    """

    name: str
    atoms: Atoms
    chemical_potential: float = 0.0

    @classmethod
    def from_ase(cls, name: str, chemical_potential: float = 0.0) -> MolecularBlock:
        """Create from ASE's built-in molecule database."""
        atoms = ase_molecule(name)
        return cls(name=name, atoms=atoms, chemical_potential=chemical_potential)

    @classmethod
    def single_atom(
        cls, symbol: str, chemical_potential: float = 0.0
    ) -> MolecularBlock:
        """Create a single-atom block."""
        atoms = Atoms(symbol)
        return cls(name=symbol, atoms=atoms, chemical_potential=chemical_potential)

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def symbols(self) -> list[str]:
        return self.atoms.get_chemical_symbols()


@dataclass
class CompositionConstraint:
    """Constraints on allowed compositions in the active region.

    Parameters
    ----------
    blocks : list of MolecularBlock
        Allowed molecular building blocks.
    min_count : dict, optional
        Minimum count per block name, e.g. {"H2O": 0, "CO2": 0}.
    max_count : dict, optional
        Maximum count per block name, e.g. {"H2O": 10, "CO2": 5}.
    total_min : int
        Minimum total number of adsorbate units.
    total_max : int
        Maximum total number of adsorbate units.
    element_pool : list of str, optional
        Allowed elements in the active region. If None, inferred from blocks.
    """

    blocks: list[MolecularBlock] = field(default_factory=list)
    min_count: dict[str, int] = field(default_factory=dict)
    max_count: dict[str, int] = field(default_factory=dict)
    total_min: int = 0
    total_max: int = 50
    element_pool: list[str] | None = None

    def get_element_pool(self) -> list[str]:
        """Return list of all allowed elements."""
        if self.element_pool is not None:
            return self.element_pool
        elements = set()
        for block in self.blocks:
            elements.update(block.symbols)
        return sorted(elements)

    def get_random_block(
        self, rng: np.random.Generator | None = None
    ) -> MolecularBlock:
        """Select a random molecular block."""
        if rng is None:
            rng = np.random.default_rng()
        return self.blocks[rng.integers(len(self.blocks))]

    def check_composition(self, block_counts: dict[str, int]) -> bool:
        """Check if a composition satisfies constraints.

        Parameters
        ----------
        block_counts : dict
            {block_name: count} for current composition.
        """
        total = sum(block_counts.values())
        if total < self.total_min or total > self.total_max:
            return False

        for name, count in block_counts.items():
            if name in self.min_count and count < self.min_count[name]:
                return False
            if name in self.max_count and count > self.max_count[name]:
                return False

        return True

    def can_add(self, block_name: str, current_counts: dict[str, int]) -> bool:
        """Check if adding one more of block_name is allowed."""
        new_counts = current_counts.copy()
        new_counts[block_name] = new_counts.get(block_name, 0) + 1
        return self.check_composition(new_counts)

    def can_remove(self, block_name: str, current_counts: dict[str, int]) -> bool:
        """Check if removing one of block_name is allowed."""
        if current_counts.get(block_name, 0) <= 0:
            return False
        new_counts = current_counts.copy()
        new_counts[block_name] -= 1
        return self.check_composition(new_counts)


# --- Common block definitions for catalyst systems ---


def water_block(mu: float = 0.0) -> MolecularBlock:
    """H2O molecular block."""
    return MolecularBlock.from_ase("H2O", chemical_potential=mu)


def co2_block(mu: float = 0.0) -> MolecularBlock:
    """CO2 molecular block."""
    return MolecularBlock.from_ase("CO2", chemical_potential=mu)


def co_block(mu: float = 0.0) -> MolecularBlock:
    """CO molecular block."""
    return MolecularBlock.from_ase("CO", chemical_potential=mu)


def oh_block(mu: float = 0.0) -> MolecularBlock:
    """OH radical block."""
    atoms = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.97]])
    return MolecularBlock(name="OH", atoms=atoms, chemical_potential=mu)


def hydrogen_block(mu: float = 0.0) -> MolecularBlock:
    """Single H atom block."""
    return MolecularBlock.single_atom("H", chemical_potential=mu)


def oxygen_block(mu: float = 0.0) -> MolecularBlock:
    """Single O atom block."""
    return MolecularBlock.single_atom("O", chemical_potential=mu)


def sulfur_block(mu: float = 0.0) -> MolecularBlock:
    """Single S atom block."""
    return MolecularBlock.single_atom("S", chemical_potential=mu)


def so2_block(mu: float = 0.0) -> MolecularBlock:
    """SO2 molecular block."""
    return MolecularBlock.from_ase("SO2", chemical_potential=mu)


def formic_acid_block(mu: float = 0.0) -> MolecularBlock:
    """HCOOH molecular block."""
    atoms = Atoms(
        "CHOOH",
        positions=[
            [0.000, 0.000, 0.000],  # C
            [0.000, 1.097, 0.000],  # H
            [-1.038, -0.587, 0.000],  # O (carbonyl)
            [1.138, -0.700, 0.000],  # O (hydroxyl)
            [1.889, -0.082, 0.000],  # H (hydroxyl)
        ],
    )
    return MolecularBlock(name="HCOOH", atoms=atoms, chemical_potential=mu)
