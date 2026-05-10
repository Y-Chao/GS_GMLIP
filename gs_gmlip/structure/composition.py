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
    bonds : list of tuple[int, int], optional
        Intramolecular bonds as (atom_i, atom_j) index pairs (0-based
        within this molecule). When provided, Hookean constraints are
        built from these bonds directly instead of auto-detection.
    n_electrons : int
        Number of electrons transferred per species in CHE model.
        Used to shift chemical potential: mu_eff = mu + n_e * U.
    spring_constant : float
        Hookean spring constant k in eV/Å² for bond constraints.
    threshold_scale : float
        Multiply the current bond length by this factor to set the
        Hookean threshold distance rt.
    """

    name: str
    atoms: Atoms
    chemical_potential: float = 0.0
    bonds: list[tuple[int, int]] | None = None
    n_electrons: int = 0
    spring_constant: float = 15.0
    threshold_scale: float = 1.3

    @classmethod
    def from_ase(cls, name: str, chemical_potential: float = 0.0) -> MolecularBlock:
        """Create from ASE's built-in molecule database."""
        atoms = ase_molecule(name)
        return cls(name=name, atoms=atoms, chemical_potential=chemical_potential)

    @classmethod
    def single_atom(
        cls,
        symbol: str,
        chemical_potential: float = 0.0,
        n_electrons: int = 0,
    ) -> MolecularBlock:
        """Create a single-atom block."""
        atoms = Atoms(symbol)
        return cls(
            name=symbol,
            atoms=atoms,
            chemical_potential=chemical_potential,
            n_electrons=n_electrons,
        )

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
    """H2O molecular block.

    CHE: H2O -> *H2O (0 electrons transferred if adsorbed molecularly).
    To model dissociation: H2O -> *OH + H+ + e- (1e), or *O + 2H+ + 2e- (2e).
    Convention: n_electrons=0 for molecular water; set manually for dissociated.
    """
    atoms = ase_molecule("H2O")
    # ASE H2O: index 0=O, 1=H, 2=H
    bonds = [(0, 1), (0, 2)]
    return MolecularBlock(
        name="H2O",
        atoms=atoms,
        chemical_potential=mu,
        bonds=bonds,
        n_electrons=0,
    )


def co2_block(mu: float = 0.0) -> MolecularBlock:
    """CO2 molecular block."""
    atoms = ase_molecule("CO2")
    # ASE CO2: index 0=C, 1=O, 2=O
    bonds = [(0, 1), (0, 2)]
    return MolecularBlock(
        name="CO2",
        atoms=atoms,
        chemical_potential=mu,
        bonds=bonds,
        n_electrons=0,
    )


def co_block(mu: float = 0.0) -> MolecularBlock:
    """CO molecular block."""
    atoms = ase_molecule("CO")
    # ASE CO: index 0=C, 1=O
    bonds = [(0, 1)]
    return MolecularBlock(
        name="CO",
        atoms=atoms,
        chemical_potential=mu,
        bonds=bonds,
        n_electrons=2,
    )


def oh_block(mu: float = 0.0) -> MolecularBlock:
    """OH radical block."""
    atoms = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.97]])
    bonds = [(0, 1)]
    return MolecularBlock(
        name="OH",
        atoms=atoms,
        chemical_potential=mu,
        bonds=bonds,
        n_electrons=1,
    )


def hydrogen_block(mu: float = 0.0) -> MolecularBlock:
    """Single H atom block.  CHE: H+ + e- -> *H (1 electron)."""
    return MolecularBlock.single_atom("H", chemical_potential=mu, n_electrons=1)


def oxygen_block(mu: float = 0.0) -> MolecularBlock:
    """Single O atom block.  CHE: H2O -> *O + 2H+ + 2e- (2 electrons)."""
    return MolecularBlock.single_atom("O", chemical_potential=mu, n_electrons=2)


def sulfur_block(mu: float = 0.0) -> MolecularBlock:
    """Single S atom block."""
    return MolecularBlock.single_atom("S", chemical_potential=mu, n_electrons=2)


def so2_block(mu: float = 0.0) -> MolecularBlock:
    """SO2 molecular block."""
    atoms = ase_molecule("SO2")
    # ASE SO2: index 0=S, 1=O, 2=O
    bonds = [(0, 1), (0, 2)]
    return MolecularBlock(
        name="SO2",
        atoms=atoms,
        chemical_potential=mu,
        bonds=bonds,
        n_electrons=0,
    )


def cooh_block(mu: float = 0.0) -> MolecularBlock:
    """*COOH carboxyl intermediate block.

    CO2 + H+ + e- -> *COOH  (1 electron in CHE).
    Geometry: C bonded to O=C and O-H, angles ~120 deg.
    """
    atoms = Atoms(
        "COOH",
        positions=[
            [0.000, 0.000, 0.000],  # C
            [-1.100, 0.520, 0.000],  # O (carbonyl, C=O)
            [1.080, 0.620, 0.000],  # O (hydroxyl, C-OH)
            [1.780, 0.000, 0.000],  # H (on hydroxyl O)
        ],
    )
    # Bonds: C-O(carbonyl), C-O(hydroxyl), O(hydroxyl)-H
    bonds = [(0, 1), (0, 2), (2, 3)]
    return MolecularBlock(
        name="COOH",
        atoms=atoms,
        chemical_potential=mu,
        bonds=bonds,
        n_electrons=1,
    )


def hcoo_block(mu: float = 0.0) -> MolecularBlock:
    """*HCOO formate intermediate block.

    CO2 + H+ + e- -> *HCOO  (1 electron in CHE).
    Geometry: H-C with two equivalent C-O bonds (bidentate formate).
    """
    atoms = Atoms(
        "HCOO",
        positions=[
            [0.000, 1.097, 0.000],  # H
            [0.000, 0.000, 0.000],  # C
            [-1.038, -0.587, 0.000],  # O
            [1.038, -0.587, 0.000],  # O
        ],
    )
    # Bonds: H-C, C-O, C-O
    bonds = [(0, 1), (1, 2), (1, 3)]
    return MolecularBlock(
        name="HCOO",
        atoms=atoms,
        chemical_potential=mu,
        bonds=bonds,
        n_electrons=1,
    )


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
    bonds = [(0, 1), (0, 2), (0, 3), (3, 4)]
    return MolecularBlock(
        name="HCOOH",
        atoms=atoms,
        chemical_potential=mu,
        bonds=bonds,
        n_electrons=0,
    )


# ---------------------------------------------------------------------------
# Block registry: central place to look up block factories by name.
# Use ``register_block`` to add custom species so that they become available
# to every runner's ``from_config`` and to ``build_blocks`` in examples.
# ---------------------------------------------------------------------------

_BLOCK_REGISTRY: dict[str, callable] = {}


def register_block(name: str, factory: callable) -> None:
    """Register a block factory function under *name*.

    Parameters
    ----------
    name : str
        Short name used in config YAML (e.g. ``"CHO"``).
    factory : callable(mu: float) -> MolecularBlock
        Factory that accepts a ``mu`` keyword and returns a MolecularBlock.

    Example
    -------
    >>> from ase import Atoms
    >>> def cho_block(mu=0.0):
    ...     atoms = Atoms("CHO", positions=[[0,0,0],[1.1,0,0],[0,1.2,0]])
    ...     return MolecularBlock(
    ...         name="CHO", atoms=atoms, chemical_potential=mu,
    ...         bonds=[(0,1),(0,2)], n_electrons=1,
    ...     )
    >>> register_block("CHO", cho_block)
    """
    _BLOCK_REGISTRY[name] = factory


def get_block_registry() -> dict[str, callable]:
    """Return a *copy* of the current block registry."""
    return dict(_BLOCK_REGISTRY)


def get_block(name: str, mu: float = 0.0) -> MolecularBlock:
    """Look up *name* in the registry and return a MolecularBlock.

    Raises ``ValueError`` if the name is not registered.
    """
    if name not in _BLOCK_REGISTRY:
        raise ValueError(
            f"Unknown block: {name!r}. "
            f"Registered blocks: {sorted(_BLOCK_REGISTRY)}. "
            f"Use register_block() to add custom species."
        )
    return _BLOCK_REGISTRY[name](mu=mu)


# Register all built-in blocks
for _name, _factory in [
    ("H2O", water_block),
    ("CO2", co2_block),
    ("CO", co_block),
    ("OH", oh_block),
    ("H", hydrogen_block),
    ("O", oxygen_block),
    ("S", sulfur_block),
    ("SO2", so2_block),
    ("HCOOH", formic_acid_block),
    ("COOH", cooh_block),
    ("HCOO", hcoo_block),
]:
    register_block(_name, _factory)
