"""Monte Carlo move classes for grand canonical ensemble.

Implements Insert, Delete, Swap, and Displace moves. Each move
proposes a new configuration and returns the log proposal ratio
needed for detailed-balance correction in the Metropolis step.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
from ase import Atoms

from gs_gmlip.interface.core import Interface
from gs_gmlip.search.region import Region

logger = logging.getLogger(__name__)


class MCMove(ABC):
    """Abstract base class for Monte Carlo moves."""

    @abstractmethod
    def propose(
        self, interface: Interface, rng: np.random.Generator
    ) -> tuple[Interface | None, float, dict]:
        """Propose a new configuration.

        Parameters
        ----------
        interface : Interface
            Current configuration.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        new_interface : Interface or None
            Proposed configuration, or None if the move is invalid.
        log_proposal_ratio : float
            Log of the proposal probability ratio (0 for symmetric proposals).
        info : dict
            Move metadata (type, species, chemical_potential, …).
        """


# ---------------------------------------------------------------------------
# Insert
# ---------------------------------------------------------------------------


class InsertMove(MCMove):
    """Insert a molecular block at a random position in a Region.

    Parameters
    ----------
    blocks : list[Atoms]
        Possible species to insert (each is an Atoms template).
    block_names : list[str]
        Names corresponding to *blocks*.
    chemical_potentials : dict[str, float]
        Species name → chemical potential (eV).
    region : Region
        Region for random placement.
    """

    def __init__(
        self,
        blocks: list[Atoms],
        block_names: list[str],
        chemical_potentials: dict[str, float],
        region: Region,
    ):
        self.blocks = blocks
        self.block_names = block_names
        self.chemical_potentials = chemical_potentials
        self.region = region

    def propose(
        self, interface: Interface, rng: np.random.Generator
    ) -> tuple[Interface | None, float, dict]:
        # Pick a random species
        idx = rng.integers(len(self.blocks))
        block = self.blocks[idx]
        name = self.block_names[idx]
        mu = self.chemical_potentials.get(name, 0.0)

        # Random position in region
        pos = self.region.random_position(rng)

        # Build molecule copy
        mol = block.copy()
        # Random rotation for multi-atom blocks
        if len(mol) > 1:
            angle = rng.uniform(0, 360)
            axis = rng.normal(0, 1, 3)
            axis /= np.linalg.norm(axis)
            mol.rotate(angle, axis)
        mol.translate(pos - mol.get_center_of_mass())

        # Create new interface via add_adsorbate
        new_iface = interface.copy()
        new_iface.add_adsorbate(mol)

        # Log proposal ratio: volume / (n_species + 1)
        n_ads = sum(len(g) for g in new_iface.adsList) if hasattr(new_iface, 'adsList') else 1
        V = 1.0  # approximate; exact ratio needs region volume
        log_ratio = np.log(max(V, 1e-12)) - np.log(max(n_ads, 1))

        return new_iface, log_ratio, {
            "move": "insert",
            "species": name,
            "chemical_potential": mu,
        }


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class DeleteMove(MCMove):
    """Remove a random adsorbate group from the interface.

    Uses ``Interface.adsList`` to identify removable groups.

    Parameters
    ----------
    chemical_potentials : dict[str, float]
        Species name → chemical potential (eV), used for acceptance.
    """

    def __init__(self, chemical_potentials: dict[str, float] | None = None):
        self.chemical_potentials = chemical_potentials or {}

    def propose(
        self, interface: Interface, rng: np.random.Generator
    ) -> tuple[Interface | None, float, dict]:
        # Collect all removable groups: adsorbates + clusters
        groups = interface.adsList + interface.clusterList
        if not groups:
            return None, 0.0, {"reason": "no_adsorbates", "move": "delete"}

        # Pick a random group
        group = groups[rng.integers(len(groups))]

        # Try to identify species name from atom symbols for chemical potential lookup
        symbols = sorted(interface.interface.symbols[i] for i in group)
        species_name = "".join(symbols)
        mu = 0.0
        for name, pot in self.chemical_potentials.items():
            if sorted(name) == symbols:
                mu = pot
                species_name = name
                break

        # Create new interface via remove_group
        new_iface = interface.copy()
        new_iface.remove_group(group)

        # Log proposal ratio: n_before / 1
        n_before = len(groups)
        log_ratio = np.log(max(n_before, 1))

        return new_iface, log_ratio, {
            "move": "delete",
            "species": species_name,
            "chemical_potential": mu,
        }


# ---------------------------------------------------------------------------
# Displace
# ---------------------------------------------------------------------------


class DisplaceMove(MCMove):
    """Random displacement of a single adsorbate/cluster group.

    Parameters
    ----------
    max_displacement : float
        Maximum displacement magnitude (Å) — uniform in [-max, max].
    """

    def __init__(self, max_displacement: float = 1.0):
        self.max_displacement = max_displacement

    def propose(
        self, interface: Interface, rng: np.random.Generator
    ) -> tuple[Interface | None, float, dict]:
        groups = interface.adsList + interface.clusterList
        if not groups:
            return None, 0.0, {"reason": "no_adsorbates", "move": "displace"}

        # Pick a random group
        group = groups[rng.integers(len(groups))]

        # Uniform random displacement
        displacement = rng.uniform(-self.max_displacement, self.max_displacement, 3)

        new_iface = interface.copy()
        for idx in group:
            new_iface.interface.positions[idx] += displacement
        new_iface._reset_cache()

        return new_iface, 0.0, {"move": "displace", "group_size": len(group)}


# ---------------------------------------------------------------------------
# Swap
# ---------------------------------------------------------------------------


class SwapMove(MCMove):
    """Swap the element of a single-atom adsorbate.

    Only single-atom groups in ``adsList`` are eligible (not clusters).

    Parameters
    ----------
    allowed_elements : list[str]
        Element symbols that can be swapped to.
    chemical_potentials : dict[str, float]
        Species name → chemical potential (eV).
    """

    def __init__(
        self,
        allowed_elements: list[str],
        chemical_potentials: dict[str, float] | None = None,
    ):
        self.allowed_elements = allowed_elements
        self.chemical_potentials = chemical_potentials or {}

    def propose(
        self, interface: Interface, rng: np.random.Generator
    ) -> tuple[Interface | None, float, dict]:
        # Only consider single-atom adsorbate groups
        single_atom_groups = [g for g in interface.adsList if len(g) == 1]
        if not single_atom_groups:
            return None, 0.0, {"reason": "no_single_atom_adsorbates", "move": "swap"}

        # Pick a random single-atom adsorbate
        group = single_atom_groups[rng.integers(len(single_atom_groups))]
        idx = group[0]
        current_symbol = interface.interface.symbols[idx]

        # Pick a different element
        candidates = [e for e in self.allowed_elements if e != current_symbol]
        if not candidates:
            return None, 0.0, {"reason": "no_alternative", "move": "swap"}

        new_symbol = candidates[rng.integers(len(candidates))]

        new_iface = interface.copy()
        new_iface.interface.symbols[idx] = new_symbol
        new_iface._reset_cache()

        mu_old = self.chemical_potentials.get(current_symbol, 0.0)
        mu_new = self.chemical_potentials.get(new_symbol, 0.0)

        return new_iface, 0.0, {
            "move": "swap",
            "old_species": current_symbol,
            "new_species": new_symbol,
            "chemical_potential": mu_new - mu_old,
        }
