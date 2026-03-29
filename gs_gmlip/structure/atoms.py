"""Extended ASE Atoms object with slab/interface support.

Provides SlabAtoms class that wraps ASE Atoms with additional functionality
for catalyst surface structure searching: slab tagging, active region
identification, composition tracking, and molecular identity preservation.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms


class SlabAtoms(Atoms):
    """Extended ASE Atoms with slab/interface system support.

    Tags convention:
        tag=0: Fixed slab atoms (substrate)
        tag=1+: Adsorbate atoms / active region (mutable during search)

    Molecules sharing the same tag value (>=1) are treated as a unit
    for molecular identity preservation during GA operations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_atoms(cls, atoms: Atoms) -> SlabAtoms:
        """Create SlabAtoms from an existing ASE Atoms object."""
        slab = cls(
            symbols=atoms.symbols,
            positions=atoms.positions.copy(),
            cell=atoms.cell.copy(),
            pbc=atoms.pbc.copy(),
        )
        if atoms.get_tags() is not None:
            slab.set_tags(atoms.get_tags())
        slab.info = atoms.info.copy()
        if atoms.constraints:
            slab.constraints = atoms.constraints
        return slab

    def get_slab_indices(self) -> np.ndarray:
        """Return indices of fixed slab atoms (tag=0)."""
        return np.where(self.get_tags() == 0)[0]

    def get_active_indices(self) -> np.ndarray:
        """Return indices of active/mutable atoms (tag > 0)."""
        return np.where(self.get_tags() > 0)[0]

    def get_active_atoms(self) -> Atoms:
        """Return Atoms object containing only active region atoms."""
        indices = self.get_active_indices()
        return self[indices]

    def get_slab_atoms(self) -> Atoms:
        """Return Atoms object containing only slab atoms."""
        indices = self.get_slab_indices()
        return self[indices]

    @property
    def n_active(self) -> int:
        """Number of active atoms."""
        return len(self.get_active_indices())

    @property
    def n_slab(self) -> int:
        """Number of slab atoms."""
        return len(self.get_slab_indices())

    def get_active_composition(self) -> dict[str, int]:
        """Return composition of the active region as {element: count}."""
        active = self.get_active_atoms()
        symbols = active.get_chemical_symbols()
        comp = {}
        for s in symbols:
            comp[s] = comp.get(s, 0) + 1
        return comp

    def get_molecule_groups(self) -> dict[int, list[int]]:
        """Return dict mapping tag -> list of atom indices for molecules.

        Only includes tags > 0 (active region).
        """
        tags = self.get_tags()
        groups = {}
        for i, t in enumerate(tags):
            if t > 0:
                groups.setdefault(t, []).append(i)
        return groups

    def set_slab_constraints(self):
        """Fix all slab atoms (tag=0) and free all active atoms."""
        slab_indices = self.get_slab_indices()
        if len(slab_indices) > 0:
            self.set_constraint(FixAtoms(indices=slab_indices))

    def tag_slab_by_height(self, z_threshold: float):
        """Tag atoms below z_threshold as slab (tag=0), above as active (tag=1).

        Parameters
        ----------
        z_threshold : float
            Height in Angstroms. Atoms with z < z_threshold get tag=0.
        """
        positions = self.get_positions()
        tags = np.where(positions[:, 2] < z_threshold, 0, 1)
        self.set_tags(tags)

    def add_adsorbate_atoms(
        self,
        adsorbate: Atoms,
        position: np.ndarray,
        tag: int | None = None,
    ) -> None:
        """Add adsorbate atoms at a given position with proper tagging.

        Parameters
        ----------
        adsorbate : Atoms
            Adsorbate molecule/atom to add.
        position : array-like, shape (3,)
            Position offset to apply to adsorbate.
        tag : int, optional
            Tag for the adsorbate. If None, uses max(existing_tags) + 1.
        """
        if tag is None:
            existing_tags = self.get_tags()
            tag = max(existing_tags) + 1 if len(existing_tags) > 0 else 1

        ads = adsorbate.copy()
        ads.translate(position - ads.get_center_of_mass())
        ads.set_tags([tag] * len(ads))
        self.extend(ads)

    def remove_adsorbate(self, tag: int) -> None:
        """Remove all atoms with the specified tag."""
        mask = self.get_tags() != tag
        indices = np.where(mask)[0]
        del self[np.where(~mask)[0]]
