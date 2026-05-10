"""Extended ASE Atoms object with slab/interface support.

Provides SlabAtoms class that wraps ASE Atoms with additional functionality
for catalyst surface structure searching: slab tagging, active region
identification, composition tracking, and molecular identity preservation.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms


def find_molecules(atoms: Atoms, scale: float = 1.2) -> list[int]:
    """Separate disconnected atoms into molecule groups via connected components.

    Parameters
    ----------
    atoms : Atoms
        Atoms object (typically the adsorbate portion).
    scale : float
        Scale factor for covalent radii to determine bonding.

    Returns
    -------
    labels : list[int]
        Molecule label for each atom (0-indexed). Atoms in the same
        connected component share the same label.
    """
    from ase.data import covalent_radii
    from ase.neighborlist import NeighborList

    n = len(atoms)
    if n == 0:
        return []

    cutoffs = [covalent_radii[atoms.numbers[i]] * scale for i in range(n)]
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    # BFS to find connected components
    labels = [-1] * n
    current_label = 0
    for start in range(n):
        if labels[start] != -1:
            continue
        # BFS from this atom
        queue = [start]
        labels[start] = current_label
        while queue:
            atom_i = queue.pop(0)
            neighbors, _ = nl.get_neighbors(atom_i)
            for j in neighbors:
                if labels[j] == -1:
                    labels[j] = current_label
                    queue.append(j)
        current_label += 1

    return labels


class SlabAtoms(Atoms):
    """Extended ASE Atoms with slab/interface system support.

    Tags convention:
        tag=0: Fixed substrate atoms (frozen during all operations)
        tag=1: Buffer region (relaxed during optimization, but NOT mutated/rattled)
        tag=2: Surface active layer (relaxed + can be mutated/rattled)
        tag=3+: Adsorbate atoms (each connected molecule gets a unique tag >=3,
                treated as a unit during GA operations)
    """

    def __init__(self, *args, **kwargs):
        # Pop custom kwargs before passing to ASE Atoms
        tag_list = kwargs.pop("tag_list", None)
        z_threshold = kwargs.pop("z_threshold", None)
        super().__init__(*args, **kwargs)
        if tag_list is not None:
            self.set_tags(tag_list)
        elif z_threshold is not None:
            self.tag_slab_by_height(z_threshold)
        elif not np.any(self.get_tags()):
            # Only set default tags if no tags exist yet
            self.set_tags([0] * len(self))

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
        """Return indices of fixed substrate atoms (tag=0)."""
        return np.where(self.get_tags() == 0)[0]

    def get_buffer_indices(self) -> np.ndarray:
        """Return indices of buffer region atoms (tag=1).

        These atoms are relaxed during optimization but NOT mutated/rattled.
        """
        return np.where(self.get_tags() == 1)[0]

    def get_surface_indices(self) -> np.ndarray:
        """Return indices of surface active atoms (tag=2).

        These atoms are relaxed AND can be mutated/rattled.
        """
        return np.where(self.get_tags() == 2)[0]

    def get_adsorbate_indices(self) -> np.ndarray:
        """Return indices of adsorbate atoms (tag>=3)."""
        return np.where(self.get_tags() >= 3)[0]

    def get_active_indices(self) -> np.ndarray:
        """Return indices of active/mutable atoms (tag >= 2, i.e. surface + adsorbates)."""
        return np.where(self.get_tags() >= 2)[0]

    def get_relaxable_indices(self) -> np.ndarray:
        """Return indices of all atoms that can be relaxed (tag > 0, i.e. buffer + surface + adsorbates)."""
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
        """Fix substrate atoms (tag=0). Buffer (tag=1), surface (tag=2),
        and adsorbate (tag>=3) atoms are free to relax."""
        slab_indices = self.get_slab_indices()
        if len(slab_indices) > 0:
            self.set_constraint(FixAtoms(indices=slab_indices))

    def tag_slab_by_height(self, z_threshold: float | tuple[float, float]) -> None:
        """Tag atoms into 3 regions by height.

        - tag=0 (substrate): z < z_threshold[0]  — fixed atoms
        - tag=1 (buffer):    z_threshold[0] <= z < z_threshold[1] — relaxed, not mutable
        - tag=2 (active):    z >= z_threshold[1] — relaxed + mutable

        Parameters
        ----------
        z_threshold : float or tuple[float, float]
            If a single float, atoms below it get tag=0, above get tag=1.
            If a tuple (z_low, z_high), atoms are split into 3 regions.
        """
        positions = self.get_positions()
        if isinstance(z_threshold, (int, float)):
            # Two-region split: substrate (tag=0) and buffer (tag=1)
            tags = np.where(positions[:, 2] < z_threshold, 0, 1)
        else:
            z_low, z_high = z_threshold
            tags = np.where(
                positions[:, 2] < z_low,
                0,
                np.where(positions[:, 2] < z_high, 1, 2),
            )
        self.set_tags(tags)

    def identify_adsorbate_from_template(
        self,
        template: Atoms,
        scale: float = 1.2,
    ) -> None:
        """Identify adsorbate atoms by matching to a template slab.
        Then assign tags by connectivity within the adsorbate region.
        Slab atoms keep their original tags; each disconnected adsorbate
        molecule gets a unique tag starting from 3.

        Parameters
        ----------
        template : Atoms
            Template slab structure (no adsorbates). Must have <= len(self) atoms.
            The first len(template) atoms of self are assumed to be the slab.
        scale : float
            Scale factor for covalent radii to determine bonding.
        """
        if len(template) == len(self):
            return  # no adsorbate atoms

        if len(template) > len(self):
            raise ValueError(
                "Template has more atoms than the current structure. "
                "Cannot identify adsorbate."
            )

        adsorbate_indices = list(range(len(template), len(self)))
        adsorbate_atoms = self[adsorbate_indices]
        mol_tags = find_molecules(adsorbate_atoms, scale=scale)

        # Assign molecule tags starting from 3
        tags = list(self.get_tags())
        for i, idx in enumerate(adsorbate_indices):
            tags[idx] = mol_tags[i] + 3
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
