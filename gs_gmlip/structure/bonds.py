"""Bond length database and bond matrix computation.

Provides common covalent bond lengths and utilities for computing
bond connectivity matrices from atomic structures.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.data import covalent_radii


class BondData:
    """Database of bond length information.

    Uses covalent radii from ASE with a scale factor to determine
    whether two atoms are bonded.
    """

    def __init__(
        self, scale: float = 1.2, custom_radii: dict[str, float] | None = None
    ):
        """
        Parameters
        ----------
        scale : float
            Scale factor applied to sum of covalent radii.
            Two atoms are bonded if distance < scale * (r1 + r2).
        custom_radii : dict, optional
            Override covalent radii for specific elements, e.g. {"Cu": 1.32}.
        """
        self.scale = scale
        self.custom_radii = custom_radii or {}

    def get_radius(self, atomic_number: int, symbol: str | None = None) -> float:
        if symbol and symbol in self.custom_radii:
            return self.custom_radii[symbol]
        return covalent_radii[atomic_number]

    def get_max_bond_length(
        self, z1: int, z2: int, sym1: str | None = None, sym2: str | None = None
    ) -> float:
        r1 = self.get_radius(z1, sym1)
        r2 = self.get_radius(z2, sym2)
        return self.scale * (r1 + r2)


def get_bond_matrix(
    atoms: Atoms,
    bond_data: BondData | None = None,
    mic: bool = True,
) -> np.ndarray:
    """Compute bond connectivity matrix for an atoms object.

    Parameters
    ----------
    atoms : Atoms
        The atomic structure.
    bond_data : BondData, optional
        Bond length parameters. Uses default if None.
    mic : bool
        Use minimum image convention for periodic systems.

    Returns
    -------
    bond_matrix : ndarray, shape (n, n), dtype bool
        bond_matrix[i, j] is True if atoms i and j are bonded.
    """
    if bond_data is None:
        bond_data = BondData()

    n = len(atoms)
    distances = atoms.get_all_distances(mic=mic)
    numbers = atoms.get_atomic_numbers()
    symbols = atoms.get_chemical_symbols()

    bond_mat = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            max_bl = bond_data.get_max_bond_length(
                numbers[i], numbers[j], symbols[i], symbols[j]
            )
            if distances[i, j] < max_bl:
                bond_mat[i, j] = True
                bond_mat[j, i] = True

    return bond_mat


def get_coordination_numbers(
    atoms: Atoms,
    bond_data: BondData | None = None,
    mic: bool = True,
) -> np.ndarray:
    """Compute coordination number for each atom.

    Returns
    -------
    coordination : ndarray, shape (n,), dtype int
    """
    bond_mat = get_bond_matrix(atoms, bond_data, mic)
    return bond_mat.sum(axis=1)


def generate_blmin(
    atom_numbers: list[int] | list[str],
    scale: float = 0.7,
    bond_data: BondData | None = None,
) -> dict[tuple[int, int], float]:
    """Generate minimum interatomic distance dict for GA operators.

    This is the `blmin` parameter used throughout the GA module.

    Parameters
    ----------
    atom_numbers : list of int or list of str
        Unique atomic numbers or chemical symbols in the system.
    scale : float
        Fraction of sum of covalent radii to use as minimum distance.
    bond_data : BondData, optional

    Returns
    -------
    blmin : dict
        {(Z1, Z2): min_distance} for all element pairs.
    """
    from ase.data import atomic_numbers as ase_atomic_numbers

    if bond_data is None:
        bond_data = BondData(scale=1.0)

    # Convert symbols to atomic numbers if needed
    resolved = []
    for item in atom_numbers:
        if isinstance(item, str):
            resolved.append(ase_atomic_numbers[item])
        else:
            resolved.append(int(item))

    unique = sorted(set(resolved))
    blmin = {}
    for i, z1 in enumerate(unique):
        for z2 in unique[i:]:
            r1 = bond_data.get_radius(z1)
            r2 = bond_data.get_radius(z2)
            blmin[(z1, z2)] = scale * (r1 + r2)
            blmin[(z2, z1)] = blmin[(z1, z2)]
    return blmin
