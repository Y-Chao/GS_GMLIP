"""The Interface class: central structure object (substrate + adsorbates/clusters)."""

from __future__ import annotations

from typing import Optional

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms

from gs_gmlip.interface.analysis import classify_appended

# Default cubic cell edge (Å) used when no cell is supplied.
DEFAULT_CELL_EDGE = 10.0


class Interface:
    """A catalyst substrate plus an interface (substrate + adsorbates/clusters).

    Indices 0..N-1 of ``interface`` are the substrate; N..M are appended
    adsorbate/cluster atoms. ``fix``/``relax`` index the substrate;
    ``adsList``/``clusterList`` are grouped index lists into ``interface``.
    """

    def __init__(
        self,
        substrate: Optional[Atoms] = None,
        interface: Optional[Atoms] = None,
        fixlist: Optional[list] = None,
        relaxlist: Optional[list] = None,
        adsList: Optional[list[list[int]]] = None,
        clusterList: Optional[list[list[int]]] = None,
        split_mol_on_cluster: bool = True,
        **kwargs,
    ):
        self.split_mol_on_cluster = split_mol_on_cluster

        self.substrate = substrate if substrate is not None else Atoms()
        self.interface = (
            interface if interface is not None else self.substrate.copy()
        )

        self._init_cell(**kwargs)
        self._init_fix_relax(fixlist, relaxlist)
        self._init_ads_cluster(adsList, clusterList)

    def _init_cell(self, **kwargs) -> None:
        if self.substrate.cell.rank == 0:
            cell = kwargs.get("cell")
            pbc = kwargs.get("pbc")
            cell = cell if cell is not None else [DEFAULT_CELL_EDGE] * 3
            pbc = pbc if pbc is not None else [False] * 3
            for atoms in (self.substrate, self.interface):
                atoms.set_cell(cell)
                atoms.set_pbc(pbc)

    def _init_fix_relax(self, fixlist, relaxlist) -> None:
        n = len(self.substrate)
        if fixlist is not None:
            if fixlist and (max(fixlist) >= n or min(fixlist) < 0):
                raise ValueError("fixlist contains invalid atom indices.")
            self.fix = list(fixlist)
        else:
            self.fix = [
                c.index
                for c in self.substrate.constraints
                if isinstance(c, FixAtoms)
            ]

        if relaxlist is not None:
            if relaxlist and (max(relaxlist) >= n or min(relaxlist) < 0):
                raise ValueError("relaxlist contains invalid atom indices.")
            self.relax = list(relaxlist)
        else:
            fixed = set(int(i) for i in self._flat_fix())
            self.relax = sorted(set(range(n)) - fixed)

    def _flat_fix(self) -> list[int]:
        """Return self.fix flattened to a plain list of ints (handles array or list)."""
        if len(self.fix) == 0:
            return []
        arr = np.atleast_1d(np.asarray(self.fix)).ravel()
        return [int(i) for i in arr]

    def _init_ads_cluster(self, adsList, clusterList) -> None:
        n_sub, n_int = len(self.substrate), len(self.interface)
        if n_int < n_sub:
            raise ValueError(
                "interface has fewer atoms than substrate; cannot classify."
            )
        if adsList is not None and clusterList is not None:
            self.adsList = adsList
            self.clusterList = clusterList
            return
        if n_int == n_sub:
            self.adsList, self.clusterList = [], []
            return
        appended = self.interface[n_sub:n_int]
        ads, clu = classify_appended(
            appended, split_mol_on_cluster=self.split_mol_on_cluster
        )
        self.adsList = [[i + n_sub for i in g] for g in ads]
        self.clusterList = [[i + n_sub for i in g] for g in clu]

    def __repr__(self) -> str:
        return (
            f"== Interface ==\n\tSubstrate: {self.substrate}\n"
            f"\tInterface: {self.interface}"
        )
