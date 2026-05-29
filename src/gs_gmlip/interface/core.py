"""The Interface class: central structure object (substrate + adsorbates/clusters)."""

from __future__ import annotations

from functools import cached_property
from typing import Optional

from ase import Atoms
from ase.constraints import FixAtoms

from gs_gmlip.interface.analysis import (
    classify_appended,
    compute_bondmatrix,
    compute_possible_bond,
    detect_layers,
    fingerprint as _fingerprint,
)
from gs_gmlip.interface.builders import primitive_slab_from_bulk, slab_from_bulk

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
            self.fix = sorted({int(i) for i in fixlist})
        else:
            self.fix = sorted(
                {
                    int(i)
                    for c in self.substrate.constraints
                    if isinstance(c, FixAtoms)
                    for i in c.index
                }
            )

        if relaxlist is not None:
            if relaxlist and (max(relaxlist) >= n or min(relaxlist) < 0):
                raise ValueError("relaxlist contains invalid atom indices.")
            self.relax = sorted({int(i) for i in relaxlist})
        else:
            self.relax = sorted(set(range(n)) - set(self.fix))

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

    _CACHED = ("bondmatrix", "possible_bond", "fingerprint", "_layer_labels")

    @cached_property
    def bondmatrix(self):
        """(N, N) integer adjacency matrix of the full interface (cached)."""
        return compute_bondmatrix(self.interface)

    @cached_property
    def possible_bond(self) -> dict[int, int]:
        """Atom index -> additional bonds it can form (valence capacity, cached)."""
        return compute_possible_bond(self.interface)

    @cached_property
    def fingerprint(self):
        """Order-invariant structural fingerprint of the interface (cached)."""
        return _fingerprint(self.interface)

    @cached_property
    def _layer_labels(self):
        """Per-atom substrate layer index (cached)."""
        return detect_layers(self.substrate)

    def _reset_cache(self) -> None:
        """Drop all cached_property values (call after any mutation)."""
        for key in self._CACHED:
            self.__dict__.pop(key, None)

    def _set_from_slab(self, slab: Atoms) -> None:
        """Reset substrate/interface to a freshly built slab (clears fix/ads/cluster)."""
        self.substrate = slab
        self.interface = slab.copy()
        self.fix = []
        self.relax = list(range(len(slab)))
        self.adsList = []
        self.clusterList = []
        self._reset_cache()

    def get_substrate_layers(self) -> int:
        """Number of atomic layers in the substrate (z-clustering)."""
        if len(self.substrate) == 0:
            return 0
        return int(self._layer_labels.max()) + 1

    def fix_substrate(
        self, layers: int | float, symmetry: Optional[str] = None
    ) -> None:
        """Fix the bottom layers of the substrate.

        Args:
            layers: If an int, fix the bottom ``layers`` atomic layers. If a
                float, fix every substrate atom with z <= ``layers``.
            symmetry: Reserved for Phase 2 (symmetric fixing); currently unused.

        Sets ``self.fix`` (flat sorted list of ints), recomputes ``self.relax``,
        applies a FixAtoms constraint to substrate and interface, and resets the
        derived-data cache.
        """
        if isinstance(layers, bool):
            raise TypeError("layers must be int or float, not bool")
        if isinstance(layers, int):
            labels = self._layer_labels
            n_layers = int(labels.max()) + 1 if len(self.substrate) else 0
            keep = set(range(min(layers, n_layers)))  # bottom layer indices
            fixed = [i for i in range(len(self.substrate)) if labels[i] in keep]
        else:
            z = self.substrate.get_positions()[:, 2]
            fixed = [i for i in range(len(self.substrate)) if z[i] <= layers]
        self.fix = sorted(fixed)
        self.relax = sorted(set(range(len(self.substrate))) - set(self.fix))
        for atoms in (self.substrate, self.interface):
            atoms.set_constraint(FixAtoms(indices=self.fix))
        # geometry is unchanged (only constraints); cache reset is conservative
        self._reset_cache()

    def build_from_bulk(
        self,
        bulk_structure,
        miller_index,
        layer: int = 4,
        vacuum: float = 15.0,
        symmetry: bool = False,
    ) -> None:
        """Build substrate/interface from a bulk structure and Miller index."""
        slab = slab_from_bulk(bulk_structure, miller_index, layer, vacuum, symmetry)
        self._set_from_slab(slab)

    def build_primitive_surface_from_bulk(
        self,
        bulk_structure,
        miller_index,
        layer: int = 4,
        vacuum: float = 15.0,
        symmetry: bool = False,
    ) -> None:
        """Build a primitive-cell surface as substrate/interface."""
        slab = primitive_slab_from_bulk(
            bulk_structure, miller_index, layer, vacuum, symmetry
        )
        self._set_from_slab(slab)

    def wrap(self) -> None:
        """Wrap interface atoms into the cell, then reset caches.

        No-op if the interface has no periodic boundary conditions (relies on
        the cell's pbc).
        """
        self.interface.wrap()
        self._reset_cache()

    def align_interface(self, loc: str = "center") -> None:
        """Shift the appended (adsorbate/cluster) region in z relative to the slab.

        Args:
            loc: ``"center"`` places the appended atoms' MEAN z-coordinate
                (not mass-weighted) at the slab's z-midpoint; ``"bottom"`` drops
                the appended atoms so their lowest atom sits ~2 Å above the
                slab's top atom.

        No-op when there are no appended atoms. Resets the derived-data cache.
        """
        n_sub = len(self.substrate)
        if len(self.interface) <= n_sub:
            return
        pos = self.interface.get_positions()
        slab_top = pos[:n_sub, 2].max()
        slab_bottom = pos[:n_sub, 2].min()
        ads_z = pos[n_sub:, 2]
        if loc == "bottom":
            shift = (slab_top + 2.0) - ads_z.min()
        elif loc == "center":
            slab_mid = (slab_bottom + slab_top) / 2.0
            shift = slab_mid - ads_z.mean()
        else:
            raise ValueError(f"unknown loc: {loc!r}")
        pos[n_sub:, 2] += shift
        self.interface.set_positions(pos)
        self._reset_cache()

    def center_interface(self) -> None:
        """Align the appended region to the slab center."""
        self.align_interface(loc="center")

    def bottom_interface(self) -> None:
        """Align the appended region just above the slab top."""
        self.align_interface(loc="bottom")
