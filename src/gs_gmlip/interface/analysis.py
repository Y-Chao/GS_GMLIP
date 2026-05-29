"""Pure structure-analysis functions backing the Interface class.

These take ASE Atoms (or index subsets) and return plain data, so they can be
unit-tested without constructing an Interface.
"""

from __future__ import annotations

from functools import lru_cache

import networkx as nx
import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from pymatgen.core import Element

from gs_gmlip.interface.constants import BOND_SCALE


@lru_cache(maxsize=None)
def is_metal(symbol: str) -> bool:
    """Return True if the element is a metal (pymatgen definition; metalloids -> False)."""
    return bool(Element(symbol).is_metal)


def build_neighbor_graph(atoms: Atoms, bond_scale: float = BOND_SCALE) -> nx.Graph:
    """Build an undirected bond graph over `atoms`.

    Two atoms i, j are bonded when distance(i, j) <= bond_scale * (r_cov_i + r_cov_j).
    Nodes are atom indices into `atoms`. Uses the minimum-image convention if the
    cell is periodic.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(atoms)))
    if len(atoms) < 2:
        return graph

    numbers = atoms.get_atomic_numbers()
    radii = np.array([covalent_radii[z] for z in numbers])
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            d = atoms.get_distance(i, j, mic=True)
            if d <= bond_scale * (radii[i] + radii[j]):
                graph.add_edge(i, j)
    return graph


def connected_components(graph: nx.Graph) -> list[list[int]]:
    """Return connected components as sorted lists of node indices."""
    return [sorted(c) for c in nx.connected_components(graph)]


def classify_appended(
    atoms: Atoms,
    split_mol_on_cluster: bool = True,
    bond_scale: float = BOND_SCALE,
) -> tuple[list[list[int]], list[list[int]]]:
    """Classify appended atoms into adsorbate groups and cluster groups.

    Args:
        atoms: The appended region only (indices local to this Atoms).
        split_mol_on_cluster: If True, separate molecular adsorbate fragments
            from a bonded metal core; if False, a metal-containing component is
            kept whole as one cluster.
        bond_scale: Bond cutoff scale passed to build_neighbor_graph.

    Returns:
        (ads_groups, cluster_groups) — each a list of index-groups (lists),
        with indices local to `atoms`.

    Rule (split=True): within a metal-containing component, an adsorbate is each
    maximal connected non-metal fragment with >=1 internal non-metal bond (size
    >=2); all metal atoms plus any lone/atomic non-metals (size-1 fragments) form
    the cluster. A component containing no metal is always one adsorbate group,
    regardless of size (so a lone non-metal with no nearby metal is a size-1
    adsorbate); the "lone non-metal joins the cluster" behavior applies only
    within a metal-containing component.
    """
    syms = atoms.get_chemical_symbols()
    graph = build_neighbor_graph(atoms, bond_scale=bond_scale)
    ads_groups: list[list[int]] = []
    cluster_groups: list[list[int]] = []

    for comp in connected_components(graph):
        if not any(is_metal(syms[i]) for i in comp):
            ads_groups.append(sorted(comp))
            continue
        if not split_mol_on_cluster:
            cluster_groups.append(sorted(comp))
            continue

        nonmetal = [i for i in comp if not is_metal(syms[i])]
        metals = [i for i in comp if is_metal(syms[i])]
        sub = graph.subgraph(nonmetal)
        cluster = list(metals)
        for frag in connected_components(sub):
            if len(frag) >= 2:
                ads_groups.append(sorted(frag))
            else:
                cluster.extend(frag)
        cluster_groups.append(sorted(cluster))

    return ads_groups, cluster_groups
