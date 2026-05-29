"""Pure structure-analysis functions backing the Interface class.

These take ASE Atoms (or index subsets) and return plain data, so they can be
unit-tested without constructing an Interface.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from pymatgen.core import Element

from gs_gmlip.interface.constants import BOND_SCALE


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
