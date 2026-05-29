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

# Neighbor-graph bond cutoff: two atoms are bonded when their distance is
# <= BOND_SCALE * (covalent_radius_i + covalent_radius_j).
BOND_SCALE = 1.2

# Atomic numbers of noble-gas closed shells, used by the valence-capacity
# heuristic in compute_possible_bond: an atom "wants" as many bonds as its
# distance to the nearest closed shell.
CLOSED_SHELLS = np.array([2, 10, 18, 36, 54, 86])

# Layer-detection z tolerance in Å: atoms within this z-spread share a layer.
DEFAULT_LAYER_TOL = 0.5


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


def compute_bondmatrix(atoms: Atoms, bond_scale: float = BOND_SCALE) -> np.ndarray:
    """Return an (N, N) integer adjacency matrix (1 = bonded, 0 = not, diag 0)."""
    n = len(atoms)
    matrix = np.zeros((n, n), dtype=int)
    graph = build_neighbor_graph(atoms, bond_scale=bond_scale)
    for i, j in graph.edges():
        matrix[i, j] = 1
        matrix[j, i] = 1
    return matrix


def expected_bonds(z: int) -> int:
    """Expected bond capacity = distance from atomic number z to nearest closed shell.

    Closed shells are the noble-gas atomic numbers (2, 10, 18, 36, 54, 86). The
    smaller of the distances to the surrounding shells is how many bonds the atom
    tends to form (H->1, O->2, N->3, C->4, Pt->8).
    """
    return int(np.min(np.abs(z - CLOSED_SHELLS)))


def compute_possible_bond(
    atoms: Atoms, bond_scale: float = BOND_SCALE
) -> dict[int, int]:
    """Return atom index -> number of *additional* bonds it can form (>= 0)."""
    matrix = compute_bondmatrix(atoms, bond_scale=bond_scale)
    current = matrix.sum(axis=1)
    numbers = atoms.get_atomic_numbers()
    return {
        i: max(0, expected_bonds(int(numbers[i])) - int(current[i]))
        for i in range(len(atoms))
    }


def detect_layers(atoms: Atoms, tol: float = DEFAULT_LAYER_TOL) -> np.ndarray:
    """Assign each atom a layer index (0 = lowest z) by greedy z-clustering.

    Atoms are sorted by z; a new layer starts when the gap to the previous
    atom's z exceeds `tol`. Returns an int array of length len(atoms), indexed
    in the atoms' original order.
    """
    if len(atoms) == 0:
        return np.empty(0, dtype=int)
    z = atoms.get_positions()[:, 2]
    order = np.argsort(z)
    labels = np.empty(len(atoms), dtype=int)
    current = 0
    prev_z = z[order[0]]
    for rank, idx in enumerate(order):
        if rank > 0 and (z[idx] - prev_z) > tol:
            current += 1
        labels[idx] = current
        prev_z = z[idx]
    return labels


def fingerprint(atoms: Atoms) -> np.ndarray:
    """Order-invariant structural fingerprint.

    For every atom pair, compute (Z_i * Z_j) / distance, then return the sorted
    vector of these values. Two structures that differ only by atom ordering
    produce identical fingerprints. Returns an empty array for <2 atoms.
    """
    n = len(atoms)
    if n < 2:
        return np.zeros(0)
    numbers = atoms.get_atomic_numbers()
    feats = []
    for i in range(n):
        for j in range(i + 1, n):
            d = atoms.get_distance(i, j, mic=True)
            feats.append(numbers[i] * numbers[j] / (d + 1e-9))
    return np.sort(np.array(feats))
