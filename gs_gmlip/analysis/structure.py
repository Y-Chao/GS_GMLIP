"""Structure analysis: similarity, clustering, and comparison.

Provides tools for analyzing collections of structures found
during global optimization: deduplication, clustering by similarity,
and structural descriptor computation.
"""

from __future__ import annotations

import logging

import numpy as np
from ase import Atoms

from gs_gmlip.structure.bonds import get_bond_matrix

logger = logging.getLogger(__name__)


def pair_distribution(
    atoms: Atoms,
    r_max: float = 8.0,
    n_bins: int = 100,
    active_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the pair distribution function.

    Parameters
    ----------
    atoms : Atoms
        Input structure.
    r_max : float
        Maximum distance (Å).
    n_bins : int
        Number of histogram bins.
    active_only : bool
        Only include active (tagged > 0) atoms.

    Returns
    -------
    r : ndarray
        Bin centers.
    g : ndarray
        Pair distribution values.
    """
    if active_only:
        indices = [i for i, t in enumerate(atoms.get_tags()) if t > 0]
        if not indices:
            indices = list(range(len(atoms)))
    else:
        indices = list(range(len(atoms)))

    from ase.geometry import get_distances

    _, D = get_distances(
        atoms.positions[indices],
        atoms.positions,
        cell=atoms.cell,
        pbc=atoms.pbc,
    )
    dists = D.ravel()
    dists = dists[dists > 0.1]
    dists = dists[dists < r_max]

    bins = np.linspace(0, r_max, n_bins + 1)
    r = 0.5 * (bins[:-1] + bins[1:])
    g, _ = np.histogram(dists, bins=bins)
    return r, g.astype(float)


def fingerprint_distance(
    atoms1: Atoms,
    atoms2: Atoms,
    r_max: float = 8.0,
    n_bins: int = 100,
) -> float:
    """Compute distance between two structures using pair distribution fingerprints.

    Parameters
    ----------
    atoms1, atoms2 : Atoms
        Structures to compare.
    r_max, n_bins : float, int
        Fingerprint parameters.

    Returns
    -------
    dist : float
        Euclidean distance between fingerprints.
    """
    _, g1 = pair_distribution(atoms1, r_max, n_bins)
    _, g2 = pair_distribution(atoms2, r_max, n_bins)

    # Normalize
    n1 = np.linalg.norm(g1)
    n2 = np.linalg.norm(g2)
    if n1 > 0:
        g1 = g1 / n1
    if n2 > 0:
        g2 = g2 / n2

    return float(np.linalg.norm(g1 - g2))


class StructureAnalyzer:
    """Analyze and cluster a collection of structures.

    Parameters
    ----------
    structures : list of Atoms
        Structures to analyze.
    energies : list of float, optional
        Corresponding energies.
    """

    def __init__(
        self,
        structures: list[Atoms],
        energies: list[float] | None = None,
    ):
        self.structures = structures
        self.energies = energies or [np.nan] * len(structures)

    def distance_matrix(self, r_max: float = 8.0, n_bins: int = 100) -> np.ndarray:
        """Compute pairwise fingerprint distances.

        Returns
        -------
        D : ndarray, shape (n, n)
            Distance matrix.
        """
        n = len(self.structures)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = fingerprint_distance(
                    self.structures[i], self.structures[j], r_max, n_bins
                )
                D[i, j] = d
                D[j, i] = d
        return D

    def deduplicate(self, threshold: float = 0.1) -> list[int]:
        """Remove duplicate structures based on fingerprint distance.

        Parameters
        ----------
        threshold : float
            Structures with distance < threshold are considered duplicates.

        Returns
        -------
        unique_indices : list of int
            Indices of unique structures.
        """
        D = self.distance_matrix()
        n = len(self.structures)
        is_duplicate = [False] * n

        for i in range(n):
            if is_duplicate[i]:
                continue
            for j in range(i + 1, n):
                if D[i, j] < threshold:
                    is_duplicate[j] = True

        return [i for i in range(n) if not is_duplicate[i]]

    def cluster(self, n_clusters: int = 5, method: str = "agglomerative") -> np.ndarray:
        """Cluster structures by similarity.

        Parameters
        ----------
        n_clusters : int
            Number of clusters.
        method : str
            'agglomerative' (default) uses hierarchical clustering.

        Returns
        -------
        labels : ndarray, shape (n,)
            Cluster labels.
        """
        from sklearn.cluster import AgglomerativeClustering

        D = self.distance_matrix()

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average",
        )
        labels = clustering.fit_predict(D)
        return labels

    def get_cluster_representatives(self, n_clusters: int = 5) -> list[int]:
        """Get the lowest-energy structure from each cluster.

        Returns
        -------
        representatives : list of int
            Indices of representative structures.
        """
        labels = self.cluster(n_clusters)
        reps = []
        for c in range(n_clusters):
            members = [i for i, l in enumerate(labels) if l == c]
            if not members:
                continue
            best = min(members, key=lambda i: self.energies[i])
            reps.append(best)
        reps.sort(key=lambda i: self.energies[i])
        return reps

    def energy_landscape_summary(self) -> dict:
        """Summary statistics of the energy landscape."""
        e = np.array(self.energies)
        valid = e[~np.isnan(e)]
        if len(valid) == 0:
            return {"n_structures": len(self.structures)}
        return {
            "n_structures": len(self.structures),
            "energy_min": float(valid.min()),
            "energy_max": float(valid.max()),
            "energy_mean": float(valid.mean()),
            "energy_std": float(valid.std()),
            "energy_range": float(valid.max() - valid.min()),
        }
