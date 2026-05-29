"""Tunable constants for the Interface structure model."""

from __future__ import annotations

import numpy as np

# Atomic numbers of noble-gas closed shells, used by the valence-capacity
# heuristic in compute_possible_bond: an atom "wants" as many bonds as its
# distance to the nearest closed shell.
CLOSED_SHELLS = np.array([2, 10, 18, 36, 54, 86])

# Default cubic cell edge (Å) used when no cell is supplied.
DEFAULT_CELL_EDGE = 10.0

# Neighbor-graph bond cutoff: two atoms are bonded if their distance is
# <= BOND_SCALE * (covalent_radius_i + covalent_radius_j).
BOND_SCALE = 1.2

# Minimum number of atoms a metal-containing component must have to be
# classified as a cluster (smaller metal components still classify as cluster
# but this lets callers raise the bar).
DEFAULT_CLUSTER_MIN_SIZE = 1

# Layer-detection z tolerance in Å: atoms within this z-spread share a layer.
DEFAULT_LAYER_TOL = 0.5
