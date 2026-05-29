# Interface Class Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the `Interface` class — the central structure object for `gs_gmlip` representing a catalyst substrate + adsorbates/clusters, with pymatgen-backed surface analysis, adsorbate/cluster classification, cached derived data, and a conversion/store/generator API.

**Architecture:** Composition wrapper — `Interface` *has* two ASE `Atoms` (`substrate`, `interface`); it does not subclass `Atoms`. Pure pymatgen math lives in free-function modules (`analysis.py`, `builders.py`); the class (`core.py`) owns state, `cached_property` caching with manual reset, mutation, and I/O. `interface.py` becomes the `interface/` package; the public import path `from gs_gmlip.interface import Interface` is preserved via `__init__.py`.

**Tech Stack:** Python ≥3.12, ASE, pymatgen (new required dep), networkx (transitive via pymatgen), numpy, pytest.

**Reference spec:** `docs/superpowers/specs/2026-05-29-interface-class-design.md`

---

## Conventions (apply to every task)

- Every `.py` file starts with a module docstring then `from __future__ import annotations`.
- Line length 88 (ruff). Google/NumPy docstrings on public functions.
- Run a single test with: `.venv/bin/pytest tests/path::test_name -v` (the repo's `.venv`; note `uv run pytest` may pick the wrong interpreter if `VIRTUAL_ENV` points elsewhere — prefer `.venv/bin/pytest` directly).
- Commit after each task with a `feat:` / `test:` / `refactor:` prefix.

## File Structure

```
src/gs_gmlip/interface/
├── __init__.py      # re-export: from gs_gmlip.interface.core import Interface
├── constants.py     # CLOSED_SHELLS, DEFAULT_* tunables
├── analysis.py      # pure fns: is_metal, build_neighbor_graph, connected_components,
│                    #           classify_appended, compute_bondmatrix,
│                    #           compute_possible_bond, detect_layers, fingerprint
├── builders.py      # pure fns: slab_from_bulk, primitive_slab_from_bulk
└── core.py          # Interface class
```

```
tests/
├── test_interface_analysis.py   # analysis.py free functions
├── test_interface_builders.py   # builders.py free functions
└── test_interface_core.py       # Interface class
```

The current `src/gs_gmlip/interface.py` is deleted (its `__init__`/validation logic moves into `core.py`, the skeleton method bodies are filled in). The current `tests/test_interface.py` (empty stub) is replaced by the three files above.

---

### Task 0: Add pymatgen dependency and create package skeleton

**Files:**
- Modify: `pyproject.toml` (dependencies list, lines ~9-20)
- Delete: `src/gs_gmlip/interface.py`
- Create: `src/gs_gmlip/interface/__init__.py`
- Create: `src/gs_gmlip/interface/constants.py`
- Create: `src/gs_gmlip/interface/core.py` (placeholder Interface so import works)

- [ ] **Step 1: Add pymatgen to dependencies**

In `pyproject.toml`, add to the `dependencies` array (keep alphabetical-ish ordering near `pandas`/`plotly`):

```toml
    "pymatgen>=2024.0.0",
```

- [ ] **Step 2: Remove the old single-file module and create the package**

```bash
git rm src/gs_gmlip/interface.py
mkdir -p src/gs_gmlip/interface
```

Create `src/gs_gmlip/interface/constants.py`:

```python
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
```

Create `src/gs_gmlip/interface/core.py`:

```python
"""The Interface class: central structure object (substrate + adsorbates/clusters)."""

from __future__ import annotations


class Interface:
    """Placeholder; implemented across later tasks."""
```

Create `src/gs_gmlip/interface/__init__.py`:

```python
"""Interface package: the central structure object for gs_gmlip."""

from __future__ import annotations

from gs_gmlip.interface.core import Interface

__all__ = ["Interface"]
```

- [ ] **Step 3: Install the new dependency into the repo venv**

Run: `unset VIRTUAL_ENV && uv sync && uv pip install -e ".[dev]" && uv pip install pymatgen pytest`
Expected: pymatgen resolves and installs; no errors.

- [ ] **Step 4: Verify the import path is preserved**

Run: `.venv/bin/python -c "from gs_gmlip.interface import Interface; print(Interface)"`
Expected: prints `<class 'gs_gmlip.interface.core.Interface'>`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/gs_gmlip/interface/
git commit -m "feat: scaffold interface package, add pymatgen dependency"
```

---

### Task 1: `is_metal` and neighbor-graph helpers (analysis.py)

**Files:**
- Create: `src/gs_gmlip/interface/analysis.py`
- Test: `tests/test_interface_analysis.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_interface_analysis.py`:

```python
"""Tests for gs_gmlip.interface.analysis."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from gs_gmlip.interface.analysis import (
    build_neighbor_graph,
    connected_components,
    is_metal,
)


def test_is_metal():
    assert is_metal("Pt") is True
    assert is_metal("Fe") is True
    assert is_metal("O") is False
    assert is_metal("C") is False
    assert is_metal("Si") is False  # metalloid -> not a metal


def test_build_neighbor_graph_diatomic():
    # CO with a ~1.13 Å bond is bonded; two far-apart atoms are not.
    co = Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.13]], cell=[10, 10, 10])
    g = build_neighbor_graph(co)
    assert g.has_edge(0, 1)


def test_connected_components_two_molecules():
    # CO near origin, O2 far away -> two components.
    atoms = Atoms(
        "COOO",
        positions=[
            [0, 0, 0],
            [0, 0, 1.13],
            [5, 5, 5],
            [5, 5, 6.2],
        ],
        cell=[10, 10, 10],
    )
    g = build_neighbor_graph(atoms)
    comps = connected_components(g)
    comps_sorted = sorted([sorted(c) for c in comps])
    assert comps_sorted == [[0, 1], [2, 3]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_interface_analysis.py -v`
Expected: FAIL — `ModuleNotFoundError` / `ImportError` (analysis.py functions not defined).

- [ ] **Step 3: Write minimal implementation**

Create `src/gs_gmlip/interface/analysis.py`:

```python
"""Pure structure-analysis functions backing the Interface class.

These take ASE Atoms (or index subsets) and return plain data, so they can be
unit-tested without constructing an Interface.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_interface_analysis.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/gs_gmlip/interface/analysis.py tests/test_interface_analysis.py
git commit -m "feat: add is_metal and neighbor-graph helpers"
```

---

### Task 2: `classify_appended` — adsorbate/cluster split

**Files:**
- Modify: `src/gs_gmlip/interface/analysis.py`
- Test: `tests/test_interface_analysis.py`

This implements the spec's final classification rule. Atoms passed in are the
**appended region only** (a sub-Atoms object). Returned index groups are **local**
to that sub-Atoms (caller offsets by `n_substrate`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_interface_analysis.py`:

```python
from gs_gmlip.interface.analysis import classify_appended


def _pt4():
    # compact Pt4 tetrahedron-ish cluster, all within bonding distance
    return Atoms(
        "Pt4",
        positions=[[0, 0, 0], [2.6, 0, 0], [1.3, 2.25, 0], [1.3, 0.75, 2.1]],
        cell=[15, 15, 15],
    )


def test_classify_bare_metal_cluster():
    ads, clu = classify_appended(_pt4())
    assert ads == []
    assert sorted(clu[0]) == [0, 1, 2, 3]


def test_classify_free_molecule_no_metal():
    h2o = Atoms(
        "OHH",
        positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]],
        cell=[15, 15, 15],
    )
    ads, clu = classify_appended(h2o)
    assert clu == []
    assert sorted(ads[0]) == [0, 1, 2]


def test_classify_co_on_cluster_splits():
    # Pt4 cluster + CO bonded to one Pt (C atom ~2.0 Å above a Pt)
    atoms = _pt4()
    atoms += Atoms("CO", positions=[[0, 0, 2.0], [0, 0, 3.13]])
    ads, clu = classify_appended(atoms, split_mol_on_cluster=True)
    # CO (indices 4,5) is a >=2-atom non-metal fragment -> adsorbate
    assert sorted(ads[0]) == [4, 5]
    # Pt4 -> cluster
    assert sorted(clu[0]) == [0, 1, 2, 3]


def test_classify_atomic_o_on_cluster_joins_cluster():
    # Pt4 + single O bonded to a Pt: lone non-metal -> cluster, no adsorbate
    atoms = _pt4()
    atoms += Atoms("O", positions=[[0, 0, 1.9]])
    ads, clu = classify_appended(atoms, split_mol_on_cluster=True)
    assert ads == []
    assert 4 in clu[0]  # the O joined the cluster


def test_classify_no_split_keeps_whole_component():
    atoms = _pt4()
    atoms += Atoms("CO", positions=[[0, 0, 2.0], [0, 0, 3.13]])
    ads, clu = classify_appended(atoms, split_mol_on_cluster=False)
    assert ads == []
    assert sorted(clu[0]) == [0, 1, 2, 3, 4, 5]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_interface_analysis.py -k classify -v`
Expected: FAIL — `ImportError: cannot import name 'classify_appended'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/gs_gmlip/interface/analysis.py`:

```python
from gs_gmlip.interface.constants import DEFAULT_CLUSTER_MIN_SIZE


def _component_has_metal(atoms: Atoms, indices: list[int]) -> bool:
    syms = atoms.get_chemical_symbols()
    return any(is_metal(syms[i]) for i in indices)


def classify_appended(
    atoms: Atoms,
    split_mol_on_cluster: bool = True,
    cluster_min_size: int = DEFAULT_CLUSTER_MIN_SIZE,
    bond_scale: float = BOND_SCALE,
) -> tuple[list[list[int]], list[list[int]]]:
    """Classify appended atoms into adsorbate groups and cluster groups.

    Args:
        atoms: The appended region only (indices local to this Atoms).
        split_mol_on_cluster: If True, separate molecular adsorbate fragments
            from a bonded metal core; if False, a metal-containing component is
            kept whole as one cluster.
        cluster_min_size: A metal-containing component smaller than this still
            counts as a cluster (the value lets callers raise the threshold).
        bond_scale: Bond cutoff scale passed to build_neighbor_graph.

    Returns:
        (ads_groups, cluster_groups) — each a list of index-groups (lists),
        with indices local to `atoms`.

    Rule (split=True): within a metal-containing component, an adsorbate is each
    maximal connected non-metal fragment with >=1 internal non-metal bond (size
    >=2); all metal atoms plus any lone/atomic non-metals (size-1 fragments) form
    the cluster.
    """
    syms = atoms.get_chemical_symbols()
    graph = build_neighbor_graph(atoms, bond_scale=bond_scale)
    ads_groups: list[list[int]] = []
    cluster_groups: list[list[int]] = []

    for comp in connected_components(graph):
        if not _component_has_metal(atoms, comp):
            ads_groups.append(sorted(comp))
            continue
        if not split_mol_on_cluster:
            cluster_groups.append(sorted(comp))
            continue

        # split: non-metal subgraph -> fragments; metals + small fragments -> cluster
        nonmetal = [i for i in comp if not is_metal(syms[i])]
        metals = [i for i in comp if is_metal(syms[i])]
        sub = graph.subgraph(nonmetal)
        cluster = list(metals)
        for frag in connected_components(sub):
            if len(frag) >= 2:
                ads_groups.append(sorted(frag))
            else:
                cluster.extend(frag)  # lone non-metal joins cluster
        cluster_groups.append(sorted(cluster))

    return ads_groups, cluster_groups
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_interface_analysis.py -k classify -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/gs_gmlip/interface/analysis.py tests/test_interface_analysis.py
git commit -m "feat: add adsorbate/cluster classification (classify_appended)"
```

---

### Task 3: `compute_bondmatrix` and `compute_possible_bond`

**Files:**
- Modify: `src/gs_gmlip/interface/analysis.py`
- Test: `tests/test_interface_analysis.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_interface_analysis.py`:

```python
from gs_gmlip.interface.analysis import (
    compute_bondmatrix,
    compute_possible_bond,
    expected_bonds,
)


def test_expected_bonds_closed_shell():
    assert expected_bonds(1) == 1   # H -> |1-2|
    assert expected_bonds(8) == 2   # O -> |8-10|
    assert expected_bonds(7) == 3   # N -> |7-10|
    assert expected_bonds(6) == 4   # C -> min(|6-2|,|6-10|)
    assert expected_bonds(78) == 8  # Pt -> |78-86|


def test_compute_bondmatrix_symmetric_binary():
    co = Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.13]], cell=[10, 10, 10])
    m = compute_bondmatrix(co)
    assert m.shape == (2, 2)
    assert m[0, 1] == 1 and m[1, 0] == 1
    assert m[0, 0] == 0  # no self-bond


def test_compute_possible_bond_water():
    # O bonded to 2 H -> O has 0 free bonds; each H bonded to 1 -> 0 free.
    h2o = Atoms(
        "OHH",
        positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]],
        cell=[10, 10, 10],
    )
    pb = compute_possible_bond(h2o)
    assert pb == {0: 0, 1: 0, 2: 0}


def test_compute_possible_bond_lone_oxygen():
    # isolated O: expected 2 bonds, 0 current -> 2 free
    o = Atoms("O", positions=[[0, 0, 0]], cell=[10, 10, 10])
    pb = compute_possible_bond(o)
    assert pb == {0: 2}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_interface_analysis.py -k "bond" -v`
Expected: FAIL — import errors for `compute_bondmatrix` etc.

- [ ] **Step 3: Write minimal implementation**

Append to `src/gs_gmlip/interface/analysis.py`:

```python
from gs_gmlip.interface.constants import CLOSED_SHELLS


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_interface_analysis.py -k "bond" -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/gs_gmlip/interface/analysis.py tests/test_interface_analysis.py
git commit -m "feat: add bondmatrix and possible_bond (closed-shell valence heuristic)"
```

---

### Task 4: `detect_layers` and `fingerprint`

**Files:**
- Modify: `src/gs_gmlip/interface/analysis.py`
- Test: `tests/test_interface_analysis.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_interface_analysis.py`:

```python
from gs_gmlip.interface.analysis import detect_layers, fingerprint


def test_detect_layers_three_layers():
    # 3 atoms at z=0, 3 at z=2, 3 at z=4 -> 3 layers; returns layer index per atom
    z = [0.0, 0.05, 0.0, 2.0, 2.0, 1.95, 4.0, 4.0, 4.0]
    atoms = Atoms("H9", positions=[[0, 0, zi] for zi in z], cell=[10, 10, 10])
    labels = detect_layers(atoms, tol=0.5)
    assert labels.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_fingerprint_is_deterministic_and_order_invariant():
    a = Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.13]], cell=[10, 10, 10])
    b = Atoms("OC", positions=[[0, 0, 1.13], [0, 0, 0]], cell=[10, 10, 10])
    fa, fb = fingerprint(a), fingerprint(b)
    assert isinstance(fa, np.ndarray)
    assert np.allclose(fa, fb)  # same structure, different atom order
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_interface_analysis.py -k "layers or fingerprint" -v`
Expected: FAIL — import errors.

- [ ] **Step 3: Write minimal implementation**

Append to `src/gs_gmlip/interface/analysis.py`:

```python
from gs_gmlip.interface.constants import DEFAULT_LAYER_TOL


def detect_layers(atoms: Atoms, tol: float = DEFAULT_LAYER_TOL) -> np.ndarray:
    """Assign each atom a layer index (0 = lowest z) by greedy z-clustering.

    Atoms are sorted by z; a new layer starts when the gap to the previous
    atom's z exceeds `tol`. Returns an int array of length len(atoms), indexed
    in the atoms' original order.
    """
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
    """Order-invariant structural fingerprint: sorted vector of (Z_i*Z_j, distance).

    Starting descriptor: for every atom pair, the sorted list of
    distances weighted by the product of atomic numbers. Two structures that
    differ only by atom ordering produce identical fingerprints.
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_interface_analysis.py -k "layers or fingerprint" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/gs_gmlip/interface/analysis.py tests/test_interface_analysis.py
git commit -m "feat: add layer detection and structural fingerprint"
```

---

### Task 5: `builders.py` — slab from bulk

**Files:**
- Create: `src/gs_gmlip/interface/builders.py`
- Test: `tests/test_interface_builders.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_interface_builders.py`:

```python
"""Tests for gs_gmlip.interface.builders."""

from __future__ import annotations

from ase import Atoms
from ase.build import bulk

from gs_gmlip.interface.builders import primitive_slab_from_bulk, slab_from_bulk


def test_slab_from_bulk_returns_atoms_with_vacuum():
    pt = bulk("Pt", "fcc", a=3.92)
    slab = slab_from_bulk(pt, (1, 1, 1), layer=4, vacuum=15.0)
    assert isinstance(slab, Atoms)
    assert len(slab) > 0
    # vacuum present: z-extent of cell exceeds atomic z-spread
    zs = slab.get_positions()[:, 2]
    assert slab.cell[2, 2] > (zs.max() - zs.min()) + 10.0


def test_primitive_slab_not_larger_than_conventional():
    pt = bulk("Pt", "fcc", a=3.92)
    slab = slab_from_bulk(pt, (1, 1, 1), layer=4, vacuum=15.0)
    prim = primitive_slab_from_bulk(pt, (1, 1, 1), layer=4, vacuum=15.0)
    assert len(prim) <= len(slab)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_interface_builders.py -v`
Expected: FAIL — `ModuleNotFoundError` for builders.

- [ ] **Step 3: Write minimal implementation**

Create `src/gs_gmlip/interface/builders.py`:

```python
"""Pure slab-construction functions backed by pymatgen's SlabGenerator."""

from __future__ import annotations

from ase import Atoms
from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor


def _to_structure(bulk_structure: Atoms | Structure) -> Structure:
    if isinstance(bulk_structure, Structure):
        return bulk_structure
    return AseAtomsAdaptor.get_structure(bulk_structure)


def slab_from_bulk(
    bulk_structure: Atoms | Structure,
    miller_index: tuple[int, int, int],
    layer: int = 4,
    vacuum: float = 15.0,
    symmetry: bool = False,
) -> Atoms:
    """Build a slab from a bulk structure and Miller index, returned as ASE Atoms.

    Args:
        bulk_structure: Bulk cell (ASE Atoms or pymatgen Structure).
        miller_index: Miller index of the surface, e.g. (1, 1, 1).
        layer: Minimum number of atomic layers (slab thickness, in layers).
        vacuum: Vacuum thickness in Å added along the surface normal.
        symmetry: If True, require symmetric (both surfaces equivalent) slabs.
    """
    structure = _to_structure(bulk_structure)
    gen = SlabGenerator(
        structure,
        miller_index=miller_index,
        min_slab_size=layer,
        min_vacuum_size=vacuum,
        in_unit_planes=True,
        center_slab=True,
    )
    slabs = gen.get_slabs(symmetrize=symmetry)
    slab = slabs[0]
    return AseAtomsAdaptor.get_atoms(slab)


def primitive_slab_from_bulk(
    bulk_structure: Atoms | Structure,
    miller_index: tuple[int, int, int],
    layer: int = 4,
    vacuum: float = 15.0,
    symmetry: bool = False,
) -> Atoms:
    """Like slab_from_bulk but reduces the slab to its primitive surface cell."""
    structure = _to_structure(bulk_structure)
    gen = SlabGenerator(
        structure,
        miller_index=miller_index,
        min_slab_size=layer,
        min_vacuum_size=vacuum,
        in_unit_planes=True,
        center_slab=True,
        primitive=True,
    )
    slabs = gen.get_slabs(symmetrize=symmetry)
    slab = slabs[0].get_orthogonal_c_slab()
    return AseAtomsAdaptor.get_atoms(slab)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_interface_builders.py -v`
Expected: PASS (2 tests). If `get_orthogonal_c_slab()` raises for a given cell, remove that call and return `AseAtomsAdaptor.get_atoms(slabs[0])`.

- [ ] **Step 5: Commit**

```bash
git add src/gs_gmlip/interface/builders.py tests/test_interface_builders.py
git commit -m "feat: add slab builders from bulk (pymatgen SlabGenerator)"
```

---

### Task 6: `Interface.__init__` + validation + `__repr__`

**Files:**
- Modify: `src/gs_gmlip/interface/core.py`
- Test: `tests/test_interface_core.py`

Ports the skeleton's `__init__` logic, adapting it to **grouped** ads/cluster
lists and delegating classification to `analysis.classify_appended`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_interface_core.py`:

```python
"""Tests for the Interface class."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms

from gs_gmlip.interface import Interface


def _slab():
    # two-layer 2x2 H "slab" stand-in (cheap, no calculator needed)
    pos = [[x, y, z] for z in (0.0, 2.0) for x in (0.0, 2.0) for y in (0.0, 2.0)]
    return Atoms("H8", positions=pos, cell=[4, 4, 12], pbc=[True, True, False])


def test_init_bare_substrate():
    intf = Interface(substrate=_slab())
    assert len(intf.substrate) == 8
    assert len(intf.interface) == 8
    assert intf.adsList == []
    assert intf.clusterList == []
    assert intf.fix == []
    assert sorted(intf.relax) == list(range(8))


def test_init_with_adsorbate_classifies():
    slab = _slab()
    full = slab + Atoms("CO", positions=[[2.0, 2.0, 4.0], [2.0, 2.0, 5.13]])
    intf = Interface(substrate=slab, interface=full)
    # CO appended at indices 8,9 -> one adsorbate group, no clusters
    assert intf.clusterList == []
    assert sorted(intf.adsList[0]) == [8, 9]


def test_init_fixlist_validation():
    with pytest.raises(ValueError):
        Interface(substrate=_slab(), fixlist=[99])


def test_init_reads_existing_fixatoms_constraint():
    slab = _slab()
    slab.set_constraint(FixAtoms(indices=[0, 1, 2, 3]))
    intf = Interface(substrate=slab)
    assert sorted(np.concatenate(intf.fix).tolist()) == [0, 1, 2, 3]


def test_repr_contains_interface():
    intf = Interface(substrate=_slab())
    assert "Interface" in repr(intf)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_interface_core.py -v`
Expected: FAIL — placeholder Interface has no `substrate` attribute.

- [ ] **Step 3: Write minimal implementation**

Replace `src/gs_gmlip/interface/core.py` with:

```python
"""The Interface class: central structure object (substrate + adsorbates/clusters)."""

from __future__ import annotations

from typing import Optional

from ase import Atoms
from ase.constraints import FixAtoms

from gs_gmlip.interface.analysis import classify_appended
from gs_gmlip.interface.constants import (
    DEFAULT_CELL_EDGE,
    DEFAULT_CLUSTER_MIN_SIZE,
)


class Interface:
    """A catalyst substrate plus an interface (substrate + adsorbates/clusters).

    Indices 0..N-1 of `interface` are the substrate; N..M are appended
    adsorbate/cluster atoms. fix/relax index the substrate; adsList/clusterList
    are grouped index lists into `interface`.
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
        cluster_min_size: int = DEFAULT_CLUSTER_MIN_SIZE,
        **kwargs,
    ):
        self.split_mol_on_cluster = split_mol_on_cluster
        self.cluster_min_size = cluster_min_size

        self.substrate = substrate if substrate is not None else Atoms()
        self.interface = interface if interface is not None else self.substrate.copy()

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
                list(c.index)
                for c in self.substrate.constraints
                if isinstance(c, FixAtoms)
            ]
            self.fix = self.fix[0] if self.fix else []

        if relaxlist is not None:
            if relaxlist and (max(relaxlist) >= n or min(relaxlist) < 0):
                raise ValueError("relaxlist contains invalid atom indices.")
            self.relax = list(relaxlist)
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
            appended,
            split_mol_on_cluster=self.split_mol_on_cluster,
            cluster_min_size=self.cluster_min_size,
        )
        # offset local indices back to interface indices
        self.adsList = [[i + n_sub for i in g] for g in ads]
        self.clusterList = [[i + n_sub for i in g] for g in clu]

    def __repr__(self) -> str:
        return (
            f"== Interface ==\n\tSubstrate: {self.substrate}\n"
            f"\tInterface: {self.interface}"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_interface_core.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/gs_gmlip/interface/core.py tests/test_interface_core.py
git commit -m "feat: implement Interface __init__, validation, classification wiring"
```

---

### Task 7: Cached properties + `_reset_cache`

**Files:**
- Modify: `src/gs_gmlip/interface/core.py`
- Test: `tests/test_interface_core.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_interface_core.py`:

```python
def test_cached_properties_present_and_cached():
    slab = _slab()
    full = slab + Atoms("CO", positions=[[2.0, 2.0, 4.0], [2.0, 2.0, 5.13]])
    intf = Interface(substrate=slab, interface=full)
    bm1 = intf.bondmatrix
    assert bm1.shape == (10, 10)
    assert intf.bondmatrix is bm1  # cached: same object on second access
    assert isinstance(intf.possible_bond, dict)
    assert isinstance(intf.fingerprint, np.ndarray)


def test_reset_cache_clears_entries():
    intf = Interface(substrate=_slab())
    _ = intf.bondmatrix
    assert "bondmatrix" in intf.__dict__
    intf._reset_cache()
    assert "bondmatrix" not in intf.__dict__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_interface_core.py -k "cached or reset" -v`
Expected: FAIL — `AttributeError: 'Interface' object has no attribute 'bondmatrix'`.

- [ ] **Step 3: Write minimal implementation**

Add imports at the top of `core.py`:

```python
from functools import cached_property
```

and import the analysis functions:

```python
from gs_gmlip.interface.analysis import (
    classify_appended,
    compute_bondmatrix,
    compute_possible_bond,
    detect_layers,
    fingerprint as _fingerprint,
)
```

Add these members to the `Interface` class:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_interface_core.py -k "cached or reset" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/gs_gmlip/interface/core.py tests/test_interface_core.py
git commit -m "feat: add cached_property derived data and _reset_cache"
```

---

### Task 8: Surface methods — `get_substrate_layers`, `fix_substrate`, builders

**Files:**
- Modify: `src/gs_gmlip/interface/core.py`
- Test: `tests/test_interface_core.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_interface_core.py`:

```python
from ase.build import bulk


def test_get_substrate_layers_two():
    intf = Interface(substrate=_slab())  # _slab has z=0 and z=2 -> 2 layers
    assert intf.get_substrate_layers() == 2


def test_fix_substrate_int_fixes_bottom_layer():
    intf = Interface(substrate=_slab())
    intf.fix_substrate(layers=1)  # fix bottom 1 layer (z=0 -> indices 0..3)
    assert sorted(intf.fix) == [0, 1, 2, 3]
    # FixAtoms applied to substrate
    assert any(isinstance(c, FixAtoms) for c in intf.substrate.constraints)


def test_fix_substrate_resets_cache():
    intf = Interface(substrate=_slab())
    _ = intf.bondmatrix
    intf.fix_substrate(layers=1)
    assert "bondmatrix" not in intf.__dict__


def test_build_from_bulk_sets_substrate():
    intf = Interface()
    intf.build_from_bulk(bulk("Pt", "fcc", a=3.92), (1, 1, 1), layer=3, vacuum=12.0)
    assert len(intf.substrate) > 0
    assert len(intf.interface) == len(intf.substrate)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_interface_core.py -k "layers or fix_substrate or build_from_bulk" -v`
Expected: FAIL — methods are still skeleton `pass`/missing.

- [ ] **Step 3: Write minimal implementation**

Add to imports in `core.py`:

```python
import numpy as np
from gs_gmlip.interface.builders import primitive_slab_from_bulk, slab_from_bulk
```

Add/replace these methods on `Interface`:

```python
    def get_substrate_layers(self) -> int:
        """Number of atomic layers in the substrate (z-clustering)."""
        if len(self.substrate) == 0:
            return 0
        return int(self._layer_labels.max()) + 1

    def fix_substrate(self, layers: int | float, symmetry: Optional[str] = None) -> None:
        """Fix the bottom `layers` layers (int) or atoms below z=`layers` (float).

        Sets self.fix, applies a FixAtoms constraint to substrate and interface,
        and resets cached data.
        """
        if isinstance(layers, int):
            labels = self._layer_labels
            n_layers = int(labels.max()) + 1
            keep = set(range(min(layers, n_layers)))  # bottom layer indices
            fixed = [i for i in range(len(self.substrate)) if labels[i] in keep]
        else:
            z = self.substrate.get_positions()[:, 2]
            fixed = [i for i in range(len(self.substrate)) if z[i] <= layers]
        self.fix = sorted(fixed)
        self.relax = sorted(set(range(len(self.substrate))) - set(self.fix))
        for atoms in (self.substrate, self.interface):
            atoms.set_constraint(FixAtoms(indices=self.fix))
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
        self.substrate = slab
        self.interface = slab.copy()
        self.fix, self.relax = [], list(range(len(slab)))
        self.adsList, self.clusterList = [], []
        self._reset_cache()

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
        self.substrate = slab
        self.interface = slab.copy()
        self.fix, self.relax = [], list(range(len(slab)))
        self.adsList, self.clusterList = [], []
        self._reset_cache()
```

Note: `_layer_labels` uses `_` prefix so it is excluded from `_reset_cache` retention concerns; add it to `_CACHED` (already listed in Task 7). The `symmetry` arg of `fix_substrate` is accepted for API stability but not used in Phase 1; document that in the docstring (already implicit) — leave a `# symmetry handling deferred to Phase 2` comment.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_interface_core.py -k "layers or fix_substrate or build_from_bulk" -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/gs_gmlip/interface/core.py tests/test_interface_core.py
git commit -m "feat: add layer count, fix_substrate, build_from_bulk surface methods"
```

---

### Task 9: Mutating geometry methods — `wrap`, `align_interface`, `center`/`bottom`

**Files:**
- Modify: `src/gs_gmlip/interface/core.py`
- Test: `tests/test_interface_core.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_interface_core.py`:

```python
def test_wrap_resets_cache_and_wraps():
    slab = _slab()
    slab.positions[0] += slab.cell[0]  # push atom outside cell along a
    intf = Interface(substrate=slab.copy(), interface=slab.copy())
    intf.interface.set_pbc([True, True, False])
    _ = intf.bondmatrix
    intf.wrap()
    assert "bondmatrix" not in intf.__dict__


def test_center_interface_shifts_adsorbate_z():
    slab = _slab()
    full = slab + Atoms("C", positions=[[2.0, 2.0, 8.0]])
    intf = Interface(substrate=slab, interface=full)
    intf.align_interface(loc="bottom")
    # after bottom alignment the adsorbate z is reduced toward the slab top
    assert intf.interface.get_positions()[8, 2] < 8.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_interface_core.py -k "wrap or center_interface" -v`
Expected: FAIL — `wrap`/`align_interface` are skeleton `pass`.

- [ ] **Step 3: Write minimal implementation**

Replace the skeleton `wrap` / `align_interface` / `center_interface` / `bottom_interface` with:

```python
    def wrap(self) -> None:
        """Wrap interface atoms into the cell, then reset caches."""
        self.interface.wrap()
        self._reset_cache()

    def align_interface(self, loc: str = "center") -> None:
        """Shift the appended (adsorbate/cluster) region in z relative to the slab.

        loc="center": place appended atoms' center of mass above the slab center.
        loc="bottom": drop appended atoms so their lowest atom sits just above the
        slab's top atom (a small 2.0 Å gap).
        """
        n_sub = len(self.substrate)
        if len(self.interface) <= n_sub:
            return
        pos = self.interface.get_positions()
        slab_top = pos[:n_sub, 2].max()
        ads_z = pos[n_sub:, 2]
        if loc == "bottom":
            shift = (slab_top + 2.0) - ads_z.min()
        elif loc == "center":
            slab_mid = (pos[:n_sub, 2].min() + slab_top) / 2.0
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_interface_core.py -k "wrap or center_interface" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/gs_gmlip/interface/core.py tests/test_interface_core.py
git commit -m "feat: add wrap and align_interface mutating methods"
```

---

### Task 10: Conversion API — `to_ase`/`from_ase`, `to_pymatgen`/`from_pymatgen`, `to_dict`/`from_dict`, `copy`

**Files:**
- Modify: `src/gs_gmlip/interface/core.py`
- Test: `tests/test_interface_core.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_interface_core.py`:

```python
from pymatgen.core import Structure


def test_to_ase_part_selection():
    slab = _slab()
    full = slab + Atoms("CO", positions=[[2.0, 2.0, 4.0], [2.0, 2.0, 5.13]])
    intf = Interface(substrate=slab, interface=full)
    assert len(intf.to_ase()) == 10
    assert len(intf.to_ase(part="substrate")) == 8


def test_to_pymatgen_returns_structure():
    slab = _slab()
    slab.set_pbc(True)
    intf = Interface(substrate=slab)
    struct = intf.to_pymatgen()
    assert isinstance(struct, Structure)


def test_dict_roundtrip():
    slab = _slab()
    full = slab + Atoms("CO", positions=[[2.0, 2.0, 4.0], [2.0, 2.0, 5.13]])
    intf = Interface(substrate=slab, interface=full)
    d = intf.to_dict()
    intf2 = Interface.from_dict(d)
    assert len(intf2.interface) == 10
    assert len(intf2.substrate) == 8
    assert intf2.adsList == intf.adsList


def test_from_ase_splits_at_n_substrate():
    slab = _slab()
    full = slab + Atoms("CO", positions=[[2.0, 2.0, 4.0], [2.0, 2.0, 5.13]])
    intf = Interface.from_ase(full, n_substrate=8)
    assert len(intf.substrate) == 8
    assert sorted(intf.adsList[0]) == [8, 9]


def test_copy_is_independent():
    intf = Interface(substrate=_slab())
    clone = intf.copy()
    clone.interface.positions[0] += 1.0
    assert not np.allclose(
        clone.interface.positions[0], intf.interface.positions[0]
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_interface_core.py -k "to_ase or to_pymatgen or dict or from_ase or copy" -v`
Expected: FAIL — conversion methods not defined; `copy` still references flat-list skeleton.

- [ ] **Step 3: Write minimal implementation**

Add imports:

```python
from ase.io.jsonio import decode as ase_decode, encode as ase_encode
from pymatgen.io.ase import AseAtomsAdaptor
```

Add methods (and replace the skeleton `copy`):

```python
    def _select(self, part: str) -> Atoms:
        if part == "interface":
            return self.interface
        if part == "substrate":
            return self.substrate
        raise ValueError(f"part must be 'interface' or 'substrate', got {part!r}")

    def to_ase(self, part: str = "interface") -> Atoms:
        """Return a copy of the chosen structure as ASE Atoms."""
        return self._select(part).copy()

    @classmethod
    def from_ase(cls, atoms: Atoms, n_substrate: int, **kwargs) -> "Interface":
        """Build an Interface from a full Atoms, splitting at index n_substrate."""
        substrate = atoms[:n_substrate]
        return cls(substrate=substrate, interface=atoms.copy(), **kwargs)

    def to_pymatgen(self, part: str = "interface") -> "Structure":
        """Return the chosen structure as a pymatgen Structure (escape hatch)."""
        return AseAtomsAdaptor.get_structure(self._select(part))

    @classmethod
    def from_pymatgen(cls, structure, n_substrate: int, **kwargs) -> "Interface":
        """Build an Interface from a pymatgen Structure, splitting at n_substrate."""
        atoms = AseAtomsAdaptor.get_atoms(structure)
        return cls.from_ase(atoms, n_substrate=n_substrate, **kwargs)

    def to_dict(self) -> dict:
        """JSON-serializable dict capturing interface, substrate size, and metadata."""
        return {
            "interface": ase_encode(self.interface),
            "n_substrate": len(self.substrate),
            "fix": list(self.fix),
            "relax": list(self.relax),
            "adsList": self.adsList,
            "clusterList": self.clusterList,
            "split_mol_on_cluster": self.split_mol_on_cluster,
            "cluster_min_size": self.cluster_min_size,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Interface":
        """Reconstruct an Interface produced by to_dict."""
        interface = ase_decode(data["interface"])
        n = data["n_substrate"]
        substrate = interface[:n]
        return cls(
            substrate=substrate,
            interface=interface,
            fixlist=data.get("fix") or None,
            relaxlist=data.get("relax") or None,
            adsList=data.get("adsList"),
            clusterList=data.get("clusterList"),
            split_mol_on_cluster=data.get("split_mol_on_cluster", True),
            cluster_min_size=data.get("cluster_min_size", DEFAULT_CLUSTER_MIN_SIZE),
        )

    def copy(self) -> "Interface":
        """Deep-ish copy: new Atoms objects, same metadata and grouped lists."""
        return Interface(
            substrate=self.substrate.copy(),
            interface=self.interface.copy(),
            fixlist=list(self.fix) or None,
            relaxlist=list(self.relax) or None,
            adsList=[list(g) for g in self.adsList],
            clusterList=[list(g) for g in self.clusterList],
            split_mol_on_cluster=self.split_mol_on_cluster,
            cluster_min_size=self.cluster_min_size,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_interface_core.py -k "to_ase or to_pymatgen or dict or from_ase or copy" -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/gs_gmlip/interface/core.py tests/test_interface_core.py
git commit -m "feat: add conversion API (ase/pymatgen/dict) and copy"
```

---

### Task 11: Store API — `write`/`read`, `write_db`/`from_db_row`

**Files:**
- Modify: `src/gs_gmlip/interface/core.py`
- Test: `tests/test_interface_core.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_interface_core.py`:

```python
def test_write_read_roundtrip(tmp_path):
    slab = _slab()
    full = slab + Atoms("CO", positions=[[2.0, 2.0, 4.0], [2.0, 2.0, 5.13]])
    intf = Interface(substrate=slab, interface=full)
    path = tmp_path / "intf.traj"
    intf.write(str(path))
    loaded = Interface.read(str(path))
    assert len(loaded.interface) == 10
    assert len(loaded.substrate) == 8
    assert loaded.adsList == intf.adsList


def test_db_roundtrip(tmp_path):
    from ase.db import connect

    slab = _slab()
    full = slab + Atoms("CO", positions=[[2.0, 2.0, 4.0], [2.0, 2.0, 5.13]])
    intf = Interface(substrate=slab, interface=full)
    db = connect(str(tmp_path / "test.db"))
    intf.write_db(db, name="trial")
    row = db.get(name="trial")
    loaded = Interface.from_db_row(row)
    assert len(loaded.substrate) == 8
    assert loaded.adsList == intf.adsList
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_interface_core.py -k "write_read or db_roundtrip" -v`
Expected: FAIL — store methods not defined.

- [ ] **Step 3: Write minimal implementation**

Add imports:

```python
from ase.io import read as ase_read, write as ase_write
```

Add methods. The metadata lives in `atoms.info` (for file round-trip) and in the
db row `data=` dict (for ase.db):

```python
    def _metadata(self) -> dict:
        return {
            "n_substrate": len(self.substrate),
            "fix": list(self.fix),
            "relax": list(self.relax),
            "adsList": self.adsList,
            "clusterList": self.clusterList,
            "split_mol_on_cluster": self.split_mol_on_cluster,
            "cluster_min_size": self.cluster_min_size,
        }

    def write(self, path: str, **kwargs) -> None:
        """Write the interface to any ASE-supported format, embedding metadata in info."""
        atoms = self.interface.copy()
        atoms.info["gs_gmlip_interface"] = self._metadata()
        ase_write(path, atoms, **kwargs)

    @classmethod
    def read(cls, path: str, **kwargs) -> "Interface":
        """Read an interface written by Interface.write."""
        atoms = ase_read(path, **kwargs)
        meta = atoms.info.get("gs_gmlip_interface", {})
        n = meta.get("n_substrate", len(atoms))
        return cls(
            substrate=atoms[:n],
            interface=atoms,
            fixlist=meta.get("fix") or None,
            relaxlist=meta.get("relax") or None,
            adsList=meta.get("adsList"),
            clusterList=meta.get("clusterList"),
            split_mol_on_cluster=meta.get("split_mol_on_cluster", True),
            cluster_min_size=meta.get("cluster_min_size", DEFAULT_CLUSTER_MIN_SIZE),
        )

    def write_db(self, db, **kvp):
        """Write to an ase.db connection; metadata stored in the row data dict."""
        return db.write(self.interface, data=self._metadata(), **kvp)

    @classmethod
    def from_db_row(cls, row) -> "Interface":
        """Reconstruct an Interface from an ase.db row written by write_db."""
        atoms = row.toatoms()
        meta = row.data.get("n_substrate") and row.data or {}
        n = meta.get("n_substrate", len(atoms))
        return cls(
            substrate=atoms[:n],
            interface=atoms,
            fixlist=meta.get("fix") or None,
            relaxlist=meta.get("relax") or None,
            adsList=meta.get("adsList"),
            clusterList=meta.get("clusterList"),
            split_mol_on_cluster=meta.get("split_mol_on_cluster", True),
            cluster_min_size=meta.get("cluster_min_size", DEFAULT_CLUSTER_MIN_SIZE),
        )
```

Note: ase.db converts the `data` dict's nested lists to numpy arrays on read.
`adsList`/`clusterList` may come back as arrays — normalize in `from_db_row` by
wrapping with `[[int(i) for i in g] for g in meta[...]]` if the equality test
fails. Adjust the implementation if `test_db_roundtrip` shows a type mismatch.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_interface_core.py -k "write_read or db_roundtrip" -v`
Expected: PASS (2 tests). If `db_roundtrip` fails on list-vs-array equality, apply the normalization noted above and re-run.

- [ ] **Step 5: Commit**

```bash
git add src/gs_gmlip/interface/core.py tests/test_interface_core.py
git commit -m "feat: add store API (write/read, ase.db)"
```

---

### Task 12: Generator hooks — `add_adsorbate`, `remove_group`, `view`

**Files:**
- Modify: `src/gs_gmlip/interface/core.py`
- Test: `tests/test_interface_core.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_interface_core.py`:

```python
def test_add_adsorbate_appends_and_reclassifies():
    intf = Interface(substrate=_slab())
    co = Atoms("CO", positions=[[2.0, 2.0, 4.0], [2.0, 2.0, 5.13]])
    new_idx = intf.add_adsorbate(co)
    assert new_idx == [8, 9]
    assert len(intf.interface) == 10
    assert sorted(intf.adsList[0]) == [8, 9]


def test_add_adsorbate_resets_cache():
    intf = Interface(substrate=_slab())
    _ = intf.bondmatrix
    intf.add_adsorbate(Atoms("O", positions=[[2.0, 2.0, 4.0]]))
    assert "bondmatrix" not in intf.__dict__


def test_remove_group_removes_atoms():
    slab = _slab()
    full = slab + Atoms("CO", positions=[[2.0, 2.0, 4.0], [2.0, 2.0, 5.13]])
    intf = Interface(substrate=slab, interface=full)
    intf.remove_group([8, 9])
    assert len(intf.interface) == 8
    assert intf.adsList == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_interface_core.py -k "add_adsorbate or remove_group" -v`
Expected: FAIL — generator hooks not defined.

- [ ] **Step 3: Write minimal implementation**

Add methods. After mutating `interface`, re-run classification over the appended
region (reuse `_init_ads_cluster` with no provided lists):

```python
    def add_adsorbate(self, mol: Atoms, offset=None) -> list[int]:
        """Append `mol` to the interface, re-classify, reset cache; return new indices.

        Args:
            mol: The adsorbate/cluster atoms to append.
            offset: Optional (3,) translation applied to mol before appending.
        """
        start = len(self.interface)
        addition = mol.copy()
        if offset is not None:
            addition.translate(offset)
        self.interface += addition
        new_indices = list(range(start, len(self.interface)))
        self._init_ads_cluster(None, None)
        self._reset_cache()
        return new_indices

    def remove_group(self, indices: list[int]) -> None:
        """Remove the given interface atoms (must be in the appended region)."""
        n_sub = len(self.substrate)
        if any(i < n_sub for i in indices):
            raise ValueError("cannot remove substrate atoms via remove_group.")
        keep = [i for i in range(len(self.interface)) if i not in set(indices)]
        self.interface = self.interface[keep]
        self._init_ads_cluster(None, None)
        self._reset_cache()

    def view(self):
        """Open the interface in the ASE GUI (thin convenience passthrough)."""
        from ase.visualize import view as ase_view

        return ase_view(self.interface)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_interface_core.py -k "add_adsorbate or remove_group" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/gs_gmlip/interface/core.py tests/test_interface_core.py
git commit -m "feat: add generator hooks (add_adsorbate, remove_group, view)"
```

---

### Task 13: Phase-2 stubs — `find_all_ads_sites`, `find_simple_ads_sites`, `identify_symmetry`

**Files:**
- Modify: `src/gs_gmlip/interface/core.py`
- Test: `tests/test_interface_core.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_interface_core.py`:

```python
def test_phase2_stubs_raise_not_implemented():
    intf = Interface(substrate=_slab())
    with pytest.raises(NotImplementedError):
        intf.find_all_ads_sites()
    with pytest.raises(NotImplementedError):
        intf.find_simple_ads_sites()
    with pytest.raises(NotImplementedError):
        intf.identify_symmetry()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_interface_core.py -k phase2 -v`
Expected: FAIL — stubs currently `...` (return None, no raise).

- [ ] **Step 3: Write minimal implementation**

Add/replace these methods:

```python
    def find_all_ads_sites(self):
        """Find all symmetry-distinct adsorption sites (Phase 2, not yet implemented)."""
        raise NotImplementedError(
            "find_all_ads_sites is planned for Phase 2 (pymatgen AdsorbateSiteFinder)."
        )

    def find_simple_ads_sites(self, height_threshold: float = 2.0):
        """Find simple adsorption sites (Phase 2, not yet implemented)."""
        raise NotImplementedError(
            "find_simple_ads_sites is planned for Phase 2."
        )

    def identify_symmetry(self, method: str = "wood") -> str:
        """Identify surface symmetry / Wood notation (Phase 2, not yet implemented)."""
        raise NotImplementedError(
            "identify_symmetry is planned for Phase 2 (Wood/matrix notation)."
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_interface_core.py -k phase2 -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add src/gs_gmlip/interface/core.py tests/test_interface_core.py
git commit -m "feat: add Phase-2 surface-method stubs (NotImplementedError)"
```

---

### Task 14: Full suite + lint, update docs index

**Files:**
- Modify: `docs/api/` (add `interface.rst`), `docs/index.rst`
- Verify: whole test suite + ruff

- [ ] **Step 1: Create the API doc page**

Create `docs/api/interface.rst`:

```rst
Interface Module
================

.. module:: gs_gmlip.interface

The central structure object: a substrate (clean slab) plus an interface
(slab + adsorbates/clusters), with pymatgen-backed surface analysis and a
conversion/store/generator API.

Interface
---------

.. autoclass:: gs_gmlip.interface.core.Interface
   :members:
   :undoc-members:

Analysis functions
------------------

.. automodule:: gs_gmlip.interface.analysis
   :members:

Builders
--------

.. automodule:: gs_gmlip.interface.builders
   :members:
```

- [ ] **Step 2: Register it in the toctree**

In `docs/index.rst`, add `api/interface` to the API Reference toctree (after `api/structure` or at the top of the API list):

```rst
   api/interface
```

- [ ] **Step 3: Run the full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: ALL PASS (analysis + builders + core; plus existing test_utils).

- [ ] **Step 4: Lint**

Run: `unset VIRTUAL_ENV && uv run ruff check src/gs_gmlip/interface/ tests/test_interface_*.py`
Expected: no errors (fix any reported issues, e.g. import order, unused imports).

- [ ] **Step 5: Commit**

```bash
git add docs/api/interface.rst docs/index.rst
git commit -m "docs: add Interface API reference page"
```

---

## Self-Review Notes

**Spec coverage check:**
- Core model (composition, grouped lists) → Task 6 ✓
- pymatgen dependency + package layout → Task 0 ✓
- Caching (cached_property + reset) → Task 7 ✓
- Classification rule (connectivity + closed-shell split) → Task 2 ✓
- possible_bond (closed-shell heuristic) → Task 3 ✓
- Surface Phase 1 (layers, fix, build) → Task 8 ✓; Phase 2 stubs → Task 13 ✓
- Mutating methods (wrap, align) → Task 9 ✓
- Conversion API (to_ase/pymatgen/dict + part=) → Task 10 ✓
- Store API (write/read, db) → Task 11 ✓
- Generator hooks (add_adsorbate, remove_group, view) → Task 12 ✓
- Tests for all → each task ✓; full-suite + lint + docs → Task 14 ✓

**Type/name consistency:** `classify_appended`, `compute_bondmatrix`,
`compute_possible_bond`, `detect_layers`, `fingerprint`, `_reset_cache`,
`_init_ads_cluster`, `_layer_labels`, `to_ase/to_pymatgen(part=...)` are used
consistently across tasks. `_CACHED` defined in Task 7 lists exactly the four
cached properties.

**Known risk flags (handled inline):**
- pymatgen `SlabGenerator` API arg names can vary by version; Task 5 Step 4 notes
  a fallback if `get_orthogonal_c_slab` fails.
- ase.db nested-list → ndarray coercion; Task 11 notes a normalization fallback.
