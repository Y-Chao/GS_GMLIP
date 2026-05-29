# Interface Class — Design Spec

**Date:** 2026-05-29
**Module:** `src/gs_gmlip/interface/`
**Status:** Approved design, ready for implementation plan

## Goal

Complete the `Interface` class — the central structure object for `gs_gmlip`.
It represents a catalyst **substrate** (clean slab) plus an **interface** (slab +
adsorbates/clusters), tracks fixed/relaxed regions, classifies adsorbates vs
clusters, performs pymatgen-backed surface analysis, and exposes a
conversion/store/generator API so other modules (generator, database, viewer)
can build, persist, and consume interfaces.

## Context

The codebase is currently stripped to a minimal core: `utils.py`,
`generator/geometry.py`, and the `interface.py` skeleton. The structure/search/
data modules that existed at v1.1.0 were removed. `Interface` is therefore the
fresh central structure object. `pymatgen` is **not** yet a dependency.

## Decisions

### Core model — composition (wrapper)
`Interface` *has* two ASE `Atoms` attributes; it does not subclass `Atoms`.

- `substrate: Atoms` — clean slab, indices `0..N-1`.
- `interface: Atoms` — full structure: substrate prefix `0..N-1` then appended
  adsorbate/cluster atoms `N..M`.
- `fix: list[int]` / `relax: list[int]` — index into the substrate region.
- `adsList: list[list[int]]` / `clusterList: list[list[int]]` — **grouped**
  index lists (each inner list = one connected molecule/cluster), indexing into
  `interface`. (Changed from the skeleton's flat lists.)

### Dependencies
`pymatgen>=2024.0.0` added as a **required** dependency (surface analysis is the
point of this class). `networkx` is used for connected components (already a
pymatgen dependency). ASE↔pymatgen conversion via
`pymatgen.io.ase.AseAtomsAdaptor`.

### Package layout
`interface.py` becomes a package. Public import path is unchanged
(`from gs_gmlip.interface import Interface`).

```
src/gs_gmlip/interface/
├── __init__.py      # re-exports Interface
├── core.py          # Interface class: state, __init__, caching, mutation, I/O, generator hooks
├── analysis.py      # pure functions: detect_layers, build_neighbor_graph, classify_appended,
│                    #                 compute_bondmatrix, compute_possible_bond, fingerprint
└── builders.py      # pure functions: slab_from_bulk, primitive_slab_from_bulk
```

Rationale: the pymatgen-heavy math is *pure* (Atoms in → data out), so it lives
as free functions that can be unit-tested without constructing an `Interface`.
The class stays small: state + caching + mutation + thin delegation.

### Caching — `cached_property` + manual reset
Expensive derived data are `functools.cached_property`. Mutating methods call
`_reset_cache()` (pops cached entries from `__dict__`).

Cached: `bondmatrix`, `possible_bond`, `fingerprint`, `_substrate_layers`.

This replaces the skeleton's fragile `x()` / `get_x()` pairs (where `x()` would
crash if `get_x()` had not run). The public name becomes the `cached_property`.

### Adsorbate / cluster classification (final rule)
Input: appended region indices `N..M`. Kwarg `split_mol_on_cluster: bool = True`.

1. Build neighbor graph over appended atoms (covalent-radius cutoff via pymatgen).
2. Connected components = candidate groups.
3. Per component:
   - No metal → **adsorbate** group.
   - Metal present and `split=False` → whole component = **cluster**.
   - Metal present and `split=True` → split:
     - **Adsorbate** = each maximal connected non-metal fragment that has **≥1
       internal non-metal–non-metal bond** (i.e. fragment size ≥2).
     - **Cluster** = all metal atoms + every non-metal atom not in a qualifying
       adsorbate fragment (lone non-metals bonded only to metals, e.g. atomic O
       or bridging O, join the cluster).

"Metal" = `pymatgen.core.Element(symbol).is_metal` (metalloids → False).

Consequences (sanity checks):
- CO on Pt₄ → Pt₄ = cluster; C–O (1 internal bond) = adsorbate.
- Atomic O on Pt₄ → lone non-metal → cluster.
- Free H₂O (no metal in component) → adsorbate.
- Bare Pt₄ → cluster.

`cluster_min_size: int` kwarg available for components that should only count as
clusters above a size threshold. A single internal `_classify_appended()` does
the work; `get_ads_list()` / `get_cluster_list()` are thin public wrappers
(signatures preserved for `__init__`).

### possible_bond — valence capacity (closed-shell heuristic)
`possible_bond: dict[int, int]` — atom index → number of *additional* bonds it
can still form, for the generator to decide attachment sites.

```python
SHELLS = np.array([2, 10, 18, 36, 54, 86])
def expected_bonds(Z: int) -> int:
    return int(np.min(np.abs(Z - SHELLS)))   # distance to nearest closed shell

possible_bond[i] = max(0, expected_bonds(Z_i) - current_coordination_i)
```

`current_coordination_i` comes from `bondmatrix`. Examples: H→1, O→2, N→3, C→4,
Pt→8. The closed-shell rationale is non-obvious → documented in a code comment.

### Surface analysis — phased
**Phase 1 (implement now):**
- `get_substrate_layers() -> int` — z-coordinate clustering of substrate atoms
  (tolerance kwarg). Backed by `cached_property _substrate_layers`.
- `fix_substrate(layers: int | float, symmetry=None)` — `int` = fix N bottom
  layers; `float` = fix below that z-height. Sets `self.fix`, applies `FixAtoms`
  to substrate & interface, resets cache.
- `build_from_bulk(bulk, miller_index, layer=4, vacuum=15.0, symmetry=False)` —
  pymatgen `SlabGenerator` → assigns `substrate`/`interface`.
- `build_primitive_surface_from_bulk(...)` — same, primitive-reduced.

**Phase 2 (documented stubs raising `NotImplementedError`):**
`find_all_ads_sites`, `find_simple_ads_sites`, `identify_symmetry` (Wood/matrix
notation — conventions to be pinned down later).

### Mutating methods (all reset cache)
- `wrap()` — ASE wrap into cell.
- `align_interface(loc)` — z-shift appended region to `center`/`bottom` of
  substrate; `center_interface()` / `bottom_interface()` delegate.
- `fix_substrate(...)` — see above.
- `add_adsorbate(...)` / `remove_group(...)` — generator hooks (below).

### API surface (build now)

**Conversion (pure, lossless):**
- `to_ase(part="interface"|"substrate") -> Atoms` (copy).
- `from_ase(atoms, n_substrate, **kw) -> Interface` (classmethod).
- `to_pymatgen(part="interface"|"substrate") -> Structure` (via AseAtomsAdaptor).
- `from_pymatgen(struct, n_substrate) -> Interface` (classmethod).
- `to_dict() -> dict` / `from_dict(d) -> Interface` — JSON-serializable: interface
  atoms (positions/numbers/cell/pbc) + `n_substrate` + fix/relax/adsList/
  clusterList + kwargs (`cluster_min_size`, `split_mol_on_cluster`,
  `expected_coordination` override if any).

`to_pymatgen()` is the documented escape hatch for ad-hoc pymatgen analysis not
wrapped on the class. Default returns the full interface as a `Structure`;
`part="substrate"` returns the clean slab.

**Store (built on conversion):**
- `write(path, **kw)` — ASE-format write of `interface`; metadata (n_substrate,
  fix/relax/grouped lists) stashed in `atoms.info` for round-trip.
- `read(path) -> Interface` (classmethod) — read + reconstruct from `atoms.info`.
- `write_db(db, **kvp)` / `from_db_row(row)` — `ase.db` row with metadata in `data=`.

**Generator hooks (mutating, reset cache):**
- `add_adsorbate(mol: Atoms, anchor=None, offset=None) -> list[int]` — append
  molecule to `interface`, return new indices, re-classify.
- `remove_group(indices: list[int])` — remove group, reshift indices, re-classify.
- `view()` — thin `ase.visualize.view(self.interface)` passthrough (only view
  convenience kept; plotting deferred).

`copy()` updated to pass through new kwargs and grouped lists.

## Out of scope (deferred)
- Plotting / notebook `_repr_html_` / site-overlay figures.
- `find_all_ads_sites`, `find_simple_ads_sites`, `identify_symmetry` (Phase 2).
- pymatgen `Molecule` output for isolated adsorbates (small follow-on helper if
  needed).

## Testing strategy
- **analysis.py / builders.py** unit-tested as pure functions: feed crafted ASE
  Atoms (CO on Pt₄, atomic O on Pt₄, free H₂O, bare Pt₄, multilayer slab) and
  assert classification, layer count, possible_bond, slab generation.
- **core.py**: construction/validation, cache invalidation after each mutating
  method, round-trip `to_dict`/`from_dict` and `write`/`read`, `copy()`
  equivalence, `to_ase`/`to_pymatgen` part selection.
- Phase-2 stubs assert `NotImplementedError`.

## Integration points
- Searchers (future) consume `Interface.interface` (full Atoms with FixAtoms
  constraints) via `to_ase()`.
- Database (future) persists via `to_dict()` / `write_db()`.
- Generator (future) builds via `build_from_bulk` + `add_adsorbate`, using
  `possible_bond` for attachment decisions.
