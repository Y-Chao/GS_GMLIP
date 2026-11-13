"""Import structures from external ASE databases into a structure store, normalized.

Calculation records from different campaigns describe the same things in
different ways (``xc`` as a key or inside calculator parameters, a cutoff as
``encut`` or ``mode.ecut``, k-points as a string or a list). The importer
reduces each row to the store's schema:

- ``role``: bulk, molecule, surface, or adsorbed (inferred from the geometry
  unless given),
- ``level``: one level-of-theory label, ``code/xc/basis``, e.g.
  ``gpaw/PBE-D3/pw600``; energies are comparable only within a level. The
  functional comes from the calculator settings (a VASP ``IVDW`` adds the
  dispersion suffix); the k-point mesh is kept as a tag, not in the level,
- ``tags``: every original key-value pair, with a few names unified
  (``pot``/``phi`` -> ``potential``, ``q`` -> ``net_charge``), plus
  ``source`` and ``source_uid`` for provenance and de-duplication.

Atoms are reordered metals-first so the substrate/adsorbate partition of
``Interface`` holds; constraints follow the reordering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import ase.db
import numpy as np
from ase import Atoms

from gm2_surf.interface.analysis import is_metal
from gm2_surf.interface.core import Interface
from gm2_surf.store import ROLES, StructureStore

# A cell direction with at least this much empty space (Å) counts as vacuum.
VACUUM_THRESHOLD = 6.0
# Original key -> unified tag name.
TAG_ALIASES = {
    "pot": "potential",
    "phi": "potential",
    "q": "net_charge",
    "encut": "cutoff",
}


def _vacuum_directions(atoms: Atoms) -> int:
    """Number of cell directions with a vacuum gap (0 bulk, 1 slab, 3 molecule/cluster)."""
    if atoms.cell.rank < 3:
        return 3
    scaled = atoms.get_scaled_positions(wrap=True)
    lengths = atoms.cell.cellpar()[:3]
    count = 0
    for axis in range(3):
        values = np.sort(scaled[:, axis])
        gaps = np.diff(np.concatenate([values, [values[0] + 1.0]]))
        if gaps.max() * lengths[axis] > VACUUM_THRESHOLD:
            count += 1
    return count


def infer_role(atoms: Atoms) -> str:
    """Guess the role of a structure from its geometry and composition.

    No vacuum is a bulk; vacuum in one direction is a surface (only metals) or
    an adsorbed system (metals plus anything else); otherwise a structure
    without metals is a molecule and the rest is ``other``.
    """
    metals = [is_metal(symbol) for symbol in atoms.get_chemical_symbols()]
    vacuum = _vacuum_directions(atoms)
    if vacuum == 0:
        return "bulk"
    if vacuum == 1 and any(metals):
        return "surface" if all(metals) else "adsorbed"
    if not any(metals):
        return "molecule"
    return "other"


def _basis_label(key_value_pairs: dict[str, Any], parameters: dict[str, Any]) -> str:
    if "encut" in key_value_pairs:
        return f"pw{key_value_pairs['encut']:g}"
    mode = parameters.get("mode")
    if isinstance(mode, dict):
        if mode.get("name") == "pw" and "ecut" in mode:
            return f"pw{mode['ecut']:g}"
        return str(mode.get("name", ""))
    if isinstance(mode, str):
        spacing = parameters.get("h")
        return f"{mode}-h{spacing:g}" if spacing else mode
    if "encut" in parameters:
        return f"pw{parameters['encut']:g}"
    return ""


def _kpoint_label(key_value_pairs: dict[str, Any], parameters: dict[str, Any]) -> str:
    kpts = key_value_pairs.get("kpts", parameters.get("kpts"))
    if isinstance(kpts, dict):
        kpts = kpts.get("size")
    if isinstance(kpts, str):
        return f"k{kpts}"
    if kpts is not None and np.ndim(kpts) == 1:
        return "k" + "-".join(str(int(k)) for k in kpts)
    return ""


# VASP ``IVDW`` values -> dispersion-correction suffix of the functional label.
_VASP_DISPERSION = {
    1: "D2",
    10: "D2",
    11: "D3",
    12: "D3BJ",
    13: "D4",
    2: "TS",
    20: "TS",
    202: "MBD",
}


# Lower-case functional name -> canonical spelling, so that the same functional
# recorded as "pbe" by one code and "PBE" by another yields one label.
_XC_NAMES = {
    "pbe": "PBE",
    "rpbe": "RPBE",
    "pbesol": "PBEsol",
    "beef-vdw": "BEEF-vdW",
    "scan": "SCAN",
    "rscan": "rSCAN",
    "r2scan": "r2SCAN",
    "pbe0": "PBE0",
    "hse06": "HSE06",
    "lda": "LDA",
}


def canonical_xc(label: str) -> str:
    """Canonical spelling of a functional label, keeping any dispersion suffix (``-D3``, ``-D3BJ``, ...)."""
    import re

    match = re.match(r"^(.*?)(-[dD]\d\w*)?$", label.strip())
    base, suffix = match.group(1), (match.group(2) or "")
    return _XC_NAMES.get(base.lower(), base) + suffix.upper().replace("-D", "-D", 1)


def _functional_label(
    key_value_pairs: dict[str, Any], parameters: dict[str, Any]
) -> tuple[str, str]:
    """Functional label and, if the row's own ``xc`` key disagrees with it, that key.

    The calculator settings are authoritative: a free-text ``xc`` key is kept
    only when it is the calculator's functional plus a dispersion suffix
    (e.g. key ``PBE-D3`` over setting ``PBE``). A VASP ``IVDW`` setting adds
    the matching suffix when the label has none.
    """
    key = str(key_value_pairs.get("xc") or "")
    setting = str(parameters.get("xc") or "")
    if not setting:
        label = key
    elif key.lower().startswith(setting.lower()) and len(key) > len(setting):
        label = key
    else:
        label = setting
    dispersion = _VASP_DISPERSION.get(parameters.get("ivdw"))
    if (
        dispersion
        and "-d" not in label.lower()
        and dispersion.lower() not in label.lower()
    ):
        label = f"{label}-{dispersion}"
    label = canonical_xc(label)
    return label, (key if key and key.lower() != label.lower() else "")


def make_level(code: str, xc: str, basis: str) -> str:
    """Level-of-theory label ``code/xc/basis``.

    The k-point mesh is deliberately not part of it: a converged mesh differs
    between a molecule, a bulk cell, and a slab of the same calculation
    series, and those must share a level to serve as each other's references.
    """
    return "/".join(part for part in (code, xc, basis) if part) if code else ""


def level_from_row(row) -> tuple[str, dict[str, Any]]:
    """Level-of-theory label and normalized settings of an ASE database row.

    Returns:
        ``(level, settings)`` where settings holds ``code``, ``xc``, ``basis``,
        and ``kpoints`` (empty strings where unknown), plus ``xc_key`` when
        the row's own ``xc`` key was overruled by the calculator settings.
        The level is empty if the row has no calculator.
    """
    key_value_pairs = dict(row.key_value_pairs)
    data = dict(row.data) if row.data else {}
    parameters = dict(row.get("calculator_parameters") or data.get("calc_params") or {})
    code = row.get("calculator", "") or ""
    xc, overruled = _functional_label(key_value_pairs, parameters)
    settings = {
        "code": code,
        "xc": xc,
        "basis": _basis_label(key_value_pairs, parameters),
        "kpoints": _kpoint_label(key_value_pairs, parameters),
    }
    if overruled:
        settings["xc_key"] = overruled
    return make_level(code, settings["xc"], settings["basis"]), settings


def geometry_hash(
    atoms: Atoms, energy: float | None = None, labels: dict[str, Any] | None = None
) -> str:
    """Order-independent fingerprint of a structure and its labels.

    Covers species, positions (to 1e-3 Å), the cell, the energy, and the
    row's key-value pairs, so two rows count as copies only when they describe
    the same structure with the same metadata. A geometry that appears again
    under a different label (e.g. as the initial state of another reaction)
    is a distinct record.
    """
    import hashlib

    rows = sorted(
        (symbol, *(round(float(x), 3) + 0.0 for x in position))
        for symbol, position in zip(atoms.get_chemical_symbols(), atoms.get_positions())
    )
    cell = tuple(round(float(x), 3) + 0.0 for x in np.asarray(atoms.cell).ravel())
    tagged = sorted((str(key), str(value)) for key, value in (labels or {}).items())
    payload = repr(
        (rows, cell, None if energy is None else round(float(energy), 6), tagged)
    )
    return hashlib.sha1(payload.encode()).hexdigest()  # noqa: S324 - fingerprint, not security


def to_interface(atoms: Atoms) -> Interface:
    """Build an ``Interface`` from any structure: metals become the substrate.

    Atoms are reordered metals-first (stable within each group); constraints
    and stored energies follow. A structure without metals is all substrate.
    """
    from ase.calculators.singlepoint import SinglePointCalculator

    symbols = atoms.get_chemical_symbols()
    metals = [i for i, symbol in enumerate(symbols) if is_metal(symbol)]
    others = [i for i, symbol in enumerate(symbols) if not is_metal(symbol)]
    order = metals + others if metals and others else list(range(len(atoms)))
    results: dict[str, Any] = {}
    if atoms.calc is not None:
        stored = getattr(atoms.calc, "results", {})
        if "energy" in stored:
            results["energy"] = float(stored["energy"])
        if "forces" in stored:
            results["forces"] = np.asarray(stored["forces"])[order]
    reordered = atoms[order]
    n_substrate = len(metals) if metals and others else len(atoms)
    interface = Interface(substrate=reordered[:n_substrate], interface=reordered)
    if results:
        # Attached last: building the Interface may set a cell, which would
        # invalidate a calculator attached earlier.
        interface.interface.calc = SinglePointCalculator(interface.interface, **results)
    return interface


def import_ase_db(
    source: str | Path,
    store: StructureStore,
    role: str = "auto",
    label_key: str | None = None,
    source_name: str | None = None,
    energy_key: str | None = None,
    code: str | None = None,
) -> dict[str, int]:
    """Import every row of an ASE database into a structure store.

    Rows already imported are skipped, so the import can be repeated safely:
    a row is recognized by its ``unique_id`` or, for copies of a database, by
    a fingerprint of its geometry and energy.

    Args:
        source: Path to the ASE database.
        store: Destination store (a project store or the global store).
        role: Role for all rows, or ``"auto"`` to infer it per structure.
        label_key: Key-value pair to use as the row label (e.g. ``motif``).
        source_name: Provenance name written to the ``source`` tag; the file
            name by default.
        energy_key: Key-value pair holding the total energy (eV) for rows
            whose calculator stored none (e.g. dispersion-wrapper results).
        code: Simulation code to record for all rows, overriding the row's
            calculator name (useful when a wrapper calculator hides the code).

    Returns:
        ``{"imported": n, "skipped": m}``.

    Raises:
        ValueError: For an unknown role.
        FileNotFoundError: If ``source`` does not exist.
    """
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"no such database: {source}")
    if role != "auto" and role not in ROLES:
        raise ValueError(f"unknown role {role!r}; choose 'auto' or one of {ROLES}")
    name = source_name or source.name
    existing = store.records(stage="imported")
    known = {rec.tags.get("source_uid") for rec in existing}
    fingerprints = {rec.tags.get("geometry_hash") for rec in existing} - {None}
    counts = {"imported": 0, "skipped": 0}
    with ase.db.connect(str(source)) as database:
        rows = list(database.select())
    for row in rows:
        if row.unique_id in known:
            counts["skipped"] += 1
            continue
        atoms = row.toatoms()
        if (
            energy_key
            and row.get("energy") is None
            and energy_key in row.key_value_pairs
        ):
            from ase.calculators.singlepoint import SinglePointCalculator

            atoms.calc = SinglePointCalculator(
                atoms, energy=float(row.key_value_pairs[energy_key])
            )
        fingerprint = geometry_hash(
            atoms,
            atoms.get_potential_energy() if atoms.calc is not None else None,
            dict(row.key_value_pairs),
        )
        if fingerprint in fingerprints:
            counts["skipped"] += 1
            continue
        fingerprints.add(fingerprint)
        level, settings = level_from_row(row)
        if code:
            settings["wrapper"] = settings["code"] if settings["code"] != code else ""
            settings["code"] = code
            level = make_level(code, settings["xc"], settings["basis"])
        tags: dict[str, Any] = {
            "source": name,
            "source_uid": row.unique_id,
            "geometry_hash": fingerprint,
            **settings,
        }
        for key, value in row.key_value_pairs.items():
            if isinstance(value, np.generic):
                value = value.item()
            name_in_tags = TAG_ALIASES.get(key, key)
            # Normalized settings win; the row's own spelling is kept in ``xc_key``.
            if name_in_tags not in settings:
                tags[name_in_tags] = value
        store.add(
            to_interface(atoms),
            stage="imported",
            method=settings["code"],
            role=infer_role(atoms) if role == "auto" else role,
            level=level,
            label=str(row.key_value_pairs.get(label_key, "")) if label_key else "",
            tags=tags,
        )
        counts["imported"] += 1
    return counts
