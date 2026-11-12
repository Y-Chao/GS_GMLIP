"""Light, text-based reference structures: JSON Lines in, JSON Lines out.

One structure per line, human-readable and easy to convert::

    {"formula": "Cu4", "role": "bulk", "level": "vasp/r2SCAN/pw450",
     "energy": -59.4, "symbols": ["Cu", ...], "positions": [[...], ...],
     "cell": [[...], ...], "pbc": [true, true, true], "label": "",
     "tags": {"code": "vasp", "xc": "r2SCAN", "basis": "pw450", "kpts": "8-8-8"}}

Energies are in eV and lengths in Å. The package ships such files in
``gm2_surf/database/`` (relaxed bulk lattices at several levels of theory and
gas-phase molecules); they are the read-only "built-in" tier that every
project can load structures from and take reference energies from, after the
project store and the global store.

``write_jsonl`` turns rows of any ``StructureStore`` into this form and
``import_jsonl`` turns a file back into store rows; the command line has
``gm2_surf export`` and ``gm2_surf import`` for both directions.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from gm2_surf.store import StructureRecord

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gm2_surf.interface import Interface
    from gm2_surf.store import StructureStore

DATABASE_DIR = Path(__file__).parent / "database"
REFERENCE_STAGE = "imported"
# Tags worth keeping in a shared file; origin paths and hashes are dropped.
KEPT_TAGS = (
    "code",
    "xc",
    "basis",
    "cutoff",
    "kpts",
    "kpoints",
    "potential",
    "net_charge",
)
_DECIMALS = 6


def to_json(record: StructureRecord, atoms: Atoms) -> dict[str, Any]:
    """One structure as a JSON-serializable dict (see the module docstring)."""
    moments = atoms.get_initial_magnetic_moments()
    data: dict[str, Any] = {
        "formula": record.formula,
        "role": record.role,
        "level": record.level,
        "label": record.label,
        "energy": None if record.energy is None else float(record.energy),
        "symbols": atoms.get_chemical_symbols(),
        "positions": [
            [round(float(x), _DECIMALS) for x in row] for row in atoms.positions
        ],
        "cell": [[round(float(x), _DECIMALS) for x in row] for row in atoms.cell.array],
        "pbc": [bool(flag) for flag in atoms.pbc],
        "tags": {
            key: record.tags[key]
            for key in KEPT_TAGS
            if record.tags.get(key) not in (None, "")
        },
    }
    if moments.any():
        data["magmoms"] = [round(float(m), 3) for m in moments]
    return data


def from_json(data: dict[str, Any]) -> Atoms:
    """The structure of one JSON line, with its energy attached if it has one."""
    atoms = Atoms(
        symbols=data["symbols"],
        positions=data["positions"],
        cell=data["cell"],
        pbc=data["pbc"],
    )
    if "magmoms" in data:
        atoms.set_initial_magnetic_moments(data["magmoms"])
    if data.get("energy") is not None:
        atoms.calc = SinglePointCalculator(atoms, energy=float(data["energy"]))
    return atoms


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """All structures of a JSON Lines file (blank lines are skipped)."""
    lines = Path(path).read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def write_jsonl(
    store: StructureStore,
    path: str | Path,
    *,
    roles: Iterable[str] | None = None,
    structure_ids: Iterable[int] | None = None,
) -> int:
    """Write rows of a store to a JSON Lines file and return how many were written.

    Args:
        store: Store to read.
        path: File to write.
        roles: Keep only these roles (all by default).
        structure_ids: Keep only these row ids (all by default).
    """
    wanted_roles = None if roles is None else set(roles)
    wanted_ids = None if structure_ids is None else {int(i) for i in structure_ids}
    lines = []
    for record in store.records():
        if wanted_roles is not None and record.role not in wanted_roles:
            continue
        if wanted_ids is not None and record.id not in wanted_ids:
            continue
        lines.append(json.dumps(to_json(record, store.get(record.id).interface)))
    Path(path).write_text("".join(f"{line}\n" for line in lines))
    return len(lines)


def import_jsonl(path: str | Path, store: StructureStore) -> int:
    """Add every structure of a JSON Lines file to a store; returns the number added."""
    from gm2_surf.importers import to_interface

    entries = read_jsonl(path)
    for data in entries:
        store.add(
            to_interface(from_json(data)),
            stage=REFERENCE_STAGE,
            method=str(data.get("tags", {}).get("code", "")),
            label=data.get("label", ""),
            role=data.get("role", ""),
            level=data.get("level", ""),
            tags={**data.get("tags", {}), "source": Path(path).name},
        )
    return len(entries)


class ReferenceLibrary:
    """Read-only structures from JSON Lines files, with the reading part of ``StructureStore``.

    Rows are numbered from 1 in file order, so ids are stable as long as the
    files are only appended to.

    Args:
        paths: JSON Lines files, read in the given order.
    """

    def __init__(self, paths: Iterable[str | Path]):
        self._entries: list[dict[str, Any]] = []
        for path in paths:
            self._entries.extend(read_jsonl(path))

    def __len__(self) -> int:
        return len(self._entries)

    def _entry(self, structure_id: int) -> dict[str, Any]:
        index = int(structure_id) - 1
        if not 0 <= index < len(self._entries):
            raise KeyError(f"no built-in reference #{structure_id}")
        return self._entries[index]

    def record(self, structure_id: int) -> StructureRecord:
        """Summary of one structure."""
        data = self._entry(structure_id)
        tags = dict(data.get("tags", {}))
        return StructureRecord(
            id=int(structure_id),
            formula=data["formula"],
            natoms=len(data["symbols"]),
            stage=REFERENCE_STAGE,
            method=str(tags.get("code", "")),
            calculator="",
            energy=data.get("energy"),
            fmax=None,
            score=None,
            parent_id=None,
            job_id="",
            step=None,
            label=data.get("label", ""),
            ctime=0.0,
            role=data.get("role", ""),
            level=data.get("level", ""),
            tags=tags,
        )

    def records(
        self,
        *,
        stage: str | None = None,
        job_id: str | None = None,
        parent_id: int | None = None,
        limit: int | None = None,
        role: str | None = None,
        level: str | None = None,
    ) -> list[StructureRecord]:
        """Structure summaries in id order, filtered like ``StructureStore.records``."""
        if job_id is not None or parent_id is not None:
            return []  # built-in references belong to no job and have no lineage
        found = [self.record(index + 1) for index in range(len(self._entries))]
        if stage is not None:
            found = [rec for rec in found if rec.stage == stage]
        if role is not None:
            found = [rec for rec in found if rec.role == role]
        if level is not None:
            found = [rec for rec in found if rec.level == level]
        return found[-limit:] if limit else found

    def get(self, structure_id: int) -> Interface:
        """Load a structure. Its energy, if stored, is attached."""
        from gm2_surf.importers import to_interface

        return to_interface(from_json(self._entry(structure_id)))


@lru_cache(maxsize=1)
def builtin_references() -> ReferenceLibrary:
    """The reference structures shipped with the package (``gm2_surf/database/*.jsonl``)."""
    return ReferenceLibrary(sorted(DATABASE_DIR.glob("*.jsonl")))
