"""Structure store: every built, relaxed, sampled, or searched structure lands here.

``StructureStore`` wraps an ``ase.db`` SQLite file and adds provenance, so the
database is a directed graph of structures: each row knows which row it was
derived from (``parent_id``), which job produced it (``job_id``), and at what
stage of the workflow (``stage``). ``Project`` bundles a store with the jobs
directory that sits next to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import ase.db
from ase.calculators.singlepoint import SinglePointCalculator

from gs_gmlip.interface.core import Interface

# Workflow stages written by the built-in tasks and the builder.
STAGES: tuple[str, ...] = (
    "built",
    "singlepoint",
    "relax_step",
    "relaxed",
    "md",
    "ga",
    "gcmc",
)


@dataclass(frozen=True)
class StructureRecord:
    """Lightweight summary of one stored structure (no coordinates).

    Attributes:
        id: Row id in the database.
        formula: Chemical formula of the full interface.
        natoms: Number of atoms.
        stage: Workflow stage, see ``STAGES``.
        method: Method that produced it (e.g. ``"BFGS"``, ``"GA"``).
        calculator: Registered calculator name, empty if none.
        energy: Potential energy in eV, ``None`` if not evaluated.
        fmax: Largest force norm in eV/Å, ``None`` if not evaluated.
        parent_id: Row id this structure was derived from, if any.
        job_id: Job that produced it, empty for hand-built structures.
        step: Step index within the job (optimizer step, MD step, GA step).
        label: Free-text label.
        ctime: Creation time in ase.db units (years since 2000).
    """

    id: int
    formula: str
    natoms: int
    stage: str
    method: str
    calculator: str
    energy: float | None
    fmax: float | None
    parent_id: int | None
    job_id: str
    step: int | None
    label: str
    ctime: float


class StructureStore:
    """Provenance-aware structure database backed by ``ase.db``.

    Connections are opened per operation, so the store is safe to share
    between the app process and worker processes writing concurrently.

    Args:
        path: Path to the SQLite file (created on first write).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self):
        return ase.db.connect(str(self.path))

    def add(
        self,
        interface: Interface,
        *,
        stage: str,
        method: str = "",
        calculator: str = "",
        parent_id: int | None = None,
        job_id: str = "",
        step: int | None = None,
        label: str = "",
    ) -> int:
        """Write a structure and return its row id.

        Energy and forces are frozen from the attached calculator (if any) into
        the row, so reading the row back never needs the original calculator.

        Args:
            interface: Structure to store.
            stage: Workflow stage, see ``STAGES``.
            method: Method that produced the structure.
            calculator: Registered calculator name.
            parent_id: Row id of the structure this one was derived from.
            job_id: Id of the producing job.
            step: Step index within the job.
            label: Free-text label.
        """
        source = interface.interface
        atoms = source.copy()
        if source.calc is not None:
            try:
                atoms.calc = SinglePointCalculator(
                    atoms,
                    energy=float(source.get_potential_energy()),
                    forces=source.get_forces(apply_constraint=False),
                )
            except Exception:  # noqa: BLE001 - a structure without results is still worth storing
                atoms.calc = None

        key_value_pairs: dict[str, str | int] = {"stage": stage}
        if method:
            key_value_pairs["method"] = method
        if calculator:
            key_value_pairs["calc_name"] = calculator
        if parent_id is not None:
            key_value_pairs["parent_id"] = int(parent_id)
        if job_id:
            key_value_pairs["job_id"] = job_id
        if step is not None:
            key_value_pairs["step"] = int(step)

        data = interface._metadata()
        data["label"] = label
        with self._connect() as db:
            return int(db.write(atoms, key_value_pairs=key_value_pairs, data=data))

    def get(self, structure_id: int) -> Interface:
        """Load a structure. Its energy and forces, if stored, are attached."""
        with self._connect() as db:
            return Interface.from_db_row(db.get(id=int(structure_id)))

    @staticmethod
    def _to_record(row) -> StructureRecord:
        data = dict(row.data) if row.data else {}
        return StructureRecord(
            id=int(row.id),
            formula=row.formula,
            natoms=int(row.natoms),
            stage=row.get("stage", ""),
            method=row.get("method", ""),
            calculator=row.get("calc_name", ""),
            energy=row.get("energy"),
            fmax=row.get("fmax"),
            parent_id=row.get("parent_id"),
            job_id=row.get("job_id", ""),
            step=row.get("step"),
            label=str(data.get("label", "")),
            ctime=float(row.ctime),
        )

    def record(self, structure_id: int) -> StructureRecord:
        """Return the summary of one structure."""
        with self._connect() as db:
            return self._to_record(db.get(id=int(structure_id)))

    def records(
        self,
        *,
        stage: str | None = None,
        job_id: str | None = None,
        parent_id: int | None = None,
        limit: int | None = None,
    ) -> list[StructureRecord]:
        """List structure summaries in ascending id order, optionally filtered.

        Args:
            stage: Keep only this workflow stage.
            job_id: Keep only structures produced by this job.
            parent_id: Keep only direct children of this structure.
            limit: Keep only the ``limit`` most recent matches.
        """
        if not self.path.exists():
            return []
        filters: dict[str, str | int] = {}
        if stage is not None:
            filters["stage"] = stage
        if job_id is not None:
            filters["job_id"] = job_id
        if parent_id is not None:
            filters["parent_id"] = int(parent_id)
        with self._connect() as db:
            found = [self._to_record(row) for row in db.select(**filters)]
        found.sort(key=lambda rec: rec.id)
        return found[-limit:] if limit else found

    def lineage(self, structure_id: int) -> list[StructureRecord]:
        """Ancestry of a structure, root first, ending with the structure itself."""
        chain: list[StructureRecord] = []
        seen: set[int] = set()
        current: int | None = int(structure_id)
        while current is not None and current not in seen:
            seen.add(current)
            rec = self.record(current)
            chain.append(rec)
            current = rec.parent_id
        return chain[::-1]

    def delete(self, structure_ids: Iterable[int]) -> None:
        """Delete rows. Children keep their (now dangling) ``parent_id``."""
        ids = [int(i) for i in structure_ids]
        if ids:
            with self._connect() as db:
                db.delete(ids)

    def __len__(self) -> int:
        if not self.path.exists():
            return 0
        with self._connect() as db:
            return db.count()


class Project:
    """A working directory: one structure database plus its jobs.

    Layout::

        <root>/structures.db
        <root>/jobs/<job_id>/{job.json, status.json, log.txt}

    Args:
        root: Project directory (created if missing).
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.jobs_dir = self.root / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.store = StructureStore(self.root / "structures.db")
