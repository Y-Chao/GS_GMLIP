"""SQLite database for candidate structure tracking.

Built on ASE's ase.db SQLite backend, providing candidate tracking
with relaxation status, origin metadata, generation numbers, and
key-value pair storage. Refactored from ase_ga_official/data.py.
"""

from __future__ import annotations

import logging
import os

import ase.db
from ase import Atoms

logger = logging.getLogger(__name__)


def _get_raw_score(atoms: Atoms) -> float:
    """Extract raw_score from atoms info."""
    return atoms.info["key_value_pairs"]["raw_score"]


def _test_raw_score(atoms: Atoms):
    """Assert raw_score is present."""
    assert "raw_score" in atoms.info.get(
        "key_value_pairs", {}
    ), "raw_score not in atoms.info['key_value_pairs']"


class CandidateDatabase:
    """Database for storing and retrieving candidate structures.

    Wraps ASE ase.db SQLite database with candidate lifecycle management:
    unrelaxed -> queued -> relaxed, with generation tracking and
    participation history for GA fitness.

    Parameters
    ----------
    db_file : str
        Path to the SQLite database file.
    """

    def __init__(self, db_file: str):
        if not os.path.isfile(db_file):
            raise FileNotFoundError(f"Database file {db_file} not found")
        self.db_file = db_file
        self.db = ase.db.connect(db_file)
        self._returned_ids: set[int] = set()

    # --- Unrelaxed candidates ---

    def get_unrelaxed_candidate(self) -> Atoms | None:
        """Return one unrelaxed candidate ready for evaluation."""
        ids = self._get_unrelaxed_ids()
        if not ids:
            return None
        atoms = self._get_latest_for_confid(ids[0])
        atoms.info["confid"] = ids[0]
        atoms.info.setdefault("data", {})
        return atoms

    def get_all_unrelaxed(self) -> list[Atoms]:
        """Return all unrelaxed candidates."""
        ids = self._get_unrelaxed_ids()
        results = []
        for confid in ids:
            atoms = self._get_latest_for_confid(confid)
            atoms.info["confid"] = confid
            atoms.info.setdefault("data", {})
            results.append(atoms)
        return results

    def _get_unrelaxed_ids(self) -> list[int]:
        """Get IDs of candidates that are unrelaxed and not queued."""
        unrelaxed = {r.gaid for r in self.db.select(relaxed=0)}
        relaxed = {r.gaid for r in self.db.select(relaxed=1)}
        queued = {r.gaid for r in self.db.select(queued=1)}
        return [
            gaid for gaid in unrelaxed if gaid not in relaxed and gaid not in queued
        ]

    def _get_latest_for_confid(self, confid: int) -> Atoms:
        """Get the latest version of a candidate."""
        entries = list(self.db.select(gaid=confid))
        entries.sort(key=lambda x: x.mtime)
        return self.db.get_atoms(entries[-1].id, add_additional_information=True)

    # --- Adding candidates ---

    def add_unrelaxed_candidate(self, atoms: Atoms, description: str) -> int:
        """Add a new unrelaxed candidate to the database.

        Parameters
        ----------
        atoms : Atoms
            Candidate structure.
        description : str
            Format "type:description", e.g. "pairing:CutSplicePairing".

        Returns
        -------
        confid : int
            Configuration ID assigned.
        """
        t, desc = description.split(":", 1)
        kwargs = {"relaxed": 0, "extinct": 0, t: 1, "description": desc}

        kvp = atoms.info.get("key_value_pairs", {})
        data = atoms.info.get("data", {})

        if "generation" not in kvp:
            kwargs["generation"] = self.get_generation_number()

        gaid = self.db.write(atoms, key_value_pairs=kvp, data=data, **kwargs)
        self.db.update(gaid, gaid=gaid)
        atoms.info["confid"] = gaid
        return gaid

    def add_relaxed_candidate(self, atoms: Atoms) -> int:
        """Add a relaxed candidate (not previously in DB as unrelaxed).

        Returns
        -------
        relax_id : int
        """
        _test_raw_score(atoms)
        kvp = atoms.info.get("key_value_pairs", {})
        data = atoms.info.get("data", {})

        if "generation" not in kvp:
            kvp["generation"] = self.get_generation_number()

        relax_id = self.db.write(atoms, relaxed=1, key_value_pairs=kvp, data=data)
        self.db.update(relax_id, gaid=relax_id)
        atoms.info["confid"] = relax_id
        atoms.info["relax_id"] = relax_id
        return relax_id

    def add_relaxed_step(self, atoms: Atoms) -> int:
        """Mark an existing unrelaxed candidate as relaxed.

        The candidate must already have confid set via add_unrelaxed_candidate.
        """
        _test_raw_score(atoms)
        gaid = atoms.info["confid"]
        kvp = atoms.info.get("key_value_pairs", {})
        data = atoms.info.get("data", {})

        if "generation" not in kvp:
            kvp["generation"] = self.get_generation_number()

        relax_id = self.db.write(
            atoms, relaxed=1, gaid=gaid, key_value_pairs=kvp, data=data
        )
        atoms.info["relax_id"] = relax_id
        return relax_id

    # --- Querying ---

    def get_all_relaxed(
        self, only_new: bool = False, use_extinct: bool = False
    ) -> list[Atoms]:
        """Return all relaxed candidates.

        Parameters
        ----------
        only_new : bool
            Only return candidates not previously returned.
        use_extinct : bool
            If True, exclude extinct candidates.
        """
        if use_extinct:
            entries = self.db.select("relaxed=1,extinct=0", sort="-raw_score")
        else:
            entries = self.db.select("relaxed=1", sort="-raw_score")

        results = []
        for entry in entries:
            if only_new and entry.gaid in self._returned_ids:
                continue
            atoms = self.db.get_atoms(entry.id, add_additional_information=True)
            atoms.info["confid"] = entry.gaid
            atoms.info["relax_id"] = entry.id
            results.append(atoms)
            self._returned_ids.add(entry.gaid)
        return results

    def get_slab(self) -> Atoms:
        """Get the simulation cell / slab."""
        return self.db.get_atoms(simulation_cell=True)

    def get_generation_number(self, pop_size: int | None = None) -> int:
        """Return the current generation number."""
        if pop_size is None:
            pop_size = self.get_param("population_size")
        if pop_size is None:
            return 0

        all_relaxed = list(self.db.select(relaxed=1))
        g = 0
        while True:
            count = len([c for c in all_relaxed if c.get("generation") == g])
            if count >= pop_size:
                g += 1
            else:
                return g

    def get_param(self, key: str):
        """Get a parameter stored during database initialization."""
        try:
            entry = self.db.get(1)
            if entry.get("data"):
                return entry.data.get(key)
        except Exception:
            pass
        return None

    def get_participation(self) -> tuple[dict[int, int], list[tuple[int, int]]]:
        """Get pairing participation data for fitness calculation.

        Returns
        -------
        frequency : dict
            {confid: n_times_used_as_parent}
        pairs : list of tuple
            List of (id1, id2) pairs that have been made.
        """
        frequency: dict[int, int] = {}
        pairs: list[tuple[int, int]] = []
        for entry in self.db.select(pairing=1):
            c1, c2 = entry.data["parents"]
            pairs.append(tuple(sorted([c1, c2])))
            frequency[c1] = frequency.get(c1, 0) + 1
            frequency[c2] = frequency.get(c2, 0) + 1
        return frequency, pairs

    def mark_as_queued(self, atoms: Atoms):
        """Mark a candidate as queued for evaluation."""
        gaid = atoms.info["confid"]
        self.db.write(
            None,
            gaid=gaid,
            queued=1,
            key_value_pairs=atoms.info.get("key_value_pairs", {}),
        )

    def kill_candidate(self, confid: int):
        """Mark a candidate as extinct."""
        for entry in self.db.select(gaid=confid):
            self.db.update(entry.id, extinct=1)

    @property
    def n_relaxed(self) -> int:
        return len(list(self.db.select(relaxed=1)))

    @property
    def n_unrelaxed(self) -> int:
        return len(self._get_unrelaxed_ids())


class PrepareDatabase:
    """Initialize a new candidate database.

    Parameters
    ----------
    db_file : str
        Path for the new database file.
    slab : Atoms, optional
        The slab / simulation cell.
    **kwargs
        Additional parameters stored as data (e.g., population_size).
    """

    def __init__(self, db_file: str, slab: Atoms | None = None, **kwargs):
        if os.path.exists(db_file):
            raise FileExistsError(f"Database file {db_file} already exists")
        self.db_file = db_file
        if slab is None:
            slab = Atoms()
        self.db = ase.db.connect(db_file)
        self.db.write(slab, data=dict(kwargs), simulation_cell=True)
        logger.info(f"Created database: {db_file}")

    def add_unrelaxed_candidate(self, atoms: Atoms, **kwargs) -> int:
        """Add an unrelaxed starting candidate."""
        gaid = self.db.write(
            atoms,
            origin="StartingCandidateUnrelaxed",
            relaxed=0,
            generation=0,
            extinct=0,
            **kwargs,
        )
        self.db.update(gaid, gaid=gaid)
        atoms.info["confid"] = gaid
        return gaid

    def add_relaxed_candidate(self, atoms: Atoms, **kwargs) -> int:
        """Add an already-relaxed starting candidate."""
        _test_raw_score(atoms)
        data = atoms.info.get("data", {})
        kvp = atoms.info.get("key_value_pairs", {})
        gaid = self.db.write(
            atoms,
            origin="StartingCandidateRelaxed",
            relaxed=1,
            generation=0,
            extinct=0,
            key_value_pairs=kvp,
            data=data,
            **kwargs,
        )
        self.db.update(gaid, gaid=gaid)
        atoms.info["confid"] = gaid
        return gaid
