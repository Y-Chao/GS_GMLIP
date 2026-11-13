"""Molecule library: reaction intermediates plus user-built molecules, in tiers.

A name is looked up in this order (first hit wins):

1. ``temporary``: molecules kept in memory for the session,
2. ``project``: ``<project>/molecules.json``, shared by that project's jobs,
3. ``global``: ``<gs_home>/molecules.json``, shared by all projects of the user,
4. ``intermediate``: the seed set of electrocatalysis intermediates
   (``gs_gmlip.interface.intermediates``),
5. ``builtin``: the small set of plain adsorbates in ``operations.ADSORBATES``.

Every molecule is stored with its surface-binding atom at the origin and the
rest of the molecule as the user oriented it; placing it on a surface maps the
molecule's +z axis onto the surface normal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from pymatgen.core import Molecule

from gs_gmlip.config import gs_home
from gs_gmlip.interface import operations as ops
from gs_gmlip.interface.intermediates import INTERMEDIATES, get_intermediate

LIBRARY_FILE = "molecules.json"
USER_SCOPES: tuple[str, ...] = ("temporary", "project", "global")
_SEEDS = {item.name: item for item in INTERMEDIATES}


class MoleculeLibrary:
    """Named molecules available as adsorbates and search building blocks.

    Args:
        root: Project directory holding the project tier; ``None`` disables it.
        global_dir: Directory holding the global tier; the user-level
            GS-GMLIP directory by default.
    """

    def __init__(
        self, root: str | Path | None = None, global_dir: str | Path | None = None
    ):
        self._paths: dict[str, Path | None] = {
            "project": Path(root) / LIBRARY_FILE if root is not None else None,
            "global": Path(global_dir if global_dir is not None else gs_home())
            / LIBRARY_FILE,
        }
        self._temporary: dict[str, dict[str, Any]] = {}

    def _stored(self, scope: str) -> dict[str, dict[str, Any]]:
        if scope == "temporary":
            return self._temporary
        path = self._paths[scope]
        if path is None or not path.exists():
            return {}
        return json.loads(path.read_text())

    def _write(self, scope: str, records: dict[str, dict[str, Any]]) -> None:
        path = self._paths[scope]
        if path is None:
            raise ValueError("this library has no project directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=2))

    def _lookup(self, name: str) -> dict[str, Any] | None:
        for scope in USER_SCOPES:
            record = self._stored(scope).get(name)
            if record is not None:
                return record
        return None

    def entries(self) -> list[dict[str, Any]]:
        """All molecules, user tiers first.

        Returns:
            Dicts with ``name``, ``source`` (tier), ``formula``, ``natoms``,
            ``reactions`` (list of reaction tags), and ``note`` (provenance).
        """
        items: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add(
            name: str, source: str, symbols: list[str], reactions: list[str], note: str
        ) -> None:
            if name in seen:
                return
            seen.add(name)
            items.append(
                {
                    "name": name,
                    "source": source,
                    "formula": Atoms(symbols).get_chemical_formula(),
                    "natoms": len(symbols),
                    "reactions": reactions,
                    "note": note,
                }
            )

        for scope in USER_SCOPES:
            for name, record in self._stored(scope).items():
                add(
                    name,
                    scope,
                    record["symbols"],
                    record.get("reactions", []),
                    record.get("note", ""),
                )
        for item in INTERMEDIATES:
            symbols = [str(site.specie) for site in get_intermediate(item.name)]
            add(
                item.name,
                "intermediate",
                symbols,
                list(item.reactions),
                f"{item.description}; {item.note}",
            )
        for name in ops.ADSORBATES:
            symbols = [str(site.specie) for site in ops.get_adsorbate(name)]
            add(name, "builtin", symbols, [], "ASE G2 geometry")
        return items

    def __contains__(self, name: str) -> bool:
        return (
            self._lookup(name) is not None or name in _SEEDS or name in ops.ADSORBATES
        )

    def get(self, name: str) -> Molecule:
        """Return a molecule with its binding atom at the origin.

        Raises:
            KeyError: If no molecule has that name.
        """
        record = self._lookup(name)
        if record is not None:
            return Molecule(record["symbols"], record["positions"])
        if name in _SEEDS:
            return get_intermediate(name)
        if name in ops.ADSORBATES:
            return ops.get_adsorbate(name)
        raise KeyError(f"unknown molecule {name!r}")

    def atoms(self, name: str) -> Atoms:
        """The molecule as ASE ``Atoms`` (for search-method building blocks)."""
        molecule = self.get(name)
        return Atoms(
            [str(site.specie) for site in molecule], positions=molecule.cart_coords
        )

    def add(
        self,
        name: str,
        atoms: Atoms,
        bind_index: int | None = None,
        scope: str = "temporary",
        reactions: list[str] | None = None,
        note: str = "",
    ) -> None:
        """Add or replace a user molecule.

        A user molecule may reuse the name of a seed intermediate or built-in
        adsorbate; it then takes precedence over it.

        Args:
            name: Library name.
            atoms: The molecule, oriented with its "up" direction along +z.
            bind_index: Index of the surface-binding atom; the lowest atom
                along z if omitted.
            scope: ``temporary`` (this session), ``project``, or ``global``.
            reactions: Reaction tags, e.g. ``["CO2RR"]``.
            note: Provenance (method, reference, DOI).

        Raises:
            ValueError: For an empty name, empty molecule, or unknown scope.
        """
        name = name.strip()
        if not name:
            raise ValueError("give the molecule a name")
        if scope not in USER_SCOPES:
            raise ValueError(f"unknown scope {scope!r}; choose from {USER_SCOPES}")
        if len(atoms) == 0:
            raise ValueError("the molecule has no atoms")
        positions = atoms.get_positions()
        if bind_index is None:
            bind_index = int(np.argmin(positions[:, 2]))
        if not 0 <= int(bind_index) < len(atoms):
            raise ValueError("binding atom index out of range")
        record = {
            "symbols": atoms.get_chemical_symbols(),
            "positions": (positions - positions[int(bind_index)]).tolist(),
            "bind_index": int(bind_index),
            "reactions": list(reactions or []),
            "note": note,
        }
        if scope == "temporary":
            self._temporary[name] = record
            return
        records = self._stored(scope)
        records[name] = record
        self._write(scope, records)

    def remove(self, name: str) -> None:
        """Remove a user molecule from every user tier.

        Raises:
            KeyError: If no user molecule has that name.
        """
        found = self._temporary.pop(name, None) is not None
        for scope in ("project", "global"):
            records = self._stored(scope)
            if name in records:
                del records[name]
                self._write(scope, records)
                found = True
        if not found:
            raise KeyError(f"no user molecule named {name!r}")
