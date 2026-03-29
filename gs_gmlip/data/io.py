"""Data I/O utilities for reading/writing structures in various formats."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import read, write


def read_structures(
    filename: str | Path, index: str = ":", format: str | None = None
) -> list[Atoms]:
    """Read structures from a file.

    Supports all ASE-readable formats: POSCAR, xyz, traj, cif, etc.

    Parameters
    ----------
    filename : str or Path
        Path to the structure file.
    index : str
        ASE index string, e.g. ":" for all, "-1" for last.
    format : str, optional
        File format. If None, ASE auto-detects.

    Returns
    -------
    structures : list of Atoms
    """
    result = read(str(filename), index=index, format=format)
    if isinstance(result, Atoms):
        return [result]
    return list(result)


def write_structures(
    filename: str | Path,
    structures: list[Atoms] | Atoms,
    format: str | None = None,
    **kwargs,
) -> None:
    """Write structures to a file.

    Parameters
    ----------
    filename : str or Path
    structures : Atoms or list of Atoms
    format : str, optional
    """
    write(str(filename), structures, format=format, **kwargs)


def structures_to_db(
    db_file: str | Path,
    structures: list[Atoms],
    **kwargs,
) -> None:
    """Write structures to an ASE SQLite database.

    Parameters
    ----------
    db_file : str or Path
    structures : list of Atoms
    kwargs
        Additional key-value pairs to store with each structure.
    """
    import ase.db

    db = ase.db.connect(str(db_file))
    for atoms in structures:
        kvp = atoms.info.get("key_value_pairs", {})
        kvp.update(kwargs)
        db.write(atoms, key_value_pairs=kvp)
