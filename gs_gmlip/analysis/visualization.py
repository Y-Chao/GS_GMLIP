"""Visualization helpers for structure search results.

Utilities for plotting energy convergence, acceptance rates,
and exporting structures for visualization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write


def save_best_structures(
    structures: list[Atoms],
    energies: list[float],
    n: int = 10,
    output_dir: str | Path = "best_structures",
    fmt: str = "vasp",
) -> list[Path]:
    """Save the n lowest energy structures.

    Parameters
    ----------
    structures : list of Atoms
        All candidate structures.
    energies : list of float
        Corresponding energies.
    n : int
        Number of best structures to save.
    output_dir : str or Path
        Output directory.
    fmt : str
        ASE write format ('vasp', 'extxyz', 'cif', etc.).

    Returns
    -------
    paths : list of Path
        Paths to saved files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sorted_idx = np.argsort(energies)[:n]
    ext_map = {"vasp": "POSCAR", "extxyz": "xyz", "cif": "cif"}
    ext = ext_map.get(fmt, fmt)

    paths = []
    for rank, idx in enumerate(sorted_idx):
        fname = output_dir / f"rank_{rank:03d}_E{energies[idx]:.4f}.{ext}"
        write(str(fname), structures[idx], format=fmt)
        paths.append(fname)

    return paths


def write_convergence_csv(
    trajectory: list[dict],
    output_file: str | Path = "convergence.csv",
) -> None:
    """Write convergence data to CSV.

    Parameters
    ----------
    trajectory : list of dict
        Search trajectory records.
    output_file : str or Path
        Output CSV file.
    """
    import csv

    if not trajectory:
        return

    keys = list(trajectory[0].keys())
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in trajectory:
            writer.writerow(row)
