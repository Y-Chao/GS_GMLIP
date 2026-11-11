"""Fill the space above a slab with solvent molecules using Packmol.

Packmol packs rigid molecules into a region subject to a minimum
intermolecular distance. Here the region is the part of the cell between two
planes parallel to the surface, bounded laterally by the four side faces of
the (possibly oblique) cell, so the result is compatible with in-plane
periodic boundary conditions. The slab and any existing adsorbates are kept
fixed and avoided.

Needs the ``packmol`` executable (``uv pip install -e ".[solvent]"`` ships
one, or install it system-wide).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.build import molecule as ase_molecule
from ase.io import read as ase_read
from ase.io import write as ase_write

from gs_gmlip.interface.core import Interface
from gs_gmlip.interface.operations import _apply_constraints, extra_constraints

WATER_DENSITY = 1.0  # g/cm^3


def packmol_available() -> bool:
    """Whether a ``packmol`` executable is on the PATH."""
    return shutil.which("packmol") is not None


def molecules_for_density(volume: float, molar_mass: float, density: float) -> int:
    """Number of molecules filling ``volume`` (Å³) at ``density`` (g/cm³).

    Args:
        volume: Region volume in Å³.
        molar_mass: Molar mass of one molecule in g/mol.
        density: Target mass density in g/cm³.
    """
    molecules_per_cubic_angstrom = density * units.mol / molar_mass * 1e-24
    return int(round(volume * molecules_per_cubic_angstrom))


def _side_planes(cell: np.ndarray, margin: float) -> list[tuple[np.ndarray, float]]:
    """Inward-facing side planes ``(normal, offset)`` of the cell, inset by ``margin``.

    A point ``r`` is inside when ``normal @ r >= offset`` for all four planes.
    """
    a, b, c = cell
    planes = []
    for edge, other, origin_far in ((a, b, b), (b, a, a)):
        normal = np.cross(edge, c)
        normal /= np.linalg.norm(normal)
        if normal @ other < 0:
            normal = -normal
        planes.append((normal, margin))
        planes.append((-normal, -(normal @ origin_far) + margin))
    return planes


def fill_solvent(
    interface: Interface,
    molecule: Atoms | None = None,
    n_molecules: int | None = None,
    density: float = WATER_DENSITY,
    gap: float = 2.0,
    thickness: float | None = None,
    tolerance: float = 2.0,
    seed: int = 0,
    max_loops: int = 100,
) -> Interface:
    """Pack solvent molecules above the surface with Packmol.

    Args:
        interface: The surface system; the cell normal must be along z
            (as produced by ``operations.build_slab``).
        molecule: Solvent molecule; water if omitted.
        n_molecules: Number of molecules. If omitted it is computed from
            ``density`` and the region volume.
        density: Target density in g/cm³ (used only when ``n_molecules`` is None).
        gap: Distance (Å) between the topmost existing atom and the solvent
            region; the same gap is left below the top of the cell.
        thickness: Thickness (Å) of the solvent region. Defaults to all the
            vacuum available between the two gaps.
        tolerance: Packmol minimum distance (Å) between atoms of different
            molecules; half of it is left free at the lateral cell faces so
            periodic images do not clash.
        seed: Packmol random seed.
        max_loops: Packmol optimization loops (``nloop``) before giving up.

    Returns:
        A new ``Interface`` with the solvent appended as adsorbate molecules.

    Raises:
        RuntimeError: If Packmol is missing or fails to converge.
        ValueError: If the region is empty or too small for one molecule.
    """
    if not packmol_available():
        raise RuntimeError(
            "packmol executable not found; install it with: uv pip install -e '.[solvent]'"
        )
    solvent = molecule.copy() if molecule is not None else ase_molecule("H2O")
    atoms = interface.interface
    cell = np.asarray(atoms.cell)
    if not np.allclose(cell[2, :2], 0.0, atol=1e-6) or not np.allclose(
        cell[:2, 2], 0.0, atol=1e-6
    ):
        raise ValueError(
            "fill_solvent needs the cell normal along z (c perpendicular to a and b)"
        )

    z_low = float(atoms.positions[:, 2].max()) + gap
    z_top = float(cell[2, 2]) - gap
    z_high = z_top if thickness is None else min(z_low + float(thickness), z_top)
    if z_high - z_low < 1.0:
        raise ValueError(
            f"no room for solvent: region is {z_high - z_low:.2f} Å thick; increase the vacuum"
        )

    area = float(np.linalg.norm(np.cross(cell[0], cell[1])))
    if n_molecules is None:
        n_molecules = molecules_for_density(
            area * (z_high - z_low), float(solvent.get_masses().sum()), density
        )
    if n_molecules < 1:
        raise ValueError("the solvent region is too small for a single molecule")

    constraints = [
        f"  over plane 0. 0. 1. {z_low:.4f}",
        f"  below plane 0. 0. 1. {z_high:.4f}",
    ]
    for normal, offset in _side_planes(cell, tolerance / 2.0):
        constraints.append(
            f"  over plane {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f} {offset:.4f}"
        )

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        fixed = atoms.copy()
        fixed.set_constraint()
        fixed.calc = None
        ase_write(workdir / "fixed.xyz", fixed, format="xyz")
        ase_write(workdir / "solvent.xyz", solvent, format="xyz")
        script = "\n".join(
            [
                f"tolerance {tolerance:.3f}",
                "filetype xyz",
                "output packed.xyz",
                f"seed {int(seed)}",
                f"nloop {int(max_loops)}",
                "",
                "structure fixed.xyz",
                "  number 1",
                "  fixed 0. 0. 0. 0. 0. 0.",
                "end structure",
                "",
                "structure solvent.xyz",
                f"  number {int(n_molecules)}",
                *constraints,
                "end structure",
                "",
            ]
        )
        # Packmol rewinds its standard input, so it must be a real file, not a pipe.
        (workdir / "pack.inp").write_text(script)
        with (workdir / "pack.inp").open() as stdin:
            completed = subprocess.run(  # noqa: S603, S607 - fixed argv, input is generated here
                ["packmol"],
                stdin=stdin,
                cwd=workdir,
                capture_output=True,
                text=True,
                check=False,
            )
        packed_path = workdir / "packed.xyz"
        if (
            completed.returncode != 0
            or "Success!" not in completed.stdout
            or not packed_path.exists()
        ):
            tail = "\n".join((completed.stdout + completed.stderr).splitlines()[-12:])
            raise RuntimeError(
                f"Packmol could not pack {n_molecules} molecules; "
                f"reduce the number/density or the tolerance.\n{tail}"
            )
        packed = ase_read(packed_path, format="xyz")

    result = interface.copy()
    size = len(solvent)
    for start in range(len(atoms), len(packed), size):
        result.interface += Atoms(
            packed.get_chemical_symbols()[start : start + size],
            positions=packed.positions[start : start + size],
        )
    result._init_ads_cluster(None, None)
    # Keep each solvent molecule as its own adsorbate group regardless of contacts.
    solvent_groups = [
        list(range(start, start + size))
        for start in range(len(atoms), len(result.interface), size)
    ]
    result.adsList = [list(g) for g in interface.adsList] + solvent_groups
    result.clusterList = [list(g) for g in interface.clusterList]
    result._reset_cache()
    _apply_constraints(result, extra_constraints(interface))
    return result


def fill_water(interface: Interface, **kwargs) -> Interface:
    """Pack water above the surface; see ``fill_solvent`` for the arguments."""
    return fill_solvent(interface, molecule=None, **kwargs)
