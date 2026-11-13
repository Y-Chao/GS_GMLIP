"""Tests for Packmol-based solvent filling."""

from __future__ import annotations

import numpy as np
import pytest

from gm2_surf.interface import operations as ops
from gm2_surf.interface.solvation import (
    fill_solvent,
    fill_water,
    molecules_for_density,
    packmol_available,
)

needs_packmol = pytest.mark.skipif(
    not packmol_available(), reason="packmol executable not installed"
)


@pytest.fixture(scope="module")
def slab():
    bulk = ops.make_bulk("Cu", "fcc", 3.61)
    slab = ops.build_slab(
        bulk, (1, 1, 1), layers=3, vacuum=30.0, supercell=(2, 2), fix_layers=1
    )
    slab = ops.place_adsorbate(slab, "CO", ops.find_adsorption_sites(slab)[0].coords)
    return ops.add_bond_constraint(
        slab, len(slab.interface) - 2, len(slab.interface) - 1
    )


def test_molecules_for_density_of_water():
    # 1 g/cm^3 water is 33.4 molecules per nm^3.
    assert molecules_for_density(1000.0, 18.015, 1.0) == 33
    assert molecules_for_density(1000.0, 18.015, 0.5) == 17


@needs_packmol
def test_fill_water_density_region_and_clashes(slab):
    thickness, gap, tolerance = 8.0, 2.0, 2.0
    filled = fill_water(slab, thickness=thickness, gap=gap, tolerance=tolerance, seed=3)
    atoms, n_before = filled.interface, len(slab.interface)
    waters = filled.adsList[len(slab.adsList) :]

    area = np.linalg.norm(np.cross(atoms.cell[0], atoms.cell[1]))
    assert len(waters) == molecules_for_density(area * thickness, 18.015, 1.0)
    assert len(atoms) == n_before + 3 * len(waters)
    assert all(
        sorted(atoms[i].symbol for i in group) == ["H", "H", "O"] for group in waters
    )

    # Input untouched; slab, CO, frozen atoms, and the bond constraint are preserved.
    assert len(slab.interface) == n_before
    assert np.allclose(atoms.positions[:n_before], slab.interface.positions)
    assert filled.fix == slab.fix
    assert filled.adsList[: len(slab.adsList)] == slab.adsList
    assert ops.describe_constraints(filled) == ops.describe_constraints(slab)

    # Water sits in the requested slab of space (Packmol precision 0.01 Å, loosened).
    z_low = slab.interface.positions[:, 2].max() + gap
    z_water = atoms.positions[n_before:, 2]
    assert z_water.min() > z_low - 0.1
    assert z_water.max() < z_low + thickness + 0.1

    # No clashes, including across the in-plane periodic boundaries.
    distances = atoms.get_all_distances(mic=True)
    for group in waters:
        others = [i for i in range(len(atoms)) if i not in group]
        assert distances[np.ix_(group, others)].min() > tolerance - 0.2
        # Rigid molecules: O-H bond lengths are those of the template.
        oxygen = [i for i in group if atoms[i].symbol == "O"][0]
        for hydrogen in (i for i in group if atoms[i].symbol == "H"):
            assert atoms.get_distance(oxygen, hydrogen) == pytest.approx(
                0.969, abs=0.01
            )


@needs_packmol
def test_fill_solvent_explicit_count_and_seed(slab):
    first = fill_solvent(slab, n_molecules=5, seed=1)
    again = fill_solvent(slab, n_molecules=5, seed=1)
    other = fill_solvent(slab, n_molecules=5, seed=2)
    assert len(first.adsList) == len(slab.adsList) + 5
    assert np.allclose(first.interface.positions, again.interface.positions)
    assert not np.allclose(first.interface.positions, other.interface.positions)


@needs_packmol
def test_fill_water_errors(slab):
    tight = ops.build_slab(
        ops.make_bulk("Cu", "fcc", 3.61), (1, 1, 1), layers=3, vacuum=4.0
    )
    with pytest.raises(ValueError, match="no room for solvent"):
        fill_water(tight)
    with pytest.raises(RuntimeError, match="could not pack"):
        fill_water(slab, n_molecules=60, thickness=5.0, max_loops=3)


def test_missing_packmol_is_reported(slab, monkeypatch):
    monkeypatch.setattr("gm2_surf.interface.solvation.shutil.which", lambda name: None)
    with pytest.raises(RuntimeError, match="packmol executable not found"):
        fill_water(slab)
