"""ASOP: supercell enumeration, composition grids, Monte Carlo selection, and the task."""

from __future__ import annotations

import json

import numpy as np
import pytest
from ase.constraints import FixAtoms

from gm2_surf.evaluator.calculators import CalculatorEvaluator
from gm2_surf.interface import operations as ops
from gm2_surf.jobs import TASKS, JobSpec, run_task
from gm2_surf.search.asop import ASOPRunner, Phase, enumerate_supercells
from gm2_surf.store import Project

SQUARE = np.diag([2.5, 2.5, 20.0])
HEXAGONAL = np.array(
    [[2.5, 0.0, 0.0], [-1.25, 2.5 * np.sqrt(3) / 2, 0.0], [0.0, 0.0, 20.0]]
)


@pytest.fixture(scope="module")
def clean():
    """Primitive Cu(100) surface cell, three layers, bottom layer fixed."""
    return ops.build_slab(
        ops.make_bulk("Cu", "fcc", 3.61),
        (1, 0, 0),
        layers=2,
        fix_layers=1,
        primitive=True,
    )


def _runner(clean, **kwargs):
    settings = {
        "max_area": 2,
        "max_coverage": 0.5,
        "max_cycles": 1,
        "search_steps": 0,
        "seed": 1,
        **kwargs,
    }
    return ASOPRunner(
        CalculatorEvaluator("emt", fmax=0.2, relax_steps=20),
        clean,
        {"O": -4.5},
        **settings,
    )


# ---------------------------------------------------------------- step 1: grids


def test_supercells_of_a_square_lattice():
    cells = enumerate_supercells(SQUARE, 1, 4)
    found = {
        (
            cell.area,
            round(cell.lengths[0] / 2.5, 3),
            round(cell.lengths[1] / 2.5, 3),
            round(cell.angle),
        )
        for cell in cells
    }
    assert found == {
        (1, 1.0, 1.0, 90),  # p(1x1)
        (2, 2.0, 1.0, 90),  # p(2x1)
        (2, 1.414, 1.414, 90),  # c(2x2)
        (
            3,
            2.236,
            1.414,
            72,
        ),  # the oblique cell of area 3; p(3x1) has aspect ratio 3 and is excluded
        (4, 2.0, 2.0, 90),  # p(2x2)
        (4, 2.236, 2.0, 63),
        (4, 2.828, 1.414, 90),  # c(4x2)
    }
    assert [cell.area for cell in cells] == sorted(cell.area for cell in cells)
    for cell in cells:
        (a1, b1), (a2, b2) = cell.matrix
        assert a1 * b2 - b1 * a2 == cell.area
        assert 60 - 1e-3 <= cell.angle <= 90 + 1e-3
        assert 1 <= cell.lengths[0] / cell.lengths[1] < 3


def test_supercells_of_a_hexagonal_lattice_include_the_paper_example():
    cells = enumerate_supercells(HEXAGONAL, 1, 4)
    shapes = {
        (
            cell.area,
            round(cell.lengths[0] / 2.5, 3),
            round(cell.lengths[1] / 2.5, 3),
            round(cell.angle),
        )
        for cell in cells
    }
    assert (1, 1.0, 1.0, 60) in shapes
    assert (3, 1.732, 1.732, 60) in shapes  # (sqrt3 x sqrt3)R30
    assert (4, 2.0, 1.732, 90) in shapes  # the (2 x sqrt3) cell of Fig. 2
    assert (4, 2.0, 2.0, 60) in shapes  # p(2x2)


def test_grids_cover_compositions_up_to_the_coverage_limit(clean):
    runner = _runner(clean, max_area=2, max_coverage=1.0)
    runner.setup()
    assert runner.sites_per_cell == 1
    by_area = {grid.supercell.area: grid for grid in runner.grids}
    assert [grid.supercell.area for grid in runner.grids] == [1, 2, 2]
    assert by_area[1].elements == [(1,)]
    assert by_area[2].elements == [(1,), (2,)]
    for grid in runner.grids:
        assert len(grid.slab) == grid.supercell.area * len(clean.substrate)
        assert abs(grid.slab.cell[0, 1]) < 1e-8  # first vector along x
        fixed = [
            i for c in grid.slab.constraints if isinstance(c, FixAtoms) for i in c.index
        ]
        assert len(fixed) == grid.supercell.area * len(clean.fix)
    assert runner.n_cycles == 3


def test_two_species_grid(clean):
    runner = ASOPRunner(
        CalculatorEvaluator("emt"),
        clean,
        {"Cu": -3.5, "O": -4.5},
        min_area=2,
        max_area=2,
        max_ratio=1.5,
    )
    runner.setup()
    (grid,) = runner.grids  # c(2x2); p(2x1) has aspect ratio 2
    assert len(grid.elements) == 3 * 3 - 1
    assert runner.formula((1, 2)) == "Cu1O2"


def test_rejects_bad_input(clean):
    with pytest.raises(ValueError, match="clean surface"):
        _runner(
            ops.place_adsorbate(clean, "O", ops.find_adsorption_sites(clean)[0].coords)
        )
    with pytest.raises(ValueError, match="chemical potential"):
        ASOPRunner(CalculatorEvaluator("emt"), clean, {})


# ---------------------------------------------------------------- step 3: selection


def test_probabilities_follow_the_phase_history(clean):
    runner = _runner(clean, min_area=4, max_area=4, max_coverage=1.0)
    runner.setup()
    grid = runner.grids[0]
    assert [count for (count,) in grid.elements] == [1, 2, 3, 4]
    uniform = runner.probabilities(0, beta=1.0)
    assert np.allclose(uniform, 0.25)  # no history: every composition is equally likely

    # History from a coarser grid: half coverage is stable, full coverage is not.
    runner.phases[(9, (1,))] = Phase(9, (1,), -0.40, clean, (0.5,))
    runner.phases[(9, (2,))] = Phase(9, (2,), +0.30, clean, (1.0,))
    soft, sharp = runner.probabilities(0, beta=0.5), runner.probabilities(0, beta=1.5)
    for probability in (soft, sharp):
        assert np.isclose(probability.sum(), 1.0)
        assert probability.argmax() == 1  # 2 of 4 sites = half coverage
        assert probability.argmin() == 3
    assert sharp[1] > soft[1]  # larger beta focuses on the favourable composition
    assert np.allclose(runner.probabilities(0, beta=0.0), 0.25)

    # Eq. 9 by hand for the composition at half coverage.
    q = np.array([0.25, 0.5, 0.75, 1.0])
    epsilon = np.minimum(
        -0.40 * np.exp(-((q - 0.5) ** 2) / 0.02),
        0.30 * np.exp(-((q - 1.0) ** 2) / 0.02),
    )
    expected = np.exp(-1.5 * (epsilon - epsilon.min()) / abs(epsilon.min()))
    assert np.allclose(sharp, expected / expected.sum())


def test_selection_respects_the_limit_and_keeps_the_best(clean):
    runner = _runner(
        clean,
        min_area=4,
        max_area=4,
        max_coverage=1.0,
        max_compositions=2,
        max_cycles=3,
    )
    runner.setup()
    runner.phases[(9, (1,))] = Phase(9, (1,), -0.40, clean, (0.5,))
    runner.cycle = 2
    for _ in range(20):
        selected = runner._select(0)
        assert 1 <= len(selected) <= 2
        assert selected[0] == (2,)


# ---------------------------------------------------------------- step 2 and the full loop


def test_grown_structures_respect_the_distance_window(clean):
    runner = _runner(clean, max_area=4, max_coverage=1.0, relax_additions=False)
    runner.setup()
    grid = next(grid for grid in runner.grids if grid.supercell.area == 4)
    grown = runner._grow(grid, (3,))
    atoms = grown.interface
    assert runner.added_counts(grown) == (3,)
    assert len(grown.substrate) == len(grid.slab)
    distances = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    for index in range(len(grid.slab), len(atoms)):
        assert distances[index].min() >= 1.5 - 1e-6


def test_run_finds_phases_and_scores_them(clean):
    runner = _runner(clean)
    runner.setup()
    infos = [runner.step() for _ in range(runner.n_cycles)]
    assert runner.finished and runner.step() == {"status": "finished"}
    assert [info["cycle"] for info in infos] == [1] * len(infos)
    assert runner.phases
    table = runner.phase_table()
    assert [row["gamma"] for row in table] == sorted(row["gamma"] for row in table)
    for phase in runner.phases.values():
        structure = phase.structure
        area = runner.grids[phase.grid].supercell.area
        assert runner.added_counts(structure) == phase.counts
        expected = (
            structure.interface.get_potential_energy()
            - area * runner.clean_energy
            - phase.counts[0] * -4.5
        ) / area
        assert phase.gamma == pytest.approx(expected)
        assert phase.gamma == pytest.approx(runner.surface_free_energy(structure))
    assert runner.get_best(1)[0] is runner.best_phases(1)[0].structure


def test_revisited_composition_restarts_from_its_best_structure(clean):
    runner = _runner(clean, max_area=1, max_coverage=1.0, max_cycles=2)
    runner.setup()
    runner.step()
    (phase,) = runner.phases.values()
    first = phase.gamma
    runner.step()
    (phase,) = runner.phases.values()
    assert phase.visits == 2
    assert phase.gamma <= first + 1e-9


def test_asop_task_stores_rows_scored_by_surface_free_energy(tmp_path, clean):
    assert TASKS["asop"].category == "global search" and TASKS["asop"].applies_to == (
        "surface",
    )
    project = Project(tmp_path / "proj")
    row_id = project.store.add(clean, stage="built", role="surface")
    params = {"chemical_potentials": {"O": -4.5}, "max_area": 2, "max_coverage": 0.5, "max_cycles": 1,
              "search_steps": 1, "max_md_steps": 20, "relax_steps": 20, "fmax": 0.2}  # fmt: skip
    context = run_task(
        JobSpec(kind="asop", structure_id=row_id, job_id="asop-a", params=params),
        project.store,
        project.root,
    )
    records = project.store.records(job_id="asop-a")
    assert context.n_structures == len(records) > 2
    assert {record.stage for record in records} == {"asop"}
    assert (
        records[0].score == 0.0 and records[0].role == "surface"
    )  # the clean surface is the reference
    assert all(record.score is not None for record in records)
    assert {record.role for record in records[1:]} == {"adsorbed"}
    table = json.loads((project.root / "asop_phases.json").read_text())
    assert table and table[0]["gamma"] == pytest.approx(
        min(record.score for record in records[1:]), abs=1e-6
    )
    assert {"supercell", "area", "composition", "O", "gamma", "visits"} <= set(table[0])
