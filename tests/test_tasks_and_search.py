"""Tests for task categories, minima hopping, user-trained model paths, and waiting on jobs."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from gm2_surf.evaluator import (
    CalculatorEvaluator,
    get_calculator,
    level_of_theory,
    resolve_model_path,
)
from gm2_surf.interface import operations as ops
from gm2_surf.jobs import (
    TASKS,
    JobSpec,
    LocalExecutor,
    register_searcher,
    register_task,
    run_task,
)
from gm2_surf.jobs.tasks import MODEL_TYPES, TASK_CATEGORIES
from gm2_surf.search.minima_hopping import MinimaHoppingRunner
from gm2_surf.store import Project


@pytest.fixture
def project(tmp_path):
    return Project(tmp_path / "proj")


@pytest.fixture(scope="module")
def adsorbed():
    slab = ops.build_slab(
        ops.make_bulk("Cu", "fcc", 3.61),
        (1, 1, 1),
        layers=3,
        supercell=(2, 2),
        fix_layers=2,
    )
    sites = ops.find_adsorption_sites(slab)
    slab = ops.place_adsorbate(slab, "CO", sites[0].coords)
    return ops.place_adsorbate(slab, "O", sites[40].coords)


# ---------------------------------------------------------------- categories


def test_builtin_tasks_have_categories_and_model_types():
    labels = {kind: (task.category, task.applies_to) for kind, task in TASKS.items()}
    assert labels["singlepoint"] == ("single point", MODEL_TYPES)
    assert labels["relax"] == ("optimization", MODEL_TYPES)
    assert labels["cell_relax"] == ("optimization", ("bulk",))
    assert labels["nvt"] == ("molecular dynamics", MODEL_TYPES)
    assert labels["npt"] == ("molecular dynamics", ("bulk",))
    assert labels["ga"] == labels["gcmc"] == ("global search", ("surface",))
    assert labels["minima_hopping"] == ("global search", ("surface", "molecule"))
    assert labels["vasp_inputs"][0] == "input files"
    assert all(task.category in TASK_CATEGORIES for task in TASKS.values())


def test_registration_validates_labels():
    with pytest.raises(ValueError, match="unknown task category"):
        register_task("bad", {}, category="nope")
    with pytest.raises(ValueError, match="unknown model types"):
        register_task("bad", {}, applies_to=("crystal",))
    with pytest.raises(ValueError, match="unknown model types"):
        register_searcher("bad", lambda *args: None, {}, applies_to=("crystal",))
    assert "bad" not in TASKS


# ---------------------------------------------------------------- minima hopping


def test_minima_hopping_adapts_temperature_and_threshold(adsorbed):
    evaluator = CalculatorEvaluator("emt", fmax=0.1, relax_steps=40)
    runner = MinimaHoppingRunner(
        evaluator, adsorbed, temperature=1200.0, ediff=0.4, max_md_steps=120, seed=3
    )
    runner.setup()
    assert (
        len(runner.get_results()) == 1 and runner.current_energy == runner.best_energy
    )
    start_fixed = adsorbed.interface.positions[adsorbed.fix]

    temperature, ediff = runner.temperature, runner.ediff
    statuses = []
    for _ in range(5):
        info = runner.step()
        statuses.append(info["status"])
        # Feedback rules of the algorithm.
        factor = {"returned": runner.beta1, "known": runner.beta2, "new": runner.beta3}[
            info["status"]
        ]
        assert runner.temperature == pytest.approx(temperature * factor)
        if info["status"] == "returned":
            assert runner.ediff == pytest.approx(ediff) and not info["accepted"]
        else:
            assert runner.ediff == pytest.approx(
                ediff * (runner.alpha1 if info["accepted"] else runner.alpha2)
            )
        temperature, ediff = runner.temperature, runner.ediff

    results = runner.get_results()
    assert len(results) == 1 + statuses.count("new")
    assert statuses.count("new") >= 1  # the search moved somewhere
    energies = [r.interface.get_potential_energy() for r in results]
    assert runner.best_energy == pytest.approx(min(energies))
    assert runner.get_best(1)[0].interface.get_potential_energy() == pytest.approx(
        min(energies)
    )
    for result in results:
        # Composition and frozen atoms are conserved through MD and relaxation.
        assert (
            result.interface.get_chemical_formula()
            == adsorbed.interface.get_chemical_formula()
        )
        assert np.allclose(result.interface.positions[result.fix], start_fixed)
    # Distinct minima only: no two results share energy and fingerprint.
    for i, first in enumerate(results):
        for second in results[i + 1 :]:
            assert not runner._same(
                first.interface.get_potential_energy(), first.fingerprint,
                second.interface.get_potential_energy(), second.fingerprint,
            )  # fmt: skip


def test_minima_hopping_task_stores_every_relaxation(project, adsorbed):
    row_id = project.store.add(adsorbed, stage="built", role="adsorbed")
    context = run_task(
        JobSpec(
            kind="minima_hopping",
            structure_id=row_id,
            job_id="mh-a",
            params={"n_steps": 4, "relax_steps": 30, "fmax": 0.1, "max_md_steps": 100},
        ),
        project.store,
        project.root,
    )
    records = project.store.records(job_id="mh-a")
    assert context.n_structures == len(records) == 5  # the start plus one per hop
    assert {(r.stage, r.role, r.level) for r in records} == {
        ("minima_hopping", "adsorbed", "emt")
    }
    assert all(r.energy is not None for r in records)
    assert min(r.energy for r in records) <= records[0].energy


# ---------------------------------------------------------------- user-trained models


def test_resolve_model_path(tmp_path):
    model = tmp_path / "run1" / "my.model"
    model.parent.mkdir()
    model.write_bytes(b"weights")
    assert resolve_model_path(model) == model.resolve()
    with pytest.raises(FileNotFoundError, match="no model at"):
        resolve_model_path(tmp_path / "missing.model")


def test_custom_models_get_their_own_level_of_theory():
    foundation = level_of_theory("mace")
    custom = level_of_theory("mace", {"model_path": "~/models/CuSCHO/stagetwo.model"})
    assert foundation == "mace/medium"
    assert custom == "mace/custom:CuSCHO/stagetwo.model"
    other = level_of_theory("mace", {"model_path": "/data/other/stagetwo.model"})
    assert other != custom  # same file name in another folder is another model
    assert (
        level_of_theory("matgl", {"model_path": "/models/tensornet_v2"})
        == "matgl/custom:models/tensornet_v2"
    )


def test_matgl_reports_a_missing_model_directory(tmp_path):
    pytest.importorskip("matgl")
    with pytest.raises(FileNotFoundError, match="no model at"):
        get_calculator("matgl", model_path=str(tmp_path / "nope"))


def test_matgl_loads_a_model_from_a_directory():
    pytest.importorskip("matgl")
    cached = sorted((Path.home() / ".cache" / "matgl").rglob("model.json"))
    if not cached:
        pytest.skip("no MatGL model directory on this machine")
    atoms = ops.build_slab(
        ops.make_bulk("Cu", "fcc", 3.61), (1, 1, 1), layers=3
    ).interface
    atoms.calc = get_calculator("matgl", model_path=str(cached[0].parent))
    assert np.isfinite(atoms.get_potential_energy())
    assert atoms.get_forces().shape == (len(atoms), 3)


# ---------------------------------------------------------------- waiting with callbacks


def test_wait_reports_updates_until_all_jobs_finish(project, adsorbed):
    row_id = project.store.add(adsorbed, stage="built", role="adsorbed")
    executor = LocalExecutor(project)
    first = executor.submit(
        JobSpec(
            kind="relax", structure_id=row_id, params={"steps": 8, "write_interval": 2}
        )
    )
    second = executor.submit(JobSpec(kind="singlepoint", structure_id=row_id))
    updates: list[tuple[str, str, int]] = []
    statuses = executor.wait(
        timeout=120,
        poll=0.1,
        on_update=lambda job, status: updates.append(
            (job, status.state, status.n_structures)
        ),
    )
    assert set(statuses) == {first, second}
    assert {status.state for status in statuses.values()} == {"done"}
    assert {job for job, _, _ in updates} == {first, second}
    assert [state for job, state, _ in updates if job == first][-1] == "done"
    assert (
        max(count for job, _, count in updates if job == first)
        == statuses[first].n_structures
        >= 1
    )


def test_wait_times_out_and_leaves_the_job_running(project, adsorbed):
    row_id = project.store.add(adsorbed, stage="built", role="adsorbed")
    executor = LocalExecutor(project)
    job_id = executor.submit(
        JobSpec(
            kind="nvt",
            structure_id=row_id,
            params={"steps": 2_000_000, "write_interval": 1000},
        )
    )
    started = time.time()
    statuses = executor.wait([job_id], timeout=1.0, poll=0.1)
    assert time.time() - started < 10
    assert statuses[job_id].active
    executor.cancel(job_id)
    assert executor.wait([job_id], timeout=30)[job_id].state == "cancelled"
