"""Tests for job specs, tasks, the worker, and the local executor."""

from __future__ import annotations

import time

import pytest

from gm2_surf.interface import operations as ops
from gm2_surf.jobs import TASKS, JobSpec, LocalExecutor, run_task
from gm2_surf.jobs.spec import list_jobs, read_spec, read_status, write_spec
from gm2_surf.jobs.worker import run_job_dir
from gm2_surf.store import Project


@pytest.fixture
def project(tmp_path):
    return Project(tmp_path / "proj")


@pytest.fixture
def input_id(project):
    slab = ops.build_slab(
        ops.make_bulk("Cu", "fcc", 3.61), (1, 1, 1), layers=3, fix_layers=1
    )
    slab = ops.place_adsorbate(slab, "CO", ops.find_adsorption_sites(slab)[0].coords)
    return project.store.add(slab, stage="built")


def _wait(executor, job_id, timeout=120.0):
    deadline = time.time() + timeout
    while executor.status(job_id).active:
        assert time.time() < deadline, "job did not finish in time"
        time.sleep(0.2)
    return executor.status(job_id)


def test_registered_tasks():
    assert {"singlepoint", "relax", "cell_relax", "nvt", "npt", "ga", "gcmc"} <= set(
        TASKS
    )


def test_spec_roundtrip(tmp_path):
    spec = JobSpec(kind="relax", structure_id=3, params={"fmax": 0.1}, job_id="relax-x")
    write_spec(tmp_path / "relax-x", spec)
    assert read_spec(tmp_path / "relax-x") == spec
    assert [s.job_id for s, _ in list_jobs(tmp_path)] == ["relax-x"]
    assert read_status(tmp_path / "relax-x").state == "queued"


def test_unknown_task(project, input_id):
    with pytest.raises(KeyError, match="unknown task"):
        run_task(
            JobSpec(kind="nope", structure_id=input_id), project.store, project.root
        )


def test_singlepoint_then_relax_then_md_write_expected_rows(project, input_id):
    store = project.store
    messages = []

    def on_progress(context, progress, message):
        messages.append((progress, message))

    context = run_task(
        JobSpec(kind="singlepoint", structure_id=input_id, job_id="sp-a"),
        store,
        project.root,
        on_progress,
    )
    assert context.n_structures == 1
    assert messages[-1][0] == 1.0

    context = run_task(
        JobSpec(
            kind="relax",
            structure_id=input_id,
            job_id="relax-a",
            params={"steps": 12, "write_interval": 4, "fmax": 1e-6},
        ),
        store,
        project.root,
    )
    records = store.records(job_id="relax-a")
    assert context.n_structures == len(records) == 4
    assert [r.stage for r in records] == ["relax_step"] * 3 + ["relaxed"]
    assert [r.step for r in records] == [4, 8, 12, 12]
    assert all(r.parent_id == input_id and r.calculator == "emt" for r in records)
    energies = [r.energy for r in records]
    assert energies[-1] < store.record(store.records(job_id="sp-a")[0].id).energy
    # Fixed atoms did not move.
    start, end = store.get(input_id), store.get(records[-1].id)
    assert (
        start.interface.positions[start.fix] == end.interface.positions[end.fix]
    ).all()

    context = run_task(
        JobSpec(
            kind="nvt",
            structure_id=input_id,
            job_id="md-a",
            params={"steps": 20, "write_interval": 10},
        ),
        store,
        project.root,
    )
    records = store.records(job_id="md-a")
    assert context.n_structures == len(records) == 3
    assert [r.step for r in records] == [0, 10, 20]
    assert {r.stage for r in records} == {"md"}


def test_ga_stores_every_evaluated_candidate(project, input_id):
    context = run_task(
        JobSpec(
            kind="ga",
            structure_id=input_id,
            job_id="ga-a",
            params={
                "blocks": {"O": 2},
                "n_initial": 3,
                "population_size": 3,
                "n_steps": 2,
                "relax_steps": 3,
                "fmax": 0.5,
            },
        ),
        project.store,
        project.root,
    )
    records = project.store.records(job_id="ga-a")
    assert 3 <= context.n_structures == len(records) <= 5
    assert [r.step for r in records] == list(range(1, len(records) + 1))
    assert all(r.formula.endswith("O3") or "O3" in r.formula for r in records)
    assert all(r.energy is not None for r in records)


def test_gcmc_stores_evaluations_with_grand_potential(project):
    bare = ops.build_slab(
        ops.make_bulk("Cu", "fcc", 3.61), (1, 1, 1), layers=3, fix_layers=1
    )
    bare_id = project.store.add(bare, stage="built")
    context = run_task(
        JobSpec(
            kind="gcmc",
            structure_id=bare_id,
            job_id="gcmc-a",
            params={"n_steps": 12, "relax_steps": 3},
        ),
        project.store,
        project.root,
    )
    records = project.store.records(job_id="gcmc-a")
    assert 2 <= context.n_structures == len(records) <= 13
    assert {r.stage for r in records} == {"gcmc"}
    # The starting surface is the reference; every other row is scored against it.
    reference = records[0]
    assert reference.score == pytest.approx(0.0)
    mu = TASKS["gcmc"].defaults["chemical_potentials"]["O"]
    assert any(record.natoms > reference.natoms for record in records)
    for record in records:
        extra_oxygen = record.natoms - reference.natoms
        expected = record.energy - reference.energy - mu * extra_oxygen
        assert record.score == pytest.approx(expected, abs=1e-8)


def test_worker_records_failure(project, input_id):
    job_dir = project.jobs_dir / "relax-bad"
    write_spec(
        job_dir,
        JobSpec(
            kind="relax",
            structure_id=input_id,
            calculator="does-not-exist",
            job_id="relax-bad",
        ),
    )
    assert run_job_dir(job_dir) == 1
    status = read_status(job_dir)
    assert status.state == "failed"
    assert "unknown calculator" in status.message


def test_local_executor_runs_job_in_subprocess(project, input_id):
    executor = LocalExecutor(project)
    job_id = executor.submit(
        JobSpec(
            kind="relax",
            structure_id=input_id,
            params={"steps": 5, "write_interval": 5},
        )
    )
    status = _wait(executor, job_id)
    assert status.state == "done", executor.log(job_id)
    assert status.progress == 1.0
    records = project.store.records(job_id=job_id)
    assert status.n_structures == len(records) >= 1
    assert status.last_structure_id == records[-1].id
    assert records[-1].stage == "relaxed"
    assert "100.0%" in executor.log(job_id)
    assert [spec.job_id for spec, _ in executor.jobs()] == [job_id]


def test_local_executor_cancel(project, input_id):
    executor = LocalExecutor(project)
    job_id = executor.submit(
        JobSpec(
            kind="nvt",
            structure_id=input_id,
            params={"steps": 2_000_000, "write_interval": 1000},
        )
    )
    deadline = time.time() + 60
    while executor.status(job_id).state != "running":
        assert time.time() < deadline
        time.sleep(0.2)
    executor.cancel(job_id)
    assert executor.status(job_id).state == "cancelled"
    executor.cancel(job_id)  # cancelling twice is a no-op


@pytest.fixture
def bulk_id(project):
    from pymatgen.io.ase import AseAtomsAdaptor

    from gm2_surf.interface.core import Interface

    bulk = ops.make_bulk("Cu", "fcc", 3.70)
    bulk.make_supercell([2, 2, 2])
    return project.store.add(
        Interface(substrate=AseAtomsAdaptor.get_atoms(bulk)), stage="built"
    )


def test_cell_relax_finds_emt_lattice_constant_and_feeds_slab_builder(project, bulk_id):
    context = run_task(
        JobSpec(
            kind="cell_relax",
            structure_id=bulk_id,
            job_id="cell-a",
            params={"write_interval": 2},
        ),
        project.store,
        project.root,
    )
    records = project.store.records(job_id="cell-a")
    assert context.n_structures == len(records)
    assert records[-1].stage == "cell_relaxed"
    assert {r.stage for r in records[:-1]} <= {"cell_relax_step"}
    relaxed = project.store.get(records[-1].id)
    # EMT equilibrium lattice constant of fcc Cu is 3.59 Å (2x2x2 conventional supercell).
    assert relaxed.interface.cell.lengths() == pytest.approx([2 * 3.59] * 3, abs=0.02)
    assert records[-1].energy < project.store.records(job_id="cell-a")[0].energy + 1e-9

    conventional = ops.conventional_cell(relaxed.interface)
    assert len(conventional) == 4
    assert conventional.lattice.a == pytest.approx(3.59, abs=0.01)
    slab = ops.build_slab(conventional, (1, 1, 1), layers=3)
    distances = slab.interface.get_all_distances(mic=True)
    distances[distances < 1e-8] = 1e9
    assert distances.min() == pytest.approx(conventional.lattice.a / 2**0.5, abs=1e-3)


@pytest.mark.parametrize("isotropic", [True, False])
def test_npt_changes_volume_and_stores_frames(project, bulk_id, isotropic):
    context = run_task(
        JobSpec(
            kind="npt",
            structure_id=bulk_id,
            job_id=f"npt-{isotropic}".lower(),
            params={
                "steps": 30,
                "write_interval": 10,
                "temperature": 600.0,
                "isotropic": isotropic,
            },
        ),
        project.store,
        project.root,
    )
    records = project.store.records(job_id=f"npt-{isotropic}".lower())
    assert context.n_structures == len(records) == 4
    assert [r.step for r in records] == [0, 10, 20, 30]
    start = project.store.get(bulk_id).interface.get_volume()
    end = project.store.get(records[-1].id).interface.get_volume()
    # a = 3.70 Å is expanded for EMT Cu, so the barostat must shrink the cell.
    assert end < start


def test_npt_rejects_constraints(project, input_id):
    with pytest.raises(ValueError, match="does not support constraints"):
        run_task(
            JobSpec(kind="npt", structure_id=input_id, job_id="npt-bad"),
            project.store,
            project.root,
        )


def test_relax_honours_bond_constraint(project, input_id):
    start = project.store.get(input_id)
    n_atoms = len(start.interface)
    constrained = ops.add_bond_constraint(start, n_atoms - 2, n_atoms - 1)
    constrained.interface.rattle(0.05, seed=1)
    bond = constrained.interface.get_distance(n_atoms - 2, n_atoms - 1)
    row_id = project.store.add(constrained, stage="built")
    run_task(
        JobSpec(
            kind="relax", structure_id=row_id, job_id="relax-c", params={"steps": 15}
        ),
        project.store,
        project.root,
    )
    relaxed = project.store.get(project.store.records(job_id="relax-c")[-1].id)
    assert relaxed.interface.get_distance(n_atoms - 2, n_atoms - 1) == pytest.approx(
        bond, abs=1e-6
    )
    assert ops.describe_constraints(relaxed)["bonds"] == [[n_atoms - 2, n_atoms - 1]]
