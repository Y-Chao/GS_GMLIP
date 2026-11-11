"""Tests for job specs, tasks, the worker, and the local executor."""

from __future__ import annotations

import time

import pytest

from gs_gmlip.interface import operations as ops
from gs_gmlip.jobs import TASKS, JobSpec, LocalExecutor, run_task
from gs_gmlip.jobs.spec import list_jobs, read_spec, read_status, write_spec
from gs_gmlip.jobs.worker import run_job_dir
from gs_gmlip.store import Project


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
    assert {"singlepoint", "relax", "md", "ga", "gcmc"} <= set(TASKS)


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
            kind="md",
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


def test_gcmc_stores_evaluations(project, input_id):
    context = run_task(
        JobSpec(
            kind="gcmc",
            structure_id=input_id,
            job_id="gcmc-a",
            params={"n_steps": 6, "relax_steps": 3},
        ),
        project.store,
        project.root,
    )
    records = project.store.records(job_id="gcmc-a")
    assert 1 <= context.n_structures == len(records) <= 7
    assert {r.stage for r in records} == {"gcmc"}


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
            kind="md",
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
