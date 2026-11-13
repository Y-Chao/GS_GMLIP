"""Remote execution, exercised end to end with the "remote" machine being a local directory."""

from __future__ import annotations

import json
import sys
import time

import pytest

from gs_gmlip.interface import operations as ops
from gs_gmlip.jobs import JobSpec
from gs_gmlip.jobs.remote import (
    SCHEDULERS,
    LocalTransport,
    ProjectExecutor,
    RemoteError,
    RemoteExecutor,
    RemoteHost,
    check_host,
    load_hosts,
    remove_host,
    save_host,
)
from gs_gmlip.store import Project

WORKER = f"{sys.executable} -m gs_gmlip.jobs.worker"


@pytest.fixture
def project(tmp_path):
    return Project(tmp_path / "proj")


@pytest.fixture
def host(tmp_path):
    return RemoteHost(
        name="box",
        host="box",
        scheduler="shell",
        remote_root=str(tmp_path / "remote"),
        command=WORKER,
        sync_interval=0,
        grace=0,
    )


@pytest.fixture
def input_id(project):
    slab = ops.build_slab(
        ops.make_bulk("Cu", "fcc", 3.61), (1, 1, 1), layers=3, fix_layers=1
    )
    slab = ops.place_adsorbate(slab, "CO", ops.find_adsorption_sites(slab)[0].coords)
    # Earlier rows make local ids differ from remote ids, as in a real project.
    project.store.add(slab, stage="built")
    return project.store.add(slab, stage="built", role="adsorbed")


def _wait(executor, job_id, timeout=120.0):
    deadline = time.time() + timeout
    while executor.status(job_id).active:
        assert time.time() < deadline, "job did not finish in time"
        time.sleep(0.2)
    return executor.status(job_id)


def test_remote_job_merges_results_with_remapped_provenance(
    project, host, input_id, tmp_path
):
    executor = RemoteExecutor(project, host, LocalTransport())
    job_id = executor.submit(
        JobSpec(
            kind="relax",
            structure_id=input_id,
            params={"steps": 6, "write_interval": 2},
        )
    )
    assert (
        "queued on box" in executor.status(job_id).message
        or not executor.status(job_id).active
    )
    status = _wait(executor, job_id)
    assert status.state == "done", executor.log(job_id)

    records = project.store.records(job_id=job_id)
    assert status.n_structures == len(records) >= 2
    assert status.last_structure_id == records[-1].id
    assert records[-1].stage == "relaxed"
    assert records[-1].energy is not None
    # The chain of parents ends at the local input row, not at remote row 1.
    assert project.store.lineage(records[-1].id)[0].id == input_id
    assert all(
        record.parent_id is not None and record.parent_id >= input_id
        for record in records
    )
    assert "100.0%" in executor.log(job_id)

    # The remote side worked on its own copy and never saw the local project.
    remote_dir = tmp_path / "remote" / job_id
    assert (
        json.loads((remote_dir / "jobs" / job_id / "job.json").read_text())[
            "structure_id"
        ]
        == 1
    )
    assert len(project.store) == 2 + len(records)

    # Syncing again does not duplicate rows.
    executor.sync(job_id)
    assert len(project.store.records(job_id=job_id)) == len(records)


def test_remote_cancel_keeps_written_structures(project, host, input_id):
    executor = RemoteExecutor(project, host, LocalTransport())
    job_id = executor.submit(
        JobSpec(
            kind="nvt",
            structure_id=input_id,
            params={"steps": 200000, "write_interval": 5},
        )
    )
    deadline = time.time() + 60
    while executor.status(job_id).n_structures == 0:
        assert time.time() < deadline and executor.status(job_id).active, executor.log(
            job_id
        )
        time.sleep(0.2)
    executor.cancel(job_id)
    status = executor.status(job_id)
    assert status.state == "cancelled"
    assert len(project.store.records(job_id=job_id)) >= 1


def test_job_that_leaves_the_queue_silently_is_failed(project, host, input_id):
    host.command = "false"  # the "worker" exits at once without writing a status
    executor = RemoteExecutor(project, host, LocalTransport())
    job_id = executor.submit(JobSpec(kind="relax", structure_id=input_id))
    status = _wait(executor, job_id, timeout=30)
    assert status.state == "failed"
    assert "left the queue" in status.message


def test_submission_failure_is_recorded(project, host, input_id):
    class Broken:
        def run(self, command, stdin=None, timeout=120.0):
            raise RemoteError("box: connection refused")

    executor = RemoteExecutor(project, host, Broken())
    spec = JobSpec(kind="relax", structure_id=input_id)
    with pytest.raises(RemoteError):
        executor.submit(spec)
    status = executor.status(spec.job_id)
    assert status.state == "failed" and "connection refused" in status.message


def test_unreachable_host_keeps_job_active(project, host, input_id):
    executor = RemoteExecutor(project, host, LocalTransport())
    job_id = executor.submit(
        JobSpec(kind="nvt", structure_id=input_id, params={"steps": 200000})
    )

    class Down:
        def run(self, command, stdin=None, timeout=120.0):
            raise RemoteError("box: no answer within 10 s")

    offline = RemoteExecutor(project, host, Down())
    status = offline.status(job_id)
    assert status.active and "cannot reach box" in status.message
    executor.cancel(job_id)


def test_user_model_is_uploaded_and_path_rewritten(project, host, input_id, tmp_path):
    model = tmp_path / "my.model"
    model.write_bytes(b"weights")
    executor = RemoteExecutor(project, host, LocalTransport())
    host.command = "true"
    spec = JobSpec(
        kind="singlepoint",
        structure_id=input_id,
        calculator="mace",
        calculator_params={"model_path": str(model)},
    )
    job_id = executor.submit(spec)
    remote_dir = tmp_path / "remote" / job_id
    assert (remote_dir / "models" / "my.model").read_bytes() == b"weights"
    remote_spec = json.loads((remote_dir / "jobs" / job_id / "job.json").read_text())
    assert remote_spec["calculator_params"]["model_path"] == str(
        (remote_dir / "models" / "my.model").resolve()
    )
    assert spec.calculator_params["model_path"] == str(
        model
    )  # the local spec is untouched


def test_pbs_script():
    host = RemoteHost(
        name="vanda",
        host="vanda",
        queue="batch_gpu",
        account="P1",
        resources="select=1:ncpus=8:ngpus=1",
        walltime="12:00:00",
        directives=["-m abe"],
        setup=["module load cuda"],
    )
    script = RemoteExecutor(None, host)._script(
        host, "relax-20261112-101010-abcdef", "/scratch/u/j"
    )
    lines = script.splitlines()
    assert lines[0] == "#!/bin/bash"
    for directive in (
        "#PBS -N relax-20261112-",
        "#PBS -q batch_gpu",
        "#PBS -P P1",
        "#PBS -l select=1:ncpus=8:ngpus=1",
        "#PBS -l walltime=12:00:00",
        "#PBS -m abe",
    ):
        assert directive in lines
    assert lines.index("module load cuda") > lines.index('cd "/scratch/u/j" || exit 1')
    assert lines[-1].startswith(
        'exec gs_gmlip run-job "/scratch/u/j/jobs/relax-20261112-101010-abcdef"'
    )
    assert host.root == "$HOME/gs_gmlip_jobs"
    assert SCHEDULERS["pbs"].submit.format(script="run.sh") == "qsub run.sh"


@pytest.mark.parametrize(
    "settings",
    [
        {"name": "bad name", "host": "h"},
        {"name": "h", "host": "-oProxyCommand=x"},
        {"name": "h", "host": "h", "scheduler": "lsf"},
        {"name": "h", "host": "h", "remote_root": "~/a; rm -rf b"},
        {"name": "h", "host": "h", "queue": "q\n#PBS -x"},
    ],
)
def test_invalid_host_settings(settings):
    with pytest.raises(ValueError):
        RemoteHost(**settings)


def test_host_registry_and_check(host):
    save_host(host)
    assert load_hosts()["box"].to_dict() == host.to_dict()
    result = check_host(host, LocalTransport())
    assert result["ok"], result
    host.command = "no_such_worker_command run-job"
    result = check_host(host, LocalTransport())
    assert not result["ok"]
    assert [check["ok"] for check in result["checks"]] == [True, True, True, False]
    remove_host("box")
    assert load_hosts() == {}


def test_app_submits_to_host_and_lists_where_jobs_run(tmp_path, host):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from gs_gmlip.app.server import create_app
    from gs_gmlip.app.session import Session

    project = Project(tmp_path / "proj")
    session = Session(project, ProjectExecutor(project, LocalTransport()))
    client = TestClient(create_app(project, session))
    assert (
        client.post("/api/hosts", json=host.to_dict()).json()["hosts"][0]["name"]
        == "box"
    )
    assert client.post("/api/hosts/box/check", json={}).json()["ok"]
    assert (
        client.post("/api/hosts", json={"name": "x y", "host": "h"}).status_code == 400
    )

    ws = "molecule-1"
    assert (
        client.post(
            "/api/molecule/create", params={"ws": ws}, json={"name": "H2O"}
        ).status_code
        == 200
    )
    job = {"kind": "relax", "calculator": "emt", "params": {"steps": 3}}
    remote_id = client.post(
        "/api/jobs", params={"ws": ws}, json={**job, "host": "box"}
    ).json()["job_id"]
    local_id = client.post("/api/jobs", params={"ws": ws}, json=job).json()["job_id"]
    assert (
        client.post(
            "/api/jobs", params={"ws": ws}, json={**job, "host": "nowhere"}
        ).status_code
        == 400
    )

    deadline = time.time() + 120
    while True:
        jobs = {item["job_id"]: item for item in client.get("/api/jobs").json()["jobs"]}
        if not any(item["active"] for item in jobs.values()):
            break
        assert time.time() < deadline
        time.sleep(0.2)
    assert jobs[remote_id]["host"] == "box" and jobs[local_id]["host"] == ""
    assert jobs[remote_id]["state"] == jobs[local_id]["state"] == "done"
    assert jobs[remote_id]["last_structure_id"] in {
        rec.id for rec in project.store.records(job_id=remote_id)
    }
    assert client.delete("/api/hosts/box").json()["hosts"] == []


def test_late_final_status_is_waited_for(project, host, input_id, tmp_path):
    """A job gone from the queue is not failed until the grace period is over."""
    host.grace = 3600
    host.command = "false"
    executor = RemoteExecutor(project, host, LocalTransport())
    job_id = executor.submit(JobSpec(kind="relax", structure_id=input_id))
    time.sleep(0.5)
    status = executor.sync(job_id)
    assert status.active and "waiting for its final status" in status.message

    # The final report becomes visible later: the job ends as reported, not as failed.
    remote_status = tmp_path / "remote" / job_id / "jobs" / job_id / "status.json"
    remote_status.write_text(
        json.dumps({"state": "done", "progress": 1.0, "message": "ok"})
    )
    assert executor.sync(job_id).state == "done"
