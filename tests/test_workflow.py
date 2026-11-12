"""Workflows: chains, fan-out over structures, parameter sweeps."""

from __future__ import annotations

import pytest

from gs_gmlip.interface import operations as ops
from gs_gmlip.jobs import LocalExecutor, StepSpec, WorkflowEngine, WorkflowSpec
from gs_gmlip.store import Project

FAST = {"steps": 3, "write_interval": 3}


@pytest.fixture
def project(tmp_path):
    return Project(tmp_path / "proj")


@pytest.fixture
def engine(project):
    return WorkflowEngine(project, LocalExecutor(project))


def _slab(site: int = 0):
    slab = ops.build_slab(
        ops.make_bulk("Cu", "fcc", 3.61), (1, 1, 1), layers=3, fix_layers=1
    )
    return ops.place_adsorbate(slab, "CO", ops.find_adsorption_sites(slab)[site].coords)


def test_chain_passes_the_last_structure_on(project, engine):
    row = project.store.add(_slab(), stage="built", role="adsorbed")
    steps = [StepSpec("relax", params=FAST), StepSpec("singlepoint")]
    workflow_id = engine.submit(WorkflowSpec("chain", [row], steps))
    state = engine.run(workflow_id, poll=0.2, timeout=120)
    assert state["state"] == "done", state
    relax_job, single_job = state["jobs"][0][0], state["jobs"][1][0]
    relaxed = project.store.records(job_id=relax_job)[-1]
    assert state["inputs"] == [[row], [relaxed.id]]
    (final,) = project.store.records(job_id=single_job)
    assert final.parent_id == relaxed.id and final.stage == "singlepoint"
    assert state["outputs"] == [final.id]
    assert project.store.lineage(final.id)[0].id == row
    assert "chain · 2/2 singlepoint" in final.label


def test_fan_out_and_sweep_with_limited_parallelism(project, engine):
    rows = [
        project.store.add(_slab(site), stage="built", role="adsorbed")
        for site in (0, 1)
    ]
    step = StepSpec(
        "nvt",
        params={"steps": 4, "write_interval": 4},
        sweep={"temperature": [200.0, 400.0, 600.0]},
        take="all",
    )
    workflow_id = engine.submit(WorkflowSpec("sweep", rows, [step], max_parallel=2))
    state = engine.state(workflow_id)
    assert (
        len(state["jobs"][0]) == 2 and len(state["pending"]) == 4
    )  # 2 structures x 3 temperatures, 2 at a time
    waiting = engine.summary(workflow_id)["steps"][0]["counts"]["waiting"]
    assert waiting == 4
    state = engine.run(workflow_id, poll=0.2, timeout=180)
    assert state["state"] == "done", state
    assert len(state["jobs"][0]) == 6
    labels = {project.store.records(job_id=job)[0].label for job in state["jobs"][0]}
    assert {label.split(" · ")[-1] for label in labels} == {
        "temperature=200.0",
        "temperature=400.0",
        "temperature=600.0",
    }
    assert len(state["outputs"]) == sum(
        len(project.store.records(job_id=job)) for job in state["jobs"][0]
    )
    summary = engine.summary(workflow_id)
    assert summary["steps"][0]["counts"] == {"done": 6} and summary["steps"][0][
        "sweep"
    ] == {"temperature": 3}


def test_best_n_feeds_the_next_step(project, engine):
    row = project.store.add(_slab(), stage="built", role="adsorbed")
    steps = [
        StepSpec("nvt", params={"steps": 12, "write_interval": 3}, take="best", n=2),
        StepSpec("singlepoint"),
    ]
    state = engine.run(
        engine.submit(WorkflowSpec("best", [row], steps)), poll=0.2, timeout=180
    )
    assert state["state"] == "done", state
    frames = project.store.records(job_id=state["jobs"][0][0])
    lowest = sorted(frames, key=lambda record: record.energy)[:2]
    assert state["inputs"][1] == [record.id for record in lowest]
    assert len(state["jobs"][1]) == 2


def test_nested_and_calculator_sweeps():
    step = StepSpec(
        "gcmc",
        params={"chemical_potentials": {"CO": -1.0}},
        sweep={"chemical_potentials.CO": [-1.5, -0.5], "calculator.model": ["a"]},
    )
    variants = step.variants()
    assert [params["chemical_potentials"]["CO"] for _, params, _ in variants] == [
        -1.5,
        -0.5,
    ]
    assert all(calculator == {"model": "a"} for _, _, calculator in variants)
    assert variants[0][0] == "chemical_potentials.CO=-1.5, calculator.model=a"
    assert (
        step.params["chemical_potentials"]["CO"] == -1.0
    )  # the step itself is untouched


@pytest.mark.parametrize(
    "bad",
    [
        {"kind": "nope"},
        {"kind": "relax", "take": "first"},
        {"kind": "relax", "n": 0},
        {"kind": "relax", "sweep": {"steps": []}},
    ],
)
def test_invalid_steps(bad):
    with pytest.raises(ValueError):
        StepSpec(**bad)


def test_invalid_workflows(project, engine):
    with pytest.raises(ValueError, match="at least one step"):
        WorkflowSpec("x", [1], [])
    with pytest.raises(ValueError, match="input structure"):
        WorkflowSpec("x", [], [StepSpec("relax")])
    with pytest.raises(KeyError):
        engine.submit(WorkflowSpec("x", [99], [StepSpec("relax")]))  # no such structure


def test_failed_step_stops_the_chain(project, engine):
    row = project.store.add(_slab(), stage="built", role="adsorbed")
    steps = [
        StepSpec("npt", params={"steps": 2}),
        StepSpec("singlepoint"),
    ]  # NPT rejects fixed atoms
    state = engine.run(
        engine.submit(WorkflowSpec("fail", [row], steps)), poll=0.2, timeout=120
    )
    assert state["state"] == "failed" and "step 1" in state["message"]
    assert len(state["jobs"]) == 1


def test_cancel(project, engine):
    row = project.store.add(_slab(), stage="built", role="adsorbed")
    step = StepSpec(
        "nvt", params={"steps": 200000}, sweep={"temperature": [100.0, 200.0, 300.0]}
    )
    workflow_id = engine.submit(WorkflowSpec("cancel", [row], [step], max_parallel=1))
    engine.cancel(workflow_id)
    state = engine.state(workflow_id)
    assert state["state"] == "cancelled" and state["pending"] == []
    assert not any(engine.executor.status(job).active for job in state["jobs"][0])
    engine.advance()  # a cancelled workflow starts nothing more
    assert len(engine.state(workflow_id)["jobs"][0]) == 1


def test_workflow_api(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from gs_gmlip.app.server import create_app

    client = TestClient(create_app(tmp_path / "proj"))
    assert client.get("/api/meta").json()["workflow_take"] == ["last", "best", "all"]
    ws = "molecule-1"
    assert (
        client.post(
            "/api/molecule/create", params={"ws": ws}, json={"name": "H2O"}
        ).status_code
        == 200
    )
    steps = [
        {"kind": "relax", "calculator": "emt", "params": FAST},
        {"kind": "singlepoint", "calculator": "emt"},
    ]
    response = client.post(
        "/api/workflows", params={"ws": ws}, json={"name": "api", "steps": steps}
    )
    assert response.status_code == 200, response.text
    workflow_id = response.json()["workflow_id"]
    assert (
        client.post(
            "/api/workflows", params={"ws": ws}, json={"name": "x", "steps": []}
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/api/workflows",
            params={"ws": ws},
            json={"steps": [{"kind": "relax", "host": "nowhere"}]},
        ).status_code
        == 400
    )

    for _ in range(600):
        jobs = client.get("/api/jobs/wait", params={"timeout": 5}).json()[
            "jobs"
        ]  # also advances workflows
        (workflow,) = client.get("/api/workflows").json()["workflows"]
        if not workflow["active"]:
            break
    assert workflow["workflow_id"] == workflow_id and workflow["state"] == "done"
    assert [step["counts"] for step in workflow["steps"]] == [{"done": 1}, {"done": 1}]
    assert len(jobs) == 2 and all(job["label"].startswith("api · ") for job in jobs)
    assert (
        client.post(f"/api/workflows/{workflow_id}/cancel").json()["workflows"][0][
            "state"
        ]
        == "done"
    )


def test_choose_folder(monkeypatch, tmp_path):
    from gs_gmlip import config

    calls = []

    class Done:
        returncode = 0
        stdout = f"{tmp_path}/new project/\n"

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return Done()

    monkeypatch.setattr(config.sys, "platform", "darwin")
    monkeypatch.setattr(config.subprocess, "run", fake_run)
    assert config.choose_folder(start=tmp_path) == f"{tmp_path}/new project"
    assert calls[0][0] == "/usr/bin/osascript" and calls[0][-1] == str(tmp_path)
    Done.returncode = 1  # the user pressed Cancel
    assert config.choose_folder() == ""
    monkeypatch.setattr(config.sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="macOS"):
        config.choose_folder()
