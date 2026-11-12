"""Tests for the plugin API: project plugin files and searcher registration."""

from __future__ import annotations

import textwrap
import time

import pytest

from gm2_surf.interface import operations as ops
from gm2_surf.jobs import TASKS, JobSpec, LocalExecutor, register_searcher, run_task
from gm2_surf.plugins import load_plugins
from gm2_surf.search.base import BaseSearcher
from gm2_surf.store import Project

PLUGIN = textwrap.dedent(
    """
    from gm2_surf.evaluator import register_calculator
    from gm2_surf.jobs import register_searcher
    from gm2_surf.search.base import BaseSearcher


    class Rattle(BaseSearcher):
        def __init__(self, evaluator, start, workdir, amplitude, seed):
            super().__init__(evaluator, workdir, seed)
            self.start, self.amplitude = start, amplitude

        def setup(self):
            self._is_setup = True

        def step(self):
            candidate = self.start.copy()
            candidate.interface.rattle(self.amplitude, seed=self.seed + self._step)
            self._step += 1
            self.evaluator.evaluate(candidate, relax=True)
            return {"step": self._step}


    register_searcher(
        "{kind}",
        lambda evaluator, start, params, workdir: Rattle(
            evaluator, start, workdir, params["amplitude"], params["seed"]
        ),
        defaults={"n_steps": 3, "relax_steps": 2, "amplitude": 0.1, "seed": 0},
        description="Rattle and relax (test plugin)",
    )


    def _emt(**params):
        from ase.calculators.emt import EMT

        return EMT()


    register_calculator("{kind}_engine", _emt, description="EMT under a plugin name")
    """
)


@pytest.fixture
def project(tmp_path):
    return Project(tmp_path / "proj")


@pytest.fixture
def input_id(project):
    slab = ops.build_slab(
        ops.make_bulk("Cu", "fcc", 3.61), (1, 1, 1), layers=3, fix_layers=1
    )
    return project.store.add(slab, stage="built")


def test_register_searcher_wraps_a_base_searcher(project, input_id):
    class Shake(BaseSearcher):
        def __init__(self, evaluator, start, workdir):
            super().__init__(evaluator, workdir)
            self.start = start

        def setup(self):
            self._is_setup = True

        def step(self):
            candidate = self.start.copy()
            candidate.interface.rattle(0.05, seed=self._step)
            self._step += 1
            self.evaluator.evaluate(candidate, relax=True)
            return {}

    register_searcher(
        "shake_test",
        lambda evaluator, start, params, workdir: Shake(evaluator, start, workdir),
        defaults={"n_steps": 4, "relax_steps": 2},
        description="test",
        score=lambda params: lambda interface: float(len(interface.interface)),
    )
    try:
        assert TASKS["shake_test"].defaults == {
            "n_steps": 4,
            "fmax": 0.1,
            "relax_steps": 2,
        }
        context = run_task(
            JobSpec(kind="shake_test", structure_id=input_id, job_id="shake-a"),
            project.store,
            project.root,
        )
        records = project.store.records(job_id="shake-a")
        assert context.n_structures == len(records) == 4
        assert {(r.stage, r.method, r.parent_id) for r in records} == {
            ("shake_test", "shake_test", input_id)
        }
        assert all(
            r.score == records[0].natoms and r.energy is not None for r in records
        )
    finally:
        TASKS.pop("shake_test", None)


def test_project_plugin_file_is_loaded_once_and_broken_plugins_are_skipped(project):
    folder = project.root / "plugins"
    folder.mkdir()
    (folder / "a_rattle.py").write_text(PLUGIN.replace("{kind}", "rattle_once"))
    (folder / "b_broken.py").write_text("raise RuntimeError('boom')\n")
    try:
        assert load_plugins(project.root) == ["a_rattle"]
        assert "rattle_once" in TASKS
        assert (
            load_plugins(project.root) == []
        )  # already loaded; the broken one is retried and skipped
    finally:
        TASKS.pop("rattle_once", None)


def test_plugin_search_runs_in_a_worker_process(project, input_id):
    folder = project.root / "plugins"
    folder.mkdir()
    (folder / "rattle.py").write_text(PLUGIN.replace("{kind}", "rattle_worker"))
    load_plugins(project.root)
    try:
        executor = LocalExecutor(project)
        job_id = executor.submit(
            JobSpec(
                kind="rattle_worker",
                structure_id=input_id,
                calculator="rattle_worker_engine",
            )
        )
        deadline = time.time() + 120
        while executor.status(job_id).active:
            assert time.time() < deadline
            time.sleep(0.2)
        assert executor.status(job_id).state == "done", executor.log(job_id)
        records = project.store.records(job_id=job_id)
        assert len(records) == 3
        assert {r.calculator for r in records} == {"rattle_worker_engine"}
    finally:
        TASKS.pop("rattle_worker", None)
