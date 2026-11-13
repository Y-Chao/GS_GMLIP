"""Workflows: chains of tasks, each fanning out over structures and parameter values.

A workflow is a list of steps run on a list of input structures:

- **Chain.** The structures a step produces are the inputs of the next step.
  ``take`` chooses which ones are passed on from each job: its ``last``
  structure (e.g. the relaxed geometry), its ``best`` ``n`` (lowest score if
  the task scores its rows, otherwise lowest energy), or ``all`` of them.
- **Fan-out.** A step starts one job per input structure.
- **Sweep.** ``sweep`` maps a parameter to a list of values; a step starts
  one job per combination. A dotted name reaches into a nested parameter
  (``chemical_potentials.O``), and the prefix ``calculator.`` addresses a
  calculator parameter.

So "relax, search, re-relax the five best, write VASP inputs" is a four-step
workflow, and "the same search at six chemical potentials on every surface
in the table" is a one-step workflow with several inputs and a sweep.

There is no daemon. A workflow lives in ``<project>/workflows/<id>/`` as
``workflow.json`` (what to run) and ``state.json`` (how far it is), and moves
forward whenever ``WorkflowEngine.advance`` is called: the app calls it each
time it refreshes the job list, and ``gs_gmlip run-workflow`` calls it in a
loop. Jobs go through the project's executor, so a step can run on this
machine or on a remote host; at most ``max_parallel`` jobs of a workflow are
active at once.
"""

from __future__ import annotations

import copy
import itertools
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gs_gmlip.jobs.executor import BaseExecutor
from gs_gmlip.jobs.spec import JobSpec, _write_json
from gs_gmlip.jobs.tasks import TASKS
from gs_gmlip.store import Project

SPEC_FILE = "workflow.json"
STATE_FILE = "state.json"
TAKE: tuple[str, ...] = ("last", "best", "all")
MAX_JOBS_PER_STEP = 500


@dataclass
class StepSpec:
    """One step of a workflow.

    Attributes:
        kind: Task name registered in ``gs_gmlip.jobs.tasks``.
        calculator: Registered calculator name.
        calculator_params: Keyword parameters for the calculator.
        params: Task parameters, merged over the task's defaults.
        host: Remote host name; empty runs on this machine.
        take: Which structures of each job feed the next step, see ``TAKE``.
        n: Number of structures passed on per job when ``take`` is ``best``.
        sweep: Parameter name -> list of values; one job per combination.
    """

    kind: str
    calculator: str = "emt"
    calculator_params: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    host: str = ""
    take: str = "last"
    n: int = 1
    sweep: dict[str, list[Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in TASKS:
            raise ValueError(f"unknown task {self.kind!r}; registered: {sorted(TASKS)}")
        if self.take not in TAKE:
            raise ValueError(f"take must be one of {TAKE}, not {self.take!r}")
        self.n = int(self.n)
        if self.n < 1:
            raise ValueError("n must be at least 1")
        for name, values in self.sweep.items():
            if not isinstance(values, list) or not values:
                raise ValueError(
                    f"sweep over {name!r} needs a non-empty list of values"
                )

    def variants(self) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
        """All sweep combinations as ``(description, params, calculator_params)``."""
        names = list(self.sweep)
        variants = []
        for values in itertools.product(*(self.sweep[name] for name in names)):
            params, calculator_params = (
                copy.deepcopy(self.params),
                copy.deepcopy(self.calculator_params),
            )
            for name, value in zip(names, values, strict=True):
                target, path = (
                    (calculator_params, name[len("calculator.") :])
                    if name.startswith("calculator.")
                    else (params, name)
                )
                *parents, leaf = path.split(".")
                for key in parents:
                    target = target.setdefault(key, {})
                    if not isinstance(target, dict):
                        raise ValueError(
                            f"cannot sweep {name!r}: {key!r} is not a group of parameters"
                        )
                target[leaf] = value
            description = ", ".join(
                f"{name}={value}" for name, value in zip(names, values, strict=True)
            )
            variants.append((description, params, calculator_params))
        return variants


@dataclass
class WorkflowSpec:
    """What a workflow runs.

    Attributes:
        name: Label; copied onto every job and structure of the workflow.
        inputs: Row ids of the structures the first step starts from.
        steps: The steps, in order.
        max_parallel: Largest number of this workflow's jobs active at once.
        workflow_id: Unique id, assigned on submission.
    """

    name: str
    inputs: list[int]
    steps: list[StepSpec]
    max_parallel: int = 4
    workflow_id: str = ""

    def __post_init__(self) -> None:
        self.steps = [
            step if isinstance(step, StepSpec) else StepSpec(**step)
            for step in self.steps
        ]
        self.inputs = [int(row) for row in self.inputs]
        self.max_parallel = max(1, int(self.max_parallel))
        if not self.steps:
            raise ValueError("a workflow needs at least one step")
        if not self.inputs:
            raise ValueError("a workflow needs at least one input structure")

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowSpec:
        """Inverse of ``to_dict``."""
        return cls(**data)


def _new_state(inputs: list[int]) -> dict[str, Any]:
    return {
        "state": "running",
        "step": 0,
        "message": "",
        "inputs": [list(inputs)],  # per step: the structures it starts from
        "jobs": [[]],  # per step: ids of submitted jobs
        "pending": [],  # job specs of the current step not submitted yet
        "errors": [],
        "submitted": time.time(),
        "finished": None,
    }


class WorkflowEngine:
    """Runs the workflows of a project through its executor.

    Args:
        project: The project.
        executor: Executor the jobs are submitted to. If its ``submit``
            accepts ``host`` (``ProjectExecutor``), steps can name remote hosts.
    """

    def __init__(self, project: Project, executor: BaseExecutor):
        self.project = project
        self.executor = executor
        self.root = project.root / "workflows"

    # ----- files -----

    def _dir(self, workflow_id: str) -> Path:
        return self.root / workflow_id

    def spec(self, workflow_id: str) -> WorkflowSpec:
        """The stored specification of a workflow."""
        return WorkflowSpec.from_dict(
            json.loads((self._dir(workflow_id) / SPEC_FILE).read_text())
        )

    def state(self, workflow_id: str) -> dict[str, Any]:
        """The stored state of a workflow."""
        return json.loads((self._dir(workflow_id) / STATE_FILE).read_text())

    def _save(self, workflow_id: str, state: dict[str, Any]) -> None:
        _write_json(self._dir(workflow_id) / STATE_FILE, state)

    def ids(self) -> list[str]:
        """Ids of all workflows, newest first."""
        if not self.root.exists():
            return []
        found = [
            path.name
            for path in self.root.iterdir()
            if (path / SPEC_FILE).exists() and (path / STATE_FILE).exists()
        ]
        return sorted(
            found, key=lambda name: self.state(name)["submitted"], reverse=True
        )

    # ----- running -----

    def submit(self, spec: WorkflowSpec) -> str:
        """Store a workflow and start its first step. Returns the workflow id."""
        for row in spec.inputs:
            self.project.store.record(row)  # raises for an unknown structure
        for step in spec.steps:
            step.variants()  # raises for an invalid sweep
        spec.workflow_id = (
            spec.workflow_id
            or f"wf-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        )
        directory = self._dir(spec.workflow_id)
        directory.mkdir(parents=True, exist_ok=True)
        _write_json(directory / SPEC_FILE, spec.to_dict())
        state = _new_state(spec.inputs)
        self._plan(spec, state)
        self._fill(spec, state)
        self._save(spec.workflow_id, state)
        return spec.workflow_id

    def _plan(self, spec: WorkflowSpec, state: dict[str, Any]) -> None:
        """Queue the jobs of the current step: one per input structure and sweep combination."""
        index = state["step"]
        step = spec.steps[index]
        pending = []
        for row in state["inputs"][index]:
            for description, params, calculator_params in step.variants():
                label = f"{spec.name} · {index + 1}/{len(spec.steps)} {step.kind}"
                if description:
                    label += f" · {description}"
                job = JobSpec(
                    step.kind,
                    int(row),
                    step.calculator,
                    calculator_params,
                    params,
                    label,
                )
                pending.append({"job": job.to_dict(), "host": step.host})
        if len(pending) > MAX_JOBS_PER_STEP:
            raise ValueError(
                f"step {index + 1} would start {len(pending)} jobs; the limit is {MAX_JOBS_PER_STEP}"
            )
        state["pending"] = pending

    def _fill(self, spec: WorkflowSpec, state: dict[str, Any]) -> None:
        """Submit queued jobs while fewer than ``max_parallel`` are active."""
        jobs = state["jobs"][state["step"]]
        active = sum(self.executor.status(job_id).active for job_id in jobs)
        while state["pending"] and active < spec.max_parallel:
            entry = state["pending"].pop(0)
            job = JobSpec.from_dict(entry["job"])
            try:
                job_id = (
                    self.executor.submit(job, host=entry["host"])
                    if entry["host"]
                    else self.executor.submit(job)
                )
            except Exception as error:  # noqa: BLE001 - one job that cannot start must not stop the others
                state["errors"].append(
                    f"step {state['step'] + 1}, structure {job.structure_id}: {error}"
                )
                if job.job_id:
                    jobs.append(job.job_id)  # the executor recorded the failed job
                continue
            jobs.append(job_id)
            active += 1

    def _outputs(self, step: StepSpec, job_ids: list[str]) -> list[int]:
        """Structures the finished jobs of a step pass on."""
        outputs: list[int] = []
        for job_id in job_ids:
            if self.executor.status(job_id).state != "done":
                continue
            records = self.project.store.records(job_id=job_id)
            if not records:
                continue
            if step.take == "last":
                chosen = records[-1:]
            elif step.take == "all":
                chosen = records
            else:
                scored = [record for record in records if record.score is not None]
                if scored:
                    ranked = sorted(scored, key=lambda record: record.score)
                else:
                    ranked = sorted(
                        (r for r in records if r.energy is not None),
                        key=lambda record: record.energy,
                    )
                chosen = ranked[: step.n]
            outputs.extend(record.id for record in chosen)
        return outputs

    def advance(self, workflow_id: str | None = None) -> None:
        """Move running workflows forward: submit queued jobs, start the next step, or finish."""
        for name in [workflow_id] if workflow_id else self.ids():
            state = self.state(name)
            if state["state"] != "running":
                continue
            spec = self.spec(name)
            before = json.dumps(state, sort_keys=True)
            self._advance(spec, state)
            if json.dumps(state, sort_keys=True) != before:
                self._save(name, state)

    def _advance(self, spec: WorkflowSpec, state: dict[str, Any]) -> None:
        self._fill(spec, state)
        index = state["step"]
        jobs = state["jobs"][index]
        if state["pending"] or any(
            self.executor.status(job_id).active for job_id in jobs
        ):
            return
        outputs = self._outputs(spec.steps[index], jobs)
        failed = sum(self.executor.status(job_id).state != "done" for job_id in jobs)
        last = index + 1 == len(spec.steps)
        if last or not outputs:
            state["finished"] = time.time()
            if not jobs or failed == len(jobs):
                state["state"], state["message"] = (
                    "failed",
                    f"step {index + 1}: no job finished",
                )
            elif not last:
                state["state"], state["message"] = (
                    "failed",
                    f"step {index + 1} produced no structure for the next step",
                )
            else:
                state["state"] = "done"
                state["message"] = f"{len(outputs)} final structures" + (
                    f", {failed} jobs failed" if failed else ""
                )
                state["outputs"] = outputs
            return
        state["step"] = index + 1
        state["inputs"].append(outputs)
        state["jobs"].append([])
        try:
            self._plan(spec, state)
        except ValueError as error:
            state["state"], state["message"], state["finished"] = (
                "failed",
                str(error),
                time.time(),
            )
            return
        self._fill(spec, state)

    def cancel(self, workflow_id: str) -> None:
        """Stop a workflow: cancel its active jobs and drop the queued ones."""
        state = self.state(workflow_id)
        if state["state"] != "running":
            return
        for job_id in state["jobs"][state["step"]]:
            if self.executor.status(job_id).active:
                self.executor.cancel(job_id)
        state.update(
            state="cancelled",
            pending=[],
            message="cancelled by user",
            finished=time.time(),
        )
        self._save(workflow_id, state)

    def run(
        self, workflow_id: str, poll: float = 1.0, timeout: float | None = None
    ) -> dict[str, Any]:
        """Advance a workflow until it ends (or ``timeout`` seconds passed); return its state."""
        deadline = None if timeout is None else time.time() + timeout
        while True:
            self.advance(workflow_id)
            state = self.state(workflow_id)
            if state["state"] != "running" or (
                deadline is not None and time.time() >= deadline
            ):
                return state
            time.sleep(poll)

    # ----- reporting -----

    def summary(self, workflow_id: str) -> dict[str, Any]:
        """A workflow with the job counts of each step, for display."""
        spec, state = self.spec(workflow_id), self.state(workflow_id)
        steps = []
        for index, step in enumerate(spec.steps):
            jobs = state["jobs"][index] if index < len(state["jobs"]) else []
            counts: dict[str, int] = {}
            for job_id in jobs:
                status = self.executor.status(job_id).state
                counts[status] = counts.get(status, 0) + 1
            if index == state["step"] and state["pending"]:
                counts["waiting"] = len(state["pending"])
            steps.append(
                {
                    "kind": step.kind,
                    "calculator": step.calculator,
                    "host": step.host,
                    "take": step.take if step.take != "best" else f"best {step.n}",
                    "sweep": {name: len(values) for name, values in step.sweep.items()},
                    "inputs": len(state["inputs"][index])
                    if index < len(state["inputs"])
                    else None,
                    "jobs": jobs,
                    "counts": counts,
                }
            )
        return {
            "workflow_id": workflow_id,
            "name": spec.name,
            "state": state["state"],
            "active": state["state"] == "running",
            "step": state["step"],
            "message": state["message"],
            "errors": state["errors"],
            "outputs": state.get("outputs", []),
            "steps": steps,
        }

    def summaries(self) -> list[dict[str, Any]]:
        """All workflows, newest first."""
        return [self.summary(name) for name in self.ids()]
