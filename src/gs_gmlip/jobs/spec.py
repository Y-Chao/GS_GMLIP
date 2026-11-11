"""Job specification and on-disk job state.

A job is a directory ``<project>/jobs/<job_id>/`` holding

- ``job.json``: the immutable ``JobSpec`` (what to run),
- ``status.json``: the mutable ``JobStatus`` (how it is going),
- ``log.txt``: worker stdout/stderr.

The spec is plain JSON, so the same job runs from the app, from the command
line (``gs_gmlip run-job <dir>``), or from a scheduler script on a cluster.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SPEC_FILE = "job.json"
STATUS_FILE = "status.json"
LOG_FILE = "log.txt"

ACTIVE_STATES: tuple[str, ...] = ("queued", "running")
FINAL_STATES: tuple[str, ...] = ("done", "failed", "cancelled")


@dataclass
class JobSpec:
    """What to run.

    Attributes:
        kind: Task name registered in ``gs_gmlip.jobs.tasks``.
        structure_id: Row id of the input structure in the project store.
        calculator: Registered calculator name.
        calculator_params: Keyword parameters for the calculator.
        params: Task parameters, merged over the task's defaults.
        label: Free-text label copied onto every structure the job writes.
        job_id: Unique id, assigned on submission.
    """

    kind: str
    structure_id: int
    calculator: str = "emt"
    calculator_params: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    label: str = ""
    job_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobSpec:
        """Inverse of ``to_dict``."""
        return cls(**data)


@dataclass
class JobStatus:
    """How a job is going.

    Attributes:
        state: ``queued``, ``running``, ``done``, ``failed``, or ``cancelled``.
        progress: Fraction complete in [0, 1].
        message: Latest progress or error message.
        pid: Worker process id (local execution).
        n_structures: Structures written to the store so far.
        last_structure_id: Most recent row id written.
        submitted: Submission time (Unix seconds).
        started: Start time (Unix seconds), ``None`` until running.
        finished: End time (Unix seconds), ``None`` until final.
    """

    state: str = "queued"
    progress: float = 0.0
    message: str = ""
    pid: int | None = None
    n_structures: int = 0
    last_structure_id: int | None = None
    submitted: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None

    @property
    def active(self) -> bool:
        """Whether the job is still queued or running."""
        return self.state in ACTIVE_STATES


def new_job_id(kind: str) -> str:
    """Unique, sortable, non-numeric job id such as ``relax-20261111-153000-a1b2c3``."""
    return f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomic JSON write, so a concurrent reader never sees a partial file."""
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def write_spec(job_dir: Path, spec: JobSpec) -> None:
    """Write ``job.json``."""
    job_dir.mkdir(parents=True, exist_ok=True)
    _write_json(job_dir / SPEC_FILE, spec.to_dict())


def read_spec(job_dir: Path) -> JobSpec:
    """Read ``job.json``."""
    return JobSpec.from_dict(json.loads((job_dir / SPEC_FILE).read_text()))


def write_status(job_dir: Path, status: JobStatus) -> None:
    """Write ``status.json``."""
    _write_json(job_dir / STATUS_FILE, asdict(status))


def read_status(job_dir: Path) -> JobStatus:
    """Read ``status.json`` (a fresh ``queued`` status if it does not exist yet)."""
    path = job_dir / STATUS_FILE
    if not path.exists():
        return JobStatus()
    return JobStatus(**json.loads(path.read_text()))


def list_jobs(jobs_dir: Path) -> list[tuple[JobSpec, JobStatus]]:
    """All jobs under ``jobs_dir``, newest first."""
    jobs = []
    for job_dir in jobs_dir.iterdir() if jobs_dir.exists() else ():
        if (job_dir / SPEC_FILE).exists():
            jobs.append((read_spec(job_dir), read_status(job_dir)))
    jobs.sort(key=lambda item: item[1].submitted, reverse=True)
    return jobs
