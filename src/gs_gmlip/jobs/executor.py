"""Executors: where a job runs.

``LocalExecutor`` starts a detached worker process on this machine. Cluster
back ends (an HPC manager submitting the same worker command) only need to
subclass ``BaseExecutor``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path

from gs_gmlip.jobs.spec import (
    LOG_FILE,
    JobSpec,
    JobStatus,
    list_jobs,
    new_job_id,
    read_status,
    write_spec,
    write_status,
)
from gs_gmlip.jobs.tasks import TASKS
from gs_gmlip.store import Project


class BaseExecutor(ABC):
    """Submits jobs of a project and reports on them.

    Args:
        project: The project whose store the jobs read from and write to.
    """

    def __init__(self, project: Project):
        self.project = project

    def job_dir(self, job_id: str) -> Path:
        """Directory of a job."""
        return self.project.jobs_dir / job_id

    def prepare(self, spec: JobSpec) -> Path:
        """Assign a job id and write the job directory. Returns the directory."""
        if spec.kind not in TASKS:
            raise KeyError(f"unknown task {spec.kind!r}; registered: {sorted(TASKS)}")
        spec.job_id = spec.job_id or new_job_id(spec.kind)
        job_dir = self.job_dir(spec.job_id)
        write_spec(job_dir, spec)
        write_status(job_dir, JobStatus())
        return job_dir

    @abstractmethod
    def submit(self, spec: JobSpec) -> str:
        """Start a job and return its id."""

    @abstractmethod
    def cancel(self, job_id: str) -> None:
        """Stop a job."""

    def status(self, job_id: str) -> JobStatus:
        """Current status of a job."""
        return read_status(self.job_dir(job_id))

    def jobs(self) -> list[tuple[JobSpec, JobStatus]]:
        """All jobs of the project, newest first, with refreshed statuses."""
        return [
            (spec, self.status(spec.job_id))
            for spec, _ in list_jobs(self.project.jobs_dir)
        ]

    def log(self, job_id: str, tail: int = 200) -> str:
        """Last ``tail`` lines of a job's log."""
        path = self.job_dir(job_id) / LOG_FILE
        if not path.exists():
            return ""
        return "\n".join(path.read_text(errors="replace").splitlines()[-tail:])


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class LocalExecutor(BaseExecutor):
    """Runs each job as a detached ``python -m gs_gmlip.jobs.worker`` process.

    Jobs keep running if the app is closed; their state lives in the job
    directory, so a restarted app picks them up again.
    """

    def __init__(self, project: Project):
        super().__init__(project)
        self._processes: dict[str, subprocess.Popen] = {}

    def submit(self, spec: JobSpec) -> str:
        """Start the worker process for ``spec`` and return the job id."""
        job_dir = self.prepare(spec)
        with (job_dir / LOG_FILE).open("w") as log:
            process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                [sys.executable, "-m", "gs_gmlip.jobs.worker", str(job_dir)],
                cwd=job_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self._processes[spec.job_id] = process
        status = read_status(job_dir)
        if status.state == "queued":
            status.pid = process.pid
            write_status(job_dir, status)
        return spec.job_id

    def status(self, job_id: str) -> JobStatus:
        """Current status; a worker that died without reporting is marked failed."""
        job_dir = self.job_dir(job_id)
        status = read_status(job_dir)
        if not status.active:
            return status
        process = self._processes.get(job_id)
        if process is not None:
            dead = process.poll() is not None
        else:
            dead = status.pid is not None and not _pid_alive(status.pid)
        if dead:
            status = read_status(job_dir)  # the worker may have finished just now
            if status.active:
                status.state = "failed"
                status.message = "worker exited without reporting; see log"
                status.finished = time.time()
                write_status(job_dir, status)
        return status

    def cancel(self, job_id: str) -> None:
        """Terminate the worker process and mark the job cancelled."""
        job_dir = self.job_dir(job_id)
        status = self.status(job_id)
        if not status.active:
            return
        process = self._processes.get(job_id)
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        elif status.pid is not None and _pid_alive(status.pid):
            os.kill(status.pid, 15)
        status = read_status(job_dir)
        if status.active:
            status.state = "cancelled"
            status.message = "cancelled by user"
            status.finished = time.time()
            write_status(job_dir, status)
