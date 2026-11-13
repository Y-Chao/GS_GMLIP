"""Jobs: specifications, tasks, and executors."""

from __future__ import annotations

from gm2_surf.jobs.executor import BaseExecutor, LocalExecutor
from gm2_surf.jobs.remote import ProjectExecutor, RemoteExecutor, RemoteHost
from gm2_surf.jobs.spec import JobSpec, JobStatus
from gm2_surf.jobs.workflow import StepSpec, WorkflowEngine, WorkflowSpec
from gm2_surf.jobs.tasks import TASKS, register_searcher, register_task, run_task

__all__ = [
    "TASKS",
    "BaseExecutor",
    "JobSpec",
    "JobStatus",
    "LocalExecutor",
    "ProjectExecutor",
    "RemoteExecutor",
    "RemoteHost",
    "StepSpec",
    "WorkflowEngine",
    "WorkflowSpec",
    "register_searcher",
    "register_task",
    "run_task",
]
