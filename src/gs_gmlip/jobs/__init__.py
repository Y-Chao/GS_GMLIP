"""Jobs: specifications, tasks, and executors."""

from __future__ import annotations

from gs_gmlip.jobs.executor import BaseExecutor, LocalExecutor
from gs_gmlip.jobs.remote import ProjectExecutor, RemoteExecutor, RemoteHost
from gs_gmlip.jobs.spec import JobSpec, JobStatus
from gs_gmlip.jobs.workflow import StepSpec, WorkflowEngine, WorkflowSpec
from gs_gmlip.jobs.tasks import TASKS, register_searcher, register_task, run_task

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
