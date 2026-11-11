"""Job worker: run one job directory to completion.

Invoked as ``python -m gs_gmlip.jobs.worker <job_dir>``. This is the single
entry point every executor uses, local or (later) through an HPC manager: the
scheduler script only has to call it inside the project directory.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from pathlib import Path

from gs_gmlip.jobs.spec import read_spec, read_status, write_status
from gs_gmlip.jobs.tasks import run_task
from gs_gmlip.plugins import load_plugins
from gs_gmlip.store import Project

logger = logging.getLogger(__name__)

# Minimum interval (s) between progress writes to status.json.
_STATUS_INTERVAL = 0.5


def run_job_dir(job_dir: str | Path) -> int:
    """Run the job stored in ``job_dir``.

    The project root is taken as ``job_dir/../..``.

    Returns:
        Process exit code: 0 on success, 1 on failure.
    """
    job_dir = Path(job_dir).resolve()
    project = Project(job_dir.parent.parent)
    load_plugins(project.root)
    spec = read_spec(job_dir)
    status = read_status(job_dir)
    status.state = "running"
    status.pid = os.getpid()
    status.started = time.time()
    write_status(job_dir, status)

    last_write = 0.0

    def on_progress(context, progress: float, message: str) -> None:
        nonlocal last_write
        logger.info("%5.1f%%  %s", 100 * progress, message)
        now = time.time()
        if progress < 1.0 and now - last_write < _STATUS_INTERVAL:
            return
        status.progress = float(progress)
        status.message = message
        status.n_structures = context.n_structures
        status.last_structure_id = context.last_structure_id
        write_status(job_dir, status)
        last_write = now

    try:
        context = run_task(spec, project.store, job_dir, on_progress)
    except Exception as error:  # noqa: BLE001 - any task failure must be recorded, not lost
        traceback.print_exc()
        status.state = "failed"
        status.message = f"{type(error).__name__}: {error}"
        status.finished = time.time()
        write_status(job_dir, status)
        return 1

    status.state = "done"
    status.progress = 1.0
    status.n_structures = context.n_structures
    status.last_structure_id = context.last_structure_id
    status.finished = time.time()
    write_status(job_dir, status)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m gs_gmlip.jobs.worker <job_dir>", file=sys.stderr)
        return 2
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout
    )
    return run_job_dir(argv[0])


if __name__ == "__main__":
    sys.exit(main())
