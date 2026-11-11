"""Command-line entry point: ``gs_gmlip app`` and ``gs_gmlip run-job``."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    """Run the ``gs_gmlip`` command line."""
    parser = argparse.ArgumentParser(prog="gs_gmlip", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    app = commands.add_parser("app", help="start the interactive app")
    app.add_argument(
        "--project",
        default="gs_project",
        help="project directory (default: ./gs_project)",
    )
    app.add_argument("--host", default="127.0.0.1")
    app.add_argument("--port", type=int, default=8080)
    app.add_argument(
        "--no-browser", action="store_true", help="do not open a browser window"
    )

    run_job = commands.add_parser("run-job", help="run one job directory to completion")
    run_job.add_argument("job_dir")

    args = parser.parse_args(argv)
    if args.command == "run-job":
        from gs_gmlip.jobs.worker import main as worker_main

        return worker_main([args.job_dir])

    try:
        from gs_gmlip.app.main import run_app
    except ImportError as error:
        print(
            f"The app needs the 'app' extra: uv pip install -e '.[app]'  ({error})",
            file=sys.stderr,
        )
        return 1
    run_app(
        project=args.project, host=args.host, port=args.port, show=not args.no_browser
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
