"""Command-line entry point: ``gs_gmlip app``, ``make-app``, ``run-job``, ``import``."""

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
        default=None,
        help="project (work) directory for all databases; if omitted, the app asks on start",
    )
    app.add_argument("--host", default="127.0.0.1")
    app.add_argument("--port", type=int, default=8080)
    app.add_argument(
        "--no-browser", action="store_true", help="do not open a browser window"
    )

    app.add_argument(
        "--allow-quit",
        action="store_true",
        help="show a Quit button in the app that stops the server",
    )

    make_app = commands.add_parser(
        "make-app", help="build a double-clickable macOS application (launcher)"
    )
    make_app.add_argument(
        "--output",
        default="~/Applications",
        help="directory that receives GS-GMLIP.app",
    )
    make_app.add_argument("--port", type=int, default=8080)

    run_workflow = commands.add_parser(
        "run-workflow", help="run one workflow directory to completion"
    )
    run_workflow.add_argument("workflow_dir", help="<project>/workflows/<workflow id>")

    run_job = commands.add_parser("run-job", help="run one job directory to completion")
    run_job.add_argument("job_dir")

    importer = commands.add_parser(
        "import",
        help="import an ASE database into the global or a project structure store",
    )
    importer.add_argument("source", nargs="+", help="ASE database file(s)")
    importer.add_argument(
        "--project",
        default=None,
        help="import into this project (default: the global store)",
    )
    importer.add_argument(
        "--role",
        default="auto",
        help="bulk, molecule, surface, adsorbed, other, or auto (default)",
    )
    importer.add_argument(
        "--label-key", default=None, help="key-value pair to use as row label"
    )

    args = parser.parse_args(argv)
    if args.command == "import":
        from gs_gmlip.importers import import_ase_db
        from gs_gmlip.store import Project, global_store

        store = Project(args.project).store if args.project else global_store()
        for source in args.source:
            try:
                counts = import_ase_db(
                    source, store, role=args.role, label_key=args.label_key
                )
            except (FileNotFoundError, ValueError) as error:
                print(f"gs_gmlip import: {error}", file=sys.stderr)
                return 1
            print(
                f"{source}: imported {counts['imported']}, skipped {counts['skipped']} -> {store.path}"
            )  # noqa: T201
        return 0
    if args.command == "make-app":
        from gs_gmlip.app.bundle import build_macos_app

        try:
            bundle = build_macos_app(args.output, port=args.port)
        except (RuntimeError, OSError) as error:
            print(f"gs_gmlip make-app: {error}", file=sys.stderr)
            return 1
        print(f"built {bundle}")  # noqa: T201
        return 0
    if args.command == "run-workflow":
        from pathlib import Path

        from gs_gmlip.jobs.remote import ProjectExecutor
        from gs_gmlip.jobs.workflow import WorkflowEngine
        from gs_gmlip.plugins import load_plugins
        from gs_gmlip.store import Project

        directory = Path(args.workflow_dir).resolve()
        project = Project(directory.parent.parent)
        load_plugins(project.root)
        try:
            state = WorkflowEngine(project, ProjectExecutor(project)).run(
                directory.name
            )
        except FileNotFoundError:
            print(f"gs_gmlip run-workflow: no workflow in {directory}", file=sys.stderr)
            return 1
        print(f"{directory.name}: {state['state']} {state['message']}")  # noqa: T201
        return 0 if state["state"] == "done" else 1
    if args.command == "run-job":
        from gs_gmlip.jobs.worker import main as worker_main

        return worker_main([args.job_dir])

    try:
        from gs_gmlip.app.server import run_app
    except ImportError as error:
        print(
            f"The app needs the 'app' extra: uv pip install -e '.[app]'  ({error})",
            file=sys.stderr,
        )
        return 1
    run_app(
        project=args.project,
        host=args.host,
        port=args.port,
        show=not args.no_browser,
        allow_quit=args.allow_quit,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
