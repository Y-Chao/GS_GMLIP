"""Command-line entry point: ``gm2_surf app``, ``make-app``, ``run-job``, ``import``."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    """Run the ``gm2_surf`` command line."""
    parser = argparse.ArgumentParser(prog="gm2_surf", description=__doc__)
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
        help="directory that receives GM2-Surf.app",
    )
    make_app.add_argument("--port", type=int, default=8080)

    run_workflow = commands.add_parser(
        "run-workflow", help="run one workflow directory to completion"
    )
    run_workflow.add_argument("workflow_dir", help="<project>/workflows/<workflow id>")

    run_job = commands.add_parser("run-job", help="run one job directory to completion")
    run_job.add_argument("job_dir")

    exporter = commands.add_parser(
        "export",
        help="write structures of a project or the global store as light JSON Lines text",
    )
    exporter.add_argument("output", help="file to write, e.g. references.jsonl")
    exporter.add_argument(
        "--project", default=None, help="export from this project directory"
    )
    exporter.add_argument(
        "--global",
        dest="to_global",
        action="store_true",
        help="export from the global store",
    )
    exporter.add_argument(
        "--role",
        action="append",
        default=None,
        help="keep only this role (repeatable); all roles by default",
    )
    importer = commands.add_parser(
        "import",
        help="import an ASE database into a project (--project DIR) or the global store (--global)",
    )
    importer.add_argument(
        "source",
        nargs="+",
        help="ASE database file(s), or .jsonl files written by `export`",
    )
    importer.add_argument(
        "--project",
        default=None,
        help="import into this project directory",
    )
    importer.add_argument(
        "--global",
        dest="to_global",
        action="store_true",
        help="import into the global store shared by all projects (never done implicitly)",
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
        from gm2_surf.importers import import_ase_db
        from gm2_surf.store import Project, global_store

        if bool(args.project) == args.to_global:
            print(
                "gm2_surf import: give exactly one target, --project DIR or --global",
                file=sys.stderr,
            )
            return 2
        store = global_store() if args.to_global else Project(args.project).store
        for source in args.source:
            try:
                if str(source).endswith(".jsonl"):
                    from gm2_surf.reference import import_jsonl

                    counts = {"imported": import_jsonl(source, store), "skipped": 0}
                else:
                    counts = import_ase_db(
                        source, store, role=args.role, label_key=args.label_key
                    )
            except (FileNotFoundError, ValueError) as error:
                print(f"gm2_surf import: {error}", file=sys.stderr)
                return 1
            print(
                f"{source}: imported {counts['imported']}, skipped {counts['skipped']} -> {store.path}"
            )  # noqa: T201
        return 0
    if args.command == "export":
        from gm2_surf.reference import write_jsonl
        from gm2_surf.store import Project, global_store

        if bool(args.project) == args.to_global:
            print(
                "gm2_surf export: give exactly one source, --project DIR or --global",
                file=sys.stderr,
            )
            return 2
        store = global_store() if args.to_global else Project(args.project).store
        written = write_jsonl(store, args.output, roles=args.role)
        print(f"wrote {written} structures to {args.output}")  # noqa: T201
        return 0
    if args.command == "make-app":
        from gm2_surf.app.bundle import build_macos_app

        try:
            bundle = build_macos_app(args.output, port=args.port)
        except (RuntimeError, OSError) as error:
            print(f"gm2_surf make-app: {error}", file=sys.stderr)
            return 1
        print(f"built {bundle}")  # noqa: T201
        return 0
    if args.command == "run-workflow":
        from pathlib import Path

        from gm2_surf.jobs.remote import ProjectExecutor
        from gm2_surf.jobs.workflow import WorkflowEngine
        from gm2_surf.plugins import load_plugins
        from gm2_surf.store import Project

        directory = Path(args.workflow_dir).resolve()
        project = Project(directory.parent.parent)
        load_plugins(project.root)
        try:
            state = WorkflowEngine(project, ProjectExecutor(project)).run(
                directory.name
            )
        except FileNotFoundError:
            print(f"gm2_surf run-workflow: no workflow in {directory}", file=sys.stderr)
            return 1
        print(f"{directory.name}: {state['state']} {state['message']}")  # noqa: T201
        return 0 if state["state"] == "done" else 1
    if args.command == "run-job":
        from gm2_surf.jobs.worker import main as worker_main

        return worker_main([args.job_dir])

    try:
        from gm2_surf.app.server import run_app
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
