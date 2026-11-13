"""JSON API and static file server for the JavaScript frontend.

The server is a thin shell around ``Session``: each route parses a JSON
payload, calls one session method, and returns the session state (or a
listing). Domain errors become HTTP 400 with the message in ``detail``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from gm2_surf.app.session import Session
from gm2_surf.store import Project

STATIC_DIR = Path(__file__).parent / "static"

# Errors that mean "the request cannot be done as asked", not "the server broke".
_DOMAIN_ERRORS = (
    ValueError,
    KeyError,
    IndexError,
    RuntimeError,
    ImportError,
    TypeError,
)


def _run(action: Callable[[], Any]) -> Any:
    try:
        return action()
    except _DOMAIN_ERRORS as error:
        message = (
            error.args[0] if isinstance(error, KeyError) and error.args else str(error)
        )
        raise HTTPException(status_code=400, detail=str(message)) from error


def create_app(
    project: str | Path | Project | None, session: Session | None = None
) -> FastAPI:  # noqa: C901, PLR0915
    """Build the FastAPI application for a project.

    Args:
        project: Project directory (or ``Project``) holding the store and jobs.
            ``None`` starts without a project: nothing is created on disk and
            every route except the project chooser answers 409 until the user
            opens or creates a project folder.
        session: Session to serve; a new one is created by default.
    """
    if project is not None and not isinstance(project, Project):
        project = Project(project)
    app = FastAPI(
        title="GM2-Surf", docs_url="/api/docs", openapi_url="/api/openapi.json"
    )
    app.state.session = session or (Session(project) if project is not None else None)
    app.state.ask_project = app.state.session is None
    # Set by ``run_app(allow_quit=True)``: a callable that stops the server.
    app.state.quit = None

    class _Current:
        """Attribute proxy to the app's current session, which changes when a project is opened."""

        def __getattr__(self, name: str) -> Any:
            if app.state.session is None:
                raise HTTPException(
                    status_code=409, detail="open or create a project folder first"
                )
            return getattr(app.state.session, name)

    session = _Current()  # type: ignore[assignment]

    def then_state(ws: str, action: Callable[[], Any]) -> dict[str, Any]:
        _run(action)
        return _run(lambda: session.state(ws))

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        if app.state.session is None:
            from gm2_surf.config import folder_picker_available, recent_projects

            return {
                "project": "",
                "recent_projects": recent_projects(),
                "folder_picker": folder_picker_available(),
                "ask_project": True,
                "app": "gm2_surf",
                "can_quit": app.state.quit is not None,
            }
        return {
            **session.meta(),
            "ask_project": app.state.ask_project,
            "app": "gm2_surf",
            "can_quit": app.state.quit is not None,
        }

    @app.post("/api/quit")
    def quit_app() -> dict[str, Any]:
        """Stop the server (only when it was started as a desktop app). Running jobs continue."""
        if app.state.quit is None:
            raise HTTPException(
                status_code=403, detail="this server was not started with --allow-quit"
            )
        app.state.quit()
        return {"stopping": True}

    @app.post("/api/project")
    def open_project(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Open (or create) a project directory; all databases live inside it."""
        path = str(payload.get("path", "")).strip()
        if not path:
            raise HTTPException(status_code=400, detail="give a project directory")
        app.state.session = _run(lambda: Session(Project(path)))
        app.state.ask_project = False
        return {
            **session.meta(),
            "ask_project": False,
            "app": "gm2_surf",
            "can_quit": app.state.quit is not None,
        }

    @app.get("/api/state")
    def state(ws: str = "") -> dict[str, Any]:
        return _run(lambda: session.state(ws))

    # ----- windows -----

    @app.get("/api/windows")
    def windows() -> dict[str, Any]:
        return {"windows": session.windows()}

    @app.post("/api/windows")
    def add_window(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        workspace_id = _run(
            lambda: session.add_window(payload["kind"], payload.get("name"))
        )
        return session.state(workspace_id)

    @app.delete("/api/windows/{workspace_id}")
    def remove_window(workspace_id: str) -> dict[str, Any]:
        _run(lambda: session.remove_window(workspace_id))
        return {"windows": session.windows()}

    # ----- building -----

    @app.post("/api/bulk/create")
    def bulk_create(
        ws: str = "", payload: dict[str, Any] = Body(...)
    ) -> dict[str, Any]:
        return then_state(
            ws,
            lambda: session.create_bulk(
                ws,
                payload["element"],
                payload["lattice"],
                payload["a"],
                payload.get("c"),
            ),
        )

    @app.post("/api/molecule/create")
    def molecule_create(
        ws: str = "", payload: dict[str, Any] = Body(...)
    ) -> dict[str, Any]:
        return then_state(ws, lambda: session.create_molecule(ws, payload["name"]))

    @app.post("/api/upload")
    async def upload(request: Request, name: str, ws: str = "") -> dict[str, Any]:
        content = await request.body()
        return then_state(ws, lambda: session.load_file(ws, name, content))

    @app.post("/api/cleave")
    def cleave(ws: str = "", payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        target = _run(
            lambda: session.cleave(
                ws,
                payload["miller"],
                target_id=payload.get("target") or None,
                layers=payload.get("layers", 4),
                vacuum=payload.get("vacuum", 15.0),
                supercell=tuple(payload.get("supercell", (1, 1))),
                fix_layers=payload.get("fix_layers", 0),
                primitive=payload.get("primitive", False),
                conventional=payload.get("conventional", True),
            )
        )
        return session.state(target)

    @app.post("/api/sites")
    def sites(ws: str = "", payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return then_state(
            ws,
            lambda: session.set_sites(
                ws,
                payload.get("show", False),
                payload.get("distinct", False),
                payload.get("height", 2.0),
            ),
        )

    @app.post("/api/library")
    def library_add(
        ws: str = "", payload: dict[str, Any] = Body(...)
    ) -> dict[str, Any]:
        return then_state(
            ws,
            lambda: session.library_add(
                ws,
                payload["name"],
                payload.get("scope", "temporary"),
                payload.get("bind_index"),
                payload.get("reactions"),
                payload.get("note", ""),
            ),
        )

    @app.post("/api/library/{name}/open")
    def library_open(name: str, ws: str = "") -> dict[str, Any]:
        return then_state(ws, lambda: session.library_open(ws, name))

    @app.delete("/api/library/{name}")
    def library_remove(name: str, ws: str = "") -> dict[str, Any]:
        return then_state(ws, lambda: session.library_remove(name))

    @app.post("/api/edit")
    def edit(ws: str = "", payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        params = dict(payload)
        operation = params.pop("op", "")
        return then_state(ws, lambda: session.edit(ws, operation, **params))

    @app.post("/api/undo")
    def undo(ws: str = "") -> dict[str, Any]:
        return then_state(ws, lambda: session.undo(ws))

    @app.post("/api/save")
    def save(
        ws: str = "", payload: dict[str, Any] = Body(default={})
    ) -> dict[str, Any]:
        return then_state(ws, lambda: session.save(ws, payload.get("label", "")))

    @app.get("/api/export")
    def export(ws: str = "", format: str = "extxyz") -> PlainTextResponse:  # noqa: A002
        text, filename = _run(lambda: session.export(ws, format))
        return PlainTextResponse(
            text, headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )

    # ----- jobs -----

    @app.get("/api/jobs")
    def jobs() -> dict[str, Any]:
        return session.jobs()

    @app.get("/api/jobs/wait")
    def wait_jobs(timeout: float = 20.0) -> dict[str, Any]:
        """Long poll: returns when no job is active, or after ``timeout`` seconds."""
        return session.wait_jobs(min(timeout, 60.0))

    @app.post("/api/jobs")
    def submit(ws: str = "", payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        job_id = _run(
            lambda: session.submit(
                ws,
                payload["kind"],
                payload.get("calculator", "emt"),
                payload.get("calculator_params"),
                payload.get("params"),
                payload.get("label", ""),
                payload.get("host", ""),
            )
        )
        return {"job_id": job_id, "state": session.state(ws)}

    @app.get("/api/workflows")
    def workflows() -> dict[str, Any]:
        return _run(lambda: session.workflow_list())

    @app.post("/api/workflows")
    def submit_workflow(
        ws: str = "", payload: dict[str, Any] = Body(...)
    ) -> dict[str, Any]:
        workflow_id = _run(
            lambda: session.submit_workflow(
                ws,
                payload.get("name", ""),
                payload.get("steps", []),
                payload.get("inputs"),
                payload.get("max_parallel", 4),
            )
        )
        return {"workflow_id": workflow_id, "state": session.state(ws)}

    @app.post("/api/workflows/{workflow_id}/cancel")
    def cancel_workflow(workflow_id: str) -> dict[str, Any]:
        return _run(lambda: session.cancel_workflow(workflow_id))

    @app.post("/api/choose-folder")
    def choose_folder(payload: dict[str, Any] = Body(default={})) -> dict[str, str]:
        """Show the native folder dialog on the machine the app runs on."""
        from gm2_surf.config import choose_folder as ask

        return {"path": _run(lambda: ask(start=payload.get("start") or None))}

    @app.get("/api/host-presets")
    def host_presets() -> dict[str, Any]:
        return _run(lambda: session.host_presets())

    @app.get("/api/hosts")
    def hosts() -> dict[str, Any]:
        return _run(lambda: session.hosts())

    @app.post("/api/hosts")
    def save_host(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return _run(lambda: session.save_host(**payload))

    @app.delete("/api/hosts/{name}")
    def remove_host(name: str) -> dict[str, Any]:
        return _run(lambda: session.remove_host(name))

    @app.post("/api/hosts/{name}/check")
    def check_host(name: str) -> dict[str, Any]:
        return _run(lambda: session.check_host(name))

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str) -> dict[str, Any]:
        _run(lambda: session.executor.cancel(job_id))
        return session.jobs()

    @app.get("/api/jobs/{job_id}/log")
    def log(job_id: str) -> dict[str, str]:
        return {"job_id": job_id, "log": session.executor.log(job_id)}

    @app.post("/api/vasp/preview")
    def vasp_preview(
        ws: str = "", payload: dict[str, Any] = Body(default={})
    ) -> dict[str, Any]:
        return _run(lambda: session.vasp_preview(ws, **payload))

    # ----- store -----

    @app.post("/api/items")
    def set_item(ws: str = "", payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return then_state(
            ws,
            lambda: session.set_item(
                ws, payload["key"], bool(payload.get("checked", True))
            ),
        )

    @app.get("/api/references")
    def references(role: str = "bulk") -> dict[str, Any]:
        return {"references": _run(lambda: session.references(role))}

    @app.get("/api/energetics")
    def energetics(ws: str = "") -> dict[str, Any]:
        return _run(lambda: session.energetics(ws))

    @app.get("/api/records")
    def records(
        stage: str = "",
        job_id: str = "",
        order: str = "time",
        scope: str = "project",
        role: str = "",
    ) -> dict[str, Any]:
        return _run(
            lambda: session.records(
                stage or None, job_id or None, order, scope, role or None
            )
        )

    @app.post("/api/records/{structure_id}/load")
    def load_record(
        structure_id: int, ws: str = "", scope: str = "project"
    ) -> dict[str, Any]:
        return then_state(ws, lambda: session.load_record(ws, structure_id, scope))

    @app.post("/api/records/to-global")
    def add_to_global(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Copy the given project rows to the global database (explicit user action)."""
        return _run(
            lambda: session.add_to_global([int(i) for i in payload.get("ids", [])])
        )

    @app.delete("/api/records/{structure_id}")
    def delete_record(structure_id: int, ws: str = "") -> dict[str, Any]:
        return then_state(ws, lambda: session.delete_record(structure_id))

    @app.get("/api/graph")
    def graph() -> dict[str, Any]:
        return session.graph()

    # ----- frontend -----

    @app.middleware("http")
    async def no_cache(request: Request, call_next):
        """Serve the frontend uncached: it is a handful of small local files, and a
        stale module after an upgrade is worse than a few extra requests."""
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"}
        )

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def _running_app(url: str) -> bool:
    """Whether a GM2-Surf app already answers at ``url``."""
    import json
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{url}/api/meta", timeout=2) as response:  # noqa: S310 - local http URL
            return json.loads(response.read()).get("app") == "gm2_surf"
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _port_free(host: str, port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def run_app(
    project: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    show: bool = True,
    allow_quit: bool = False,
) -> None:
    """Start the app server (blocks until stopped).

    If a GM2-Surf app already runs on ``port``, that one is shown and no
    second server is started. If the port is taken by something else, the
    next free port is used.

    Args:
        project: Project (work) directory holding ``structures.db``, ``jobs/``,
            ``molecules.json``, and ``plugins/``. If omitted, no project is
            opened and nothing is written: the app first asks the user to
            open a recent project or create one in a folder of their choice.
        host: Interface to bind.
        port: Preferred port to serve on.
        show: Open a browser window on start.
        allow_quit: Offer a Quit button in the app that stops this server
            (for the desktop launcher, which has no terminal to press Ctrl-C in).
    """
    import threading
    import webbrowser

    import uvicorn

    for candidate in range(port, port + 50):
        url = f"http://{host}:{candidate}"
        if _port_free(host, candidate):
            port = candidate
            break
        if _running_app(url):
            print(f"GM2-Surf app already running on {url}")  # noqa: T201
            if show:
                webbrowser.open(url)
            return
    else:
        raise RuntimeError(f"no free port between {port} and {port + 49}")

    app = create_app(project)
    url = f"http://{host}:{port}"
    where = (
        app.state.session.project.root
        if app.state.session
        else "none yet, chosen in the app"
    )
    print(f"GM2-Surf app on {url}  (project: {where})")  # noqa: T201
    server = uvicorn.Server(
        uvicorn.Config(app, host=host, port=port, log_level="warning")
    )
    if allow_quit:
        app.state.quit = lambda: setattr(server, "should_exit", True)
    if show:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    server.run()
