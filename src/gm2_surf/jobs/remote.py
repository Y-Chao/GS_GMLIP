"""Remote execution: run a job on another machine over SSH, through a scheduler.

A remote job is a self-contained mini project on the remote machine::

    <remote_root>/<job_id>/structures.db          input structure (row 1), then results
    <remote_root>/<job_id>/jobs/<job_id>/job.json
    <remote_root>/<job_id>/plugins/*.py           copies of the project plugins
    <remote_root>/<job_id>/models/...             user-trained model, if one is used
    <remote_root>/<job_id>/run.sh                 scheduler script

so the worker there is the same ``gm2_surf run-job`` used locally and never
needs to see the local project. Each synchronisation is one SSH round trip:
it asks the scheduler whether the job is still alive and streams back the
status, the log, and the results database. New rows are merged into the local
project store with their provenance (``parent_id``) remapped, so a remote job
can be followed live exactly like a local one.

Hosts are user-level settings in ``<gm2_home>/hosts.json``. SSH must work
without a prompt (key, agent, or an open ControlMaster connection).
"""

from __future__ import annotations

import io
import json
import re
import shlex
import subprocess
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Protocol

from gm2_surf.config import gm2_home
from gm2_surf.interface.core import Interface
from gm2_surf.jobs.executor import BaseExecutor, LocalExecutor
from gm2_surf.jobs.spec import (
    LOG_FILE,
    SPEC_FILE,
    STATUS_FILE,
    JobSpec,
    JobStatus,
    read_status,
    write_status,
)
from gm2_surf.store import Project, StructureStore

HOSTS_FILE = "hosts.json"
REMOTE_FILE = "remote.json"
_QUEUE_FILE = "queue"
_SYNC_DIR = ".sync"
_DB_FILE = "structures.db"
_SAFE_PATH = re.compile(r"^[A-Za-z0-9_./~$-]+$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


# ----------------------------------------------------------------------
# Schedulers
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Scheduler:
    """How to talk to a queueing system. Commands are ``str.format`` templates.

    Attributes:
        name: Registry key.
        submit: Command submitting ``{script}``; its output holds the job id.
        alive: Command that succeeds while job ``{id}`` is queued or running.
        cancel: Command that removes job ``{id}``.
        id_pattern: Regular expression whose first match in the submit output
            is the scheduler's job id.
    """

    name: str
    submit: str
    alive: str
    cancel: str
    id_pattern: str = r"\S+"

    def header(self, host: RemoteHost, job_id: str, directory: str) -> list[str]:
        """Scheduler directives for the top of the job script."""
        return []


class _PBS(Scheduler):
    def header(self, host: RemoteHost, job_id: str, directory: str) -> list[str]:
        lines = [
            f"#PBS -N {job_id[:15]}",
            "#PBS -j oe",
            f"#PBS -o {directory}/scheduler.log",
        ]
        if host.queue:
            lines.append(f"#PBS -q {host.queue}")
        if host.account:
            lines.append(f"#PBS -P {host.account}")
        if host.resources:
            lines.append(f"#PBS -l {host.resources}")
        if host.walltime:
            lines.append(f"#PBS -l walltime={host.walltime}")
        return lines + [f"#PBS {line}" for line in host.directives]


SCHEDULERS: dict[str, Scheduler] = {
    "pbs": _PBS(
        name="pbs", submit="qsub {script}", alive="qstat {id}", cancel="qdel {id}"
    ),
    # No queue: start the script directly on the remote machine (a workstation).
    "shell": Scheduler(
        name="shell",
        submit="nohup bash {script} > /dev/null 2>&1 < /dev/null & echo $!",
        alive="kill -0 {id}",
        cancel="kill {id}",
        id_pattern=r"\d+",
    ),
}


# ----------------------------------------------------------------------
# Hosts
# ----------------------------------------------------------------------


@dataclass
class RemoteHost:
    """A machine jobs can be sent to.

    Attributes:
        name: Name shown in the app.
        host: SSH destination (an alias from ``~/.ssh/config`` or ``user@host``).
        scheduler: Key of ``SCHEDULERS`` (``pbs`` or ``shell``).
        remote_root: Remote directory that receives one sub-directory per job.
        command: Worker command; the job directory is appended.
        setup: Shell lines run before the worker (module loads, environment).
        queue: Queue name.
        account: Project or account to charge.
        resources: Resource request, e.g. ``select=1:ncpus=8:ngpus=1:mem=32gb``.
        walltime: Wall-time limit, e.g. ``24:00:00``.
        directives: Further scheduler directives, without the ``#PBS`` prefix.
        sync_interval: Minimum seconds between two synchronisations of a job.
        grace: Seconds a job may be gone from the queue without a final status
            before it is declared failed. Shared file systems can show the
            login node a compute node's last writes with a delay.
    """

    name: str
    host: str
    scheduler: str = "pbs"
    remote_root: str = "~/gm2_surf_jobs"
    command: str = "gm2_surf run-job"
    setup: list[str] = field(default_factory=list)
    queue: str = ""
    account: str = ""
    resources: str = ""
    walltime: str = ""
    directives: list[str] = field(default_factory=list)
    sync_interval: float = 15.0
    grace: float = 60.0

    def __post_init__(self) -> None:
        if not _SAFE_NAME.match(self.name):
            raise ValueError(
                f"host name {self.name!r} may only contain letters, digits, '.', '_' and '-'"
            )
        if (
            not self.host
            or self.host.startswith("-")
            or any(char.isspace() for char in self.host)
        ):
            raise ValueError(f"invalid SSH destination {self.host!r}")
        if self.scheduler not in SCHEDULERS:
            raise ValueError(
                f"unknown scheduler {self.scheduler!r}; choose from {sorted(SCHEDULERS)}"
            )
        if not _SAFE_PATH.match(self.remote_root):
            raise ValueError(
                f"remote directory {self.remote_root!r} may not contain spaces or shell characters"
            )
        for name in ("queue", "account", "resources", "walltime"):
            if "\n" in getattr(self, name):
                raise ValueError(f"{name} must be a single line")
        self.setup = [str(line) for line in self.setup if str(line).strip()]
        self.directives = [
            str(line).strip() for line in self.directives if str(line).strip()
        ]
        self.sync_interval = float(self.sync_interval)
        self.grace = float(self.grace)

    @property
    def root(self) -> str:
        """Remote root with a leading ``~`` written as ``$HOME`` (safe inside a script)."""
        if self.remote_root == "~" or self.remote_root.startswith("~/"):
            return "$HOME" + self.remote_root[1:]
        return self.remote_root

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RemoteHost:
        """Inverse of ``to_dict``; unknown keys are ignored."""
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


def load_hosts() -> dict[str, RemoteHost]:
    """Configured remote hosts by name."""
    path = gm2_home() / HOSTS_FILE
    if not path.exists():
        return {}
    hosts = {}
    for entry in json.loads(path.read_text()):
        host = RemoteHost.from_dict(entry)
        hosts[host.name] = host
    return hosts


def _write_hosts(hosts: dict[str, RemoteHost]) -> None:
    (gm2_home() / HOSTS_FILE).write_text(
        json.dumps([host.to_dict() for host in hosts.values()], indent=2)
    )


def save_host(host: RemoteHost) -> None:
    """Add a host, or replace the one with the same name."""
    hosts = load_hosts()
    hosts[host.name] = host
    _write_hosts(hosts)


def remove_host(name: str) -> None:
    """Forget a host. Jobs already sent to it stay reachable."""
    hosts = load_hosts()
    hosts.pop(name, None)
    _write_hosts(hosts)


# ----------------------------------------------------------------------
# Transports
# ----------------------------------------------------------------------


class RemoteError(RuntimeError):
    """A remote command failed or the machine could not be reached."""


class Transport(Protocol):
    """Runs a shell command on the remote machine."""

    def run(
        self, command: str, stdin: bytes | None = None, timeout: float = 120.0
    ) -> bytes:
        """Run ``command`` and return its standard output; raise ``RemoteError`` on failure."""


def _execute(argv: list[str], stdin: bytes | None, timeout: float, where: str) -> bytes:
    try:
        done = subprocess.run(
            argv, input=stdin, capture_output=True, timeout=timeout, check=False
        )  # noqa: S603
    except subprocess.TimeoutExpired as error:
        raise RemoteError(f"{where}: no answer within {timeout:.0f} s") from error
    except OSError as error:
        raise RemoteError(f"{where}: {error}") from error
    if done.returncode != 0:
        detail = done.stderr.decode(errors="replace").strip().splitlines()
        raise RemoteError(
            f"{where}: {detail[-1] if detail else f'exit code {done.returncode}'}"
        )
    return done.stdout


class SSHTransport:
    """Runs commands through the system ``ssh``, never prompting for a password."""

    def __init__(self, host: str):
        self.host = host

    def run(
        self, command: str, stdin: bytes | None = None, timeout: float = 120.0
    ) -> bytes:
        """Run ``command`` with bash on the host."""
        argv = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            self.host,
            f"bash -c {shlex.quote(command)}",
        ]
        return _execute(argv, stdin, timeout, self.host)


class LocalTransport:
    """Runs the "remote" commands on this machine (tests, or a second directory tree)."""

    def run(
        self, command: str, stdin: bytes | None = None, timeout: float = 120.0
    ) -> bytes:
        """Run ``command`` with bash."""
        return _execute(["bash", "-c", command], stdin, timeout, "local shell")


# ----------------------------------------------------------------------
# Executor
# ----------------------------------------------------------------------


def _tar(files: dict[str, Path | bytes]) -> bytes:
    """A tar archive of ``{archive name: file, directory, or content}``."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, source in files.items():
            if isinstance(source, bytes):
                info = tarfile.TarInfo(name)
                info.size = len(source)
                info.mtime = int(time.time())
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(source))
            else:
                archive.add(str(source), arcname=name)
    return buffer.getvalue()


def _record_kwargs(record) -> dict[str, Any]:
    return {
        "stage": record.stage,
        "method": record.method,
        "calculator": record.calculator,
        "job_id": record.job_id,
        "step": record.step,
        "label": record.label,
        "score": record.score,
        "role": record.role,
        "level": record.level,
        "tags": record.tags,
    }


class RemoteExecutor(BaseExecutor):
    """Sends jobs to a ``RemoteHost`` and merges their results into the project.

    Args:
        project: Local project.
        host: Target machine for new jobs. Existing jobs are followed with the
            host settings they were submitted with.
        transport: How commands reach the machine; SSH to ``host.host`` by default.
    """

    def __init__(
        self,
        project: Project,
        host: RemoteHost | None = None,
        transport: Transport | None = None,
    ):
        super().__init__(project)
        self.host = host
        self._transport = transport
        self._last_sync: dict[str, float] = {}

    def _connection(self, host: RemoteHost) -> Transport:
        return self._transport or SSHTransport(host.host)

    # ----- bookkeeping -----

    def _info(self, job_id: str) -> dict[str, Any]:
        return json.loads((self.job_dir(job_id) / REMOTE_FILE).read_text())

    def _write_info(self, job_id: str, info: dict[str, Any]) -> None:
        (self.job_dir(job_id) / REMOTE_FILE).write_text(json.dumps(info, indent=2))

    @staticmethod
    def is_remote(job_dir: Path) -> bool:
        """Whether the job in ``job_dir`` was sent to a remote host."""
        return (job_dir / REMOTE_FILE).exists()

    # ----- submission -----

    def _script(self, host: RemoteHost, job_id: str, directory: str) -> str:
        job_path = f"{directory}/jobs/{job_id}"
        lines = [
            "#!/bin/bash",
            *SCHEDULERS[host.scheduler].header(host, job_id, directory),
        ]
        lines += [f'cd "{directory}" || exit 1', *host.setup]
        lines.append(f'exec {host.command} "{job_path}" > "{job_path}/{LOG_FILE}" 2>&1')
        return "\n".join(lines) + "\n"

    def _upload_model(
        self, spec: JobSpec, files: dict[str, Path | bytes], directory: str
    ) -> JobSpec:
        """Ship a user-trained model that exists on this machine and point the remote spec at it."""
        model_path = str(spec.calculator_params.get("model_path") or "")
        local = Path(model_path).expanduser() if model_path else None
        if local is None or not local.exists():
            return spec  # no custom model, or a path that is meant to exist on the remote machine
        files[f"models/{local.name}"] = local
        params = {
            **spec.calculator_params,
            "model_path": f"{directory}/models/{local.name}",
        }
        return JobSpec.from_dict({**spec.to_dict(), "calculator_params": params})

    def submit(self, spec: JobSpec) -> str:
        """Upload the job and hand it to the scheduler. Returns the job id."""
        if self.host is None:
            raise ValueError("no remote host selected")
        host = self.host
        job_dir = self.prepare(spec)
        job_id = spec.job_id
        connection = self._connection(host)
        try:
            # The script is written with $HOME; everything else needs the real path.
            directory = connection.run(
                f'mkdir -p "{host.root}/{job_id}" && cd "{host.root}/{job_id}" && pwd'
            )
            directory = directory.decode().strip().splitlines()[-1]

            files: dict[str, Path | bytes] = {}
            remote_spec = JobSpec.from_dict({**spec.to_dict(), "structure_id": 1})
            remote_spec = self._upload_model(remote_spec, files, directory)
            with tempfile.TemporaryDirectory() as scratch:
                seed = StructureStore(Path(scratch) / _DB_FILE)
                record = self.project.store.record(spec.structure_id)
                seed.add(
                    self.project.store.get(spec.structure_id), **_record_kwargs(record)
                )
                files[_DB_FILE] = seed.path.read_bytes()
            files[f"jobs/{job_id}/{SPEC_FILE}"] = json.dumps(
                remote_spec.to_dict(), indent=2
            ).encode()
            files[f"jobs/{job_id}/{STATUS_FILE}"] = json.dumps(
                asdict(JobStatus())
            ).encode()
            files["run.sh"] = self._script(host, job_id, directory).encode()
            plugins = self.project.root / "plugins"
            for plugin in sorted(plugins.glob("*.py")) if plugins.is_dir() else ():
                files[f"plugins/{plugin.name}"] = plugin
            connection.run(
                f'tar -xf - -C "{directory}"', stdin=_tar(files), timeout=600
            )

            scheduler = SCHEDULERS[host.scheduler]
            output = connection.run(
                f'cd "{directory}" || exit 1; {scheduler.submit.format(script="run.sh")}'
            ).decode()
            match = re.search(scheduler.id_pattern, output.strip())
            if match is None:
                raise RemoteError(
                    f"{host.name}: could not read a job id from {output.strip()!r}"
                )
        except RemoteError as error:
            status = read_status(job_dir)
            status.state = "failed"
            status.message = f"submission failed: {error}"
            status.finished = time.time()
            write_status(job_dir, status)
            self._write_info(
                job_id,
                {
                    "host": host.to_dict(),
                    "directory": "",
                    "scheduler_id": "",
                    "id_map": {},
                },
            )
            raise

        info = {
            "host": host.to_dict(),
            "directory": directory,
            "scheduler_id": match.group(0),
            "id_map": {"1": spec.structure_id},
        }
        self._write_info(job_id, info)
        status = read_status(job_dir)
        status.message = f"queued on {host.name} as {info['scheduler_id']}"
        write_status(job_dir, status)
        return job_id

    # ----- synchronisation -----

    def _merge(self, job_id: str, info: dict[str, Any], database: Path) -> int | None:
        """Copy rows not seen before into the project store. Returns the newest local id."""
        id_map: dict[str, int] = info["id_map"]
        remote = StructureStore(database)
        newest = None
        for record in remote.records():
            if str(record.id) in id_map:
                continue
            interface: Interface = remote.get(record.id)
            parent = (
                id_map.get(str(record.parent_id))
                if record.parent_id is not None
                else None
            )
            newest = self.project.store.add(
                interface, parent_id=parent, **_record_kwargs(record)
            )
            id_map[str(record.id)] = newest
            self._write_info(
                job_id, info
            )  # after every row: a crash must not duplicate rows
        return newest

    def sync(self, job_id: str) -> JobStatus:
        """Fetch status, log, and new structures of a job now."""
        job_dir = self.job_dir(job_id)
        info = self._info(job_id)
        status = read_status(job_dir)
        if not status.active or not info["directory"]:
            return status
        host = RemoteHost.from_dict(info["host"])
        scheduler = SCHEDULERS[host.scheduler]
        # Snapshot first: tar must not read a file while the worker replaces or extends it.
        job_path = f"jobs/{job_id}"
        command = (
            f'cd "{info["directory"]}" && mkdir -p {_SYNC_DIR} && '
            f"if {scheduler.alive.format(id=info['scheduler_id'])} > /dev/null 2>&1; "
            f"then echo alive > {_SYNC_DIR}/{_QUEUE_FILE}; else echo gone > {_SYNC_DIR}/{_QUEUE_FILE}; fi; "
            f"cp -f {_DB_FILE} {job_path}/{STATUS_FILE} {job_path}/{LOG_FILE} {_SYNC_DIR}/ 2> /dev/null; "
            f"tar -cf - -C {_SYNC_DIR} ."
        )
        self._last_sync[job_id] = time.time()
        try:
            payload = self._connection(host).run(command, timeout=600)
        except RemoteError as error:
            status.message = f"cannot reach {host.name}: {error}"
            write_status(job_dir, status)
            return status

        with tempfile.TemporaryDirectory() as scratch:
            with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
                archive.extractall(scratch, filter="data")
            fetched = Path(scratch)
            alive = (fetched / _QUEUE_FILE).read_text().strip() == "alive"
            if (fetched / LOG_FILE).exists():
                (job_dir / LOG_FILE).write_bytes((fetched / LOG_FILE).read_bytes())
            newest = None
            # A copy taken while the worker was writing is unusable; the next sync retries.
            incomplete = False
            if (fetched / _DB_FILE).exists():
                try:
                    newest = self._merge(job_id, info, fetched / _DB_FILE)
                except Exception:  # noqa: BLE001 - any read error means a partial copy
                    incomplete = True
            reported = None
            if (fetched / STATUS_FILE).exists():
                try:
                    reported = read_status(fetched)
                except (ValueError, TypeError):
                    incomplete = True

        local_ids = [value for key, value in info["id_map"].items() if key != "1"]
        status.n_structures = len(local_ids)
        if newest is not None or local_ids:
            status.last_structure_id = newest if newest is not None else max(local_ids)
        if reported is not None and reported.state != "queued":
            status.started = reported.started
            status.progress = reported.progress
            status.message = reported.message
            if reported.active:
                status.state = "running"
            elif not incomplete:
                status.state = reported.state
                status.finished = reported.finished
        if status.active and not alive:
            # Gone from the queue without a final report. The report may only be late
            # (file-system lag, or a results copy taken mid-write), so wait before
            # concluding that the job was killed, hit the wall time, or never started.
            gone_since = info.setdefault("gone_since", time.time())
            self._write_info(job_id, info)
            if time.time() - gone_since >= host.grace and not incomplete:
                status.state = "failed"
                status.message = (
                    f"job left the queue on {host.name} without reporting; see log"
                )
                status.finished = time.time()
            else:
                status.message = (
                    f"left the queue on {host.name}; waiting for its final status"
                )
        elif info.pop("gone_since", None) is not None:
            self._write_info(job_id, info)
        write_status(job_dir, status)
        return status

    def status(self, job_id: str) -> JobStatus:
        """Status of a job, synchronised at most every ``sync_interval`` seconds."""
        job_dir = self.job_dir(job_id)
        status = read_status(job_dir)
        if not status.active:
            return status
        interval = float(self._info(job_id)["host"].get("sync_interval", 15.0))
        if time.time() - self._last_sync.get(job_id, 0.0) < interval:
            return status
        return self.sync(job_id)

    def cancel(self, job_id: str) -> None:
        """Remove the job from the queue, keeping the structures it has written."""
        status = self.sync(job_id)
        if not status.active:
            return
        info = self._info(job_id)
        host = RemoteHost.from_dict(info["host"])
        self._connection(host).run(
            SCHEDULERS[host.scheduler].cancel.format(id=info["scheduler_id"])
        )
        status.state = "cancelled"
        status.message = "cancelled by user"
        status.finished = time.time()
        write_status(self.job_dir(job_id), status)


def check_host(host: RemoteHost, transport: Transport | None = None) -> dict[str, Any]:
    """Check that a host can be reached and has what a job needs.

    Returns:
        ``{"ok": bool, "checks": [{"name", "ok", "detail"}, ...]}``.
    """
    connection = transport or SSHTransport(host.host)
    scheduler = SCHEDULERS[host.scheduler]
    programs = [scheduler.submit.split()[0], host.command.split()[0]]
    setup = "\n".join(host.setup)
    probes = "; ".join(
        f'command -v {program} > /dev/null 2>&1 && echo "ok {program}" || echo "missing {program}"'
        for program in programs
    )
    try:
        output = connection.run(
            f'{setup}\nmkdir -p "{host.root}" && echo "ok directory"; {probes}',
            timeout=60,
        )
    except RemoteError as error:
        return {
            "ok": False,
            "checks": [{"name": "connection", "ok": False, "detail": str(error)}],
        }
    found = {}
    for line in output.decode().splitlines():
        state, _, key = line.partition(" ")
        if state in ("ok", "missing"):
            found[key] = state
    checks = [{"name": "connection", "ok": True, "detail": host.host}]
    labels = {
        "directory": host.remote_root,
        programs[0]: f"{host.scheduler} scheduler",
        programs[1]: "worker command",
    }
    for key, detail in labels.items():
        ok = found.get(key) == "ok"
        checks.append(
            {"name": key, "ok": ok, "detail": detail if ok else f"{detail}: not found"}
        )
    return {"ok": all(check["ok"] for check in checks), "checks": checks}


class ProjectExecutor(BaseExecutor):
    """Routes each job to where it runs: this machine, or the remote host it was sent to.

    Args:
        project: Local project.
        transport: Transport for all remote hosts (tests); SSH by default.
    """

    def __init__(self, project: Project, transport: Transport | None = None):
        super().__init__(project)
        self.local = LocalExecutor(project)
        self.remote = RemoteExecutor(project, transport=transport)
        self._transport = transport

    def _for(self, job_id: str) -> BaseExecutor:
        return (
            self.remote
            if RemoteExecutor.is_remote(self.job_dir(job_id))
            else self.local
        )

    def submit(self, spec: JobSpec, host: str = "") -> str:
        """Start a job locally, or on the configured host named ``host``."""
        if not host:
            return self.local.submit(spec)
        hosts = load_hosts()
        if host not in hosts:
            raise ValueError(f"unknown host {host!r}; configured: {sorted(hosts)}")
        return RemoteExecutor(self.project, hosts[host], self._transport).submit(spec)

    def status(self, job_id: str) -> JobStatus:
        """Current status of a job."""
        return self._for(job_id).status(job_id)

    def cancel(self, job_id: str) -> None:
        """Stop a job."""
        self._for(job_id).cancel(job_id)

    def where(self, job_id: str) -> str:
        """Name of the host a job runs on, empty for this machine."""
        job_dir = self.job_dir(job_id)
        if not RemoteExecutor.is_remote(job_dir):
            return ""
        return str(
            json.loads((job_dir / REMOTE_FILE).read_text())["host"].get("name", "")
        )


__all__ = [
    "SCHEDULERS",
    "LocalTransport",
    "ProjectExecutor",
    "RemoteError",
    "RemoteExecutor",
    "RemoteHost",
    "SSHTransport",
    "Scheduler",
    "Transport",
    "check_host",
    "load_hosts",
    "remove_host",
    "save_host",
]
