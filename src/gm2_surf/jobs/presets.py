"""Predefined remote-host settings for the clusters this package is used on.

A preset fills the entries of the Hosts dialog (scheduler, directory, queue,
resources, wall time, worker command, environment setup); the user-specific
part (SSH alias, user name, project code) comes from ``~/.ssh/config`` and
the dialog. Site facts follow the sites' public documentation and the
author's cluster notes (checked 2026-11-13): default resources are the
smallest unit the site favours (a whole chunk on Vanda, a quarter node on
ASPIRE 2A), software lives in ``~/apps/<code>``, and VASP builds are
registered as modules ``vasp/<version>-<build>`` under ``~/apps/modulefiles``.

Only PBS sites are listed, because ``SCHEDULERS`` has no Slurm entry yet.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

WORKER_COMMAND = "~/apps/gm2_surf/.venv/bin/gm2_surf run-job"
JOB_DIRECTORY = "gm2_surf_jobs"


@dataclass(frozen=True)
class HostPreset:
    """Defaults for one cluster and the patterns that recognise its SSH alias.

    Attributes:
        name: Preset key and default host name.
        description: One line shown in the dialog.
        hostname: Regular expression searched in the alias's ``HostName``.
        alias: Regular expression searched in the alias itself.
        scratch: Directory that receives the job directory; ``{user}`` is the
            SSH user of the alias.
        queue: Queue to submit to.
        resources: PBS ``-l`` request.
        walltime: Wall-time limit.
        setup: Shell lines run before the worker. Lines starting with ``#``
            are reminders of what the user still has to provide.
        notes: Limits and access notes shown under the form.
    """

    name: str
    description: str
    hostname: str
    alias: str
    scratch: str
    queue: str
    resources: str
    walltime: str = "02:00:00"
    setup: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


def _vasp_setup(module: str, potcars: str) -> tuple[str, ...]:
    """Environment for the ``vasp`` calculator from a VASP module registered under ``~/apps/modulefiles``."""
    return (
        "module use ~/apps/modulefiles",
        f"module load {module}",
        'export ASE_VASP_COMMAND="mpirun -np ${NCPUS:-1} vasp_std"',
        potcars,
    )


HOST_PRESETS: dict[str, HostPreset] = {
    preset.name: preset
    for preset in (
        HostPreset(
            name="vanda",
            description="NUS Vanda (PBS; 2x36 cores/512 GB, GPU 4x A40 48 GB)",
            hostname=r"^vanda",
            alias=r"vanda",
            scratch="/scratch/{user}",
            # Execution queues only take jobs from the routing queues (auto, auto_free, autox).
            queue="auto_free",
            # One whole 36-core chunk: partial chunks are not accepted.
            resources="select=1:ncpus=36:mpiprocs=36:mem=64gb",
            setup=_vasp_setup("vasp/6.6.1-vd-cpu", "export VASP_PP_PATH=~/apps/vasp"),
            notes=(
                "NUS VPN off campus; login nodes are for files and job submission only.",
                "Limits: wall time <= 240 h on CPU nodes, 168 h on GPU nodes; about 8 running jobs.",
                "Allocation unit: a 36-core chunk (2 per node, up to 250 GB each); request whole chunks only, "
                "e.g. select=2:ncpus=36:mpiprocs=36 with place=pack for a full node.",
                "/scratch/<user> is 500 GB; files idle for 60 days may be purged. With a project allocation, "
                "set Remote directory to the project scratch.",
                "VASP 6.6.1 (CPU; DFT-D4, simple-DFT-D3, Libxc, libbeef, VTST, plugins). No GPU VASP.",
            ),
        ),
        HostPreset(
            name="aspire2a",
            description="NSCC ASPIRE 2A (PBS Pro; CPU 2x64 cores/512 GB, GPU 4x A100 40 GB)",
            hostname=r"^aspire2a",
            alias=r"aspire[-_]?2a|^(asp|nscc[-_]?)?2a$|^nscc$",
            scratch="~/scratch",
            queue="normal",
            # A quarter of a 128-core node.
            resources="select=1:ncpus=32:mpiprocs=32:mem=64gb",
            setup=_vasp_setup("vasp/6.6.1-as-cpu", "export VASP_PP_PATH=~/apps/vasp"),
            notes=(
                "NUS VPN off campus. Jobs are rejected without an Account (NSCC project ID, #PBS -P).",
                "CPU nodes have 128 cores: a quarter node (32) for molecules and small structures, a full node "
                "(ncpus=128:mpiprocs=128, higher priority) for larger ones.",
                "Limits: wall time <= 24 h; about 90 running jobs.",
                "GitHub answers 403 on the cluster: copy sources from this Mac.",
                "VASP 6.6.1 CPU (vasp/6.6.1-as-cpu) and GPU (vasp/6.6.1-as-gpu, one A100: ncpus=16:ngpus=1).",
            ),
        ),
        HostPreset(
            name="aspire2b",
            description="NSCC ASPIRE 2B (PBS Pro; CPU 2x96 cores/768 GB, GPU 8x H200 141 GB)",
            hostname=r"^aspire2b",
            alias=r"aspire[-_]?2b|^(asp|nscc[-_]?)?2b$",
            scratch="~",
            queue="normal",
            resources="select=1:ncpus=48:mpiprocs=48",
            notes=(
                "Check Point VPN to vpn2.nscc.sg with DUO first, then keep an SSH connection open.",
                "Jobs are rejected without an Account (NSCC project ID, #PBS -P).",
                "Scratch is /scratch/users/<org>/<institution>/<user>: set Remote directory once known.",
                "GPU jobs: about 24 cores per GPU, e.g. select=1:ncpus=24:ngpus=1. Outbound internet is whitelisted.",
            ),
        ),
        HostPreset(
            name="hopper",
            description="NUS Hopper (PBS; 2x56 cores/2 TB, 8x H100 80 GB or H200 141 GB)",
            hostname=r"^hopper",
            alias=r"hopper",
            scratch="/scratch/{user}",
            queue="",
            resources="select=1:ncpus=14:ngpus=1",
            notes=(
                "Limits: wall time <= 144 h. GPU workloads only; do not send CPU-only jobs.",
                "Set Account to the NUS HPC project allocation code (#PBS -P).",
            ),
        ),
    )
}


def ssh_aliases(config: str | Path = "~/.ssh/config") -> list[str]:
    """Plain ``Host`` aliases of an SSH config file (patterns with ``*``, ``?`` or ``!`` are skipped)."""
    path = Path(config).expanduser()
    if not path.is_file():
        return []
    aliases: list[str] = []
    for line in path.read_text().splitlines():
        words = line.split("#", 1)[0].split()
        if len(words) > 1 and words[0].lower() == "host":
            aliases.extend(word for word in words[1:] if not re.search(r"[*?!]", word))
    return list(dict.fromkeys(aliases))


def resolve_alias(alias: str) -> dict[str, str]:
    """``hostname`` and ``user`` SSH would use for an alias (``ssh -G``; nothing is contacted)."""
    try:
        done = subprocess.run(
            ["ssh", "-G", alias],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    found = {}
    for line in done.stdout.splitlines():
        key, _, value = line.partition(" ")
        if key in ("hostname", "user"):
            found[key] = value.strip()
    return found


def _score(preset: HostPreset, alias: str, hostname: str) -> int | None:
    """Match quality (lower is better): alias equals the name, HostName matches, alias matches."""
    if alias.lower() == preset.name:
        return 0
    if hostname and re.search(preset.hostname, hostname, re.IGNORECASE):
        return 1
    if re.search(preset.alias, alias, re.IGNORECASE):
        return 2
    return None


def host_presets(
    aliases: list[str] | None = None,
    resolve: Callable[[str], dict[str, str]] = resolve_alias,
) -> list[dict[str, Any]]:
    """Presets as ready-to-fill host settings, matched to the user's SSH aliases.

    Args:
        aliases: SSH aliases to match (those of ``~/.ssh/config`` by default).
        resolve: Returns ``hostname`` and ``user`` for an alias.

    Returns:
        One dict per preset with the ``RemoteHost`` fields, plus ``description``,
        ``notes`` and ``configured`` (whether an SSH alias for it was found).
    """
    resolved = {
        alias: resolve(alias)
        for alias in (ssh_aliases() if aliases is None else aliases)
    }
    items = []
    for preset in HOST_PRESETS.values():
        scored = [
            (score, index, alias)
            for index, (alias, info) in enumerate(resolved.items())
            if (score := _score(preset, alias, info.get("hostname", ""))) is not None
        ]
        alias = min(scored)[2] if scored else ""
        user = resolved.get(alias, {}).get("user", "")
        scratch = preset.scratch.format(user=user or "<user>")
        items.append(
            {
                "name": preset.name,
                "host": alias or preset.name,
                "scheduler": "pbs",
                "remote_root": f"{scratch}/{JOB_DIRECTORY}",
                "command": WORKER_COMMAND,
                "setup": list(preset.setup),
                "queue": preset.queue,
                "account": "",
                "resources": preset.resources,
                "walltime": preset.walltime,
                "directives": [],
                "description": preset.description,
                "notes": list(preset.notes),
                "configured": bool(alias),
            }
        )
    return items
