"""Predefined cluster settings for the Hosts dialog."""

from __future__ import annotations

import pytest

from gm2_surf.jobs.presets import (
    HOST_PRESETS,
    WORKER_COMMAND,
    host_presets,
    ssh_aliases,
)
from gm2_surf.jobs.remote import RemoteHost

SSH = {
    "vanda": {"hostname": "vanda.nus.edu.sg", "user": "e0123456"},
    "asp2a": {"hostname": "aspire2a.nus.edu.sg", "user": "someone"},
    "github.com": {"hostname": "github.com", "user": "git"},
}


@pytest.fixture
def presets() -> dict[str, dict]:
    return {item["name"]: item for item in host_presets(list(SSH), SSH.__getitem__)}


def test_presets_match_ssh_aliases_and_fill_user_paths(presets):
    vanda, aspire = presets["vanda"], presets["aspire2a"]
    assert (vanda["host"], vanda["configured"]) == ("vanda", True)
    assert vanda["remote_root"] == "/scratch/e0123456/gm2_surf_jobs"
    assert (vanda["queue"], vanda["resources"], vanda["walltime"]) == (
        "auto_free",
        "select=1:ncpus=16:mem=64gb",
        "02:00:00",
    )
    assert (aspire["host"], aspire["queue"]) == (
        "asp2a",
        "normal",
    )  # matched by HostName
    assert aspire["remote_root"] == "~/scratch/gm2_surf_jobs"
    hopper = presets["hopper"]
    assert (
        not hopper["configured"]
        and hopper["host"] == "hopper"
        and "ngpus=1" in hopper["resources"]
    )
    assert (
        hopper["remote_root"] == "/scratch/<user>/gm2_surf_jobs"
    )  # to be filled in by the user


def test_vasp_environment_comes_from_the_registered_module(presets):
    setup = presets["vanda"]["setup"]
    assert setup[:2] == [
        "module use ~/apps/modulefiles",
        "module load vasp/6.6.1-vd-cpu",
    ]
    assert any(
        line.startswith("export ASE_VASP_COMMAND=") and "vasp_std" in line
        for line in setup
    )
    assert "export VASP_PP_PATH=~/apps/vasp" in presets["aspire2a"]["setup"]
    assert presets["hopper"]["setup"] == []  # GPU-only site, no VASP build


def test_every_configured_preset_is_a_valid_host(presets):
    assert set(presets) == set(HOST_PRESETS)
    for item in presets.values():
        if not item["configured"]:
            continue
        host = RemoteHost.from_dict(item)
        assert (
            host.scheduler == "pbs" and host.command == WORKER_COMMAND and host.walltime
        )


def test_ssh_aliases_skip_patterns(tmp_path):
    config = tmp_path / "config"
    config.write_text(
        "Host *\n  User x\nHost vanda hopper  # two\nHost CN*\nhost asp2a\nHostName ignored\n"
    )
    assert ssh_aliases(config) == ["vanda", "hopper", "asp2a"]
    assert ssh_aliases(tmp_path / "missing") == []


def test_api_lists_presets(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from gm2_surf.app.server import create_app
    from gm2_surf.jobs import presets as module

    monkeypatch.setattr(module, "ssh_aliases", lambda: ["vanda"])
    monkeypatch.setattr(module, "resolve_alias", SSH.__getitem__)
    found = (
        TestClient(create_app(tmp_path / "proj"))
        .get("/api/host-presets")
        .json()["presets"]
    )
    assert [item["name"] for item in found] == list(HOST_PRESETS)
