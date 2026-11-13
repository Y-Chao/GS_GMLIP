"""Tests for the session layer and the JSON API behind the JavaScript frontend."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from gs_gmlip.app.server import STATIC_DIR, create_app  # noqa: E402
from gs_gmlip.app.session import Session  # noqa: E402
from gs_gmlip.interface.operations import describe_constraints as ops_constraints  # noqa: E402
from gs_gmlip.store import Project  # noqa: E402

BULK, SURFACE, MOLECULE = "bulk-1", "surface-1", "molecule-1"


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(tmp_path / "proj"))


def post(client, path, ws, json=None):
    return client.post(path, params={"ws": ws}, json=json or {})


@pytest.fixture
def slab_client(client):
    """Client with fcc Cu in the bulk window and a 2x2 Cu(111) slab in the surface window."""
    assert (
        post(
            client,
            "/api/bulk/create",
            BULK,
            {"element": "Cu", "lattice": "fcc", "a": 3.61},
        ).status_code
        == 200
    )
    response = post(
        client,
        "/api/cleave",
        BULK,
        {
            "miller": [1, 1, 1],
            "layers": 3,
            "vacuum": 24.0,
            "supercell": [2, 2],
            "fix_layers": 1,
            "target": SURFACE,
        },
    )
    assert response.status_code == 200
    assert response.json()["workspace"]["id"] == SURFACE
    return client


def _wait(client, timeout=120.0):
    deadline = time.time() + timeout
    while any(job["active"] for job in client.get("/api/jobs").json()["jobs"]):
        assert time.time() < deadline, "job did not finish in time"
        time.sleep(0.2)
    return client.get("/api/jobs").json()["jobs"]


def test_frontend_files_are_served_uncached(client):
    response = client.get("/")
    assert response.status_code == 200 and "GS-GMLIP" in response.text
    for name in ("app.js", "viewer.js", "graph.js", "style.css"):
        assert (STATIC_DIR / name).is_file()
        served = client.get(f"/static/{name}")
        assert served.status_code == 200
        assert served.headers["cache-control"] == "no-store"


def test_meta_lists_choices(client):
    meta = client.get("/api/meta").json()
    assert "fcc" in meta["lattices"] and "CH3OH" in meta["molecule_names"]
    assert meta["window_kinds"] == ["bulk", "surface", "molecule"]
    assert meta["reactions"] == ["HER", "OER", "ORR", "CO2RR", "CORR"]
    assert meta["library_scopes"] == ["temporary", "project", "global"]
    assert not meta["ask_project"]
    assert {task["kind"] for task in meta["tasks"]} >= {
        "relax",
        "cell_relax",
        "nvt",
        "npt",
        "ga",
        "gcmc",
        "vasp_inputs",
    }
    calculators = {entry["name"]: entry for entry in meta["calculators"]}
    assert calculators["emt"]["available"]
    assert (
        calculators["matgl"]["defaults"]["model"]
        == "TensorNet-PES-MatPES-r2SCAN-2025.2"
    )
    assert {preset["name"] for preset in meta["vasp"]["presets"]} >= {
        "static-r2SCAN",
        "relax-PBE",
    }


def test_windows_open_and_close(client):
    listing = client.get("/api/windows").json()["windows"]
    assert [(w["id"], w["kind"], w["formula"]) for w in listing] == [
        (BULK, "bulk", None),
        (SURFACE, "surface", None),
        (MOLECULE, "molecule", None),
    ]
    state = client.post(
        "/api/windows", json={"kind": "molecule", "name": "Probe"}
    ).json()
    assert state["workspace"] == {
        "id": "molecule-2",
        "name": "Probe",
        "kind": "molecule",
    }
    assert len(state["windows"]) == 4
    assert client.post("/api/windows", json={"kind": "nope"}).status_code == 400
    remaining = client.delete("/api/windows/molecule-2").json()["windows"]
    assert [w["id"] for w in remaining] == [BULK, SURFACE, MOLECULE]
    assert client.delete("/api/windows/molecule-2").status_code == 400
    client.delete(f"/api/windows/{SURFACE}")
    client.delete(f"/api/windows/{MOLECULE}")
    assert (
        client.delete(f"/api/windows/{BULK}").status_code == 400
    )  # the last window stays
    assert client.get("/api/state", params={"ws": "nope"}).status_code == 400


def test_empty_window_and_errors(client):
    state = client.get("/api/state", params={"ws": SURFACE}).json()
    assert state["scene"] is None and state["info"] is None and not state["can_undo"]
    response = post(client, "/api/edit", SURFACE, {"op": "delete", "indices": [0]})
    assert (
        response.status_code == 400 and "window is empty" in response.json()["detail"]
    )
    assert post(client, "/api/cleave", BULK, {"miller": [1, 1, 1]}).status_code == 400
    assert post(client, "/api/undo", SURFACE).status_code == 400


def test_windows_hold_independent_structures(slab_client):
    client = slab_client
    bulk = client.get("/api/state", params={"ws": BULK}).json()
    surface = client.get("/api/state", params={"ws": SURFACE}).json()
    assert bulk["info"]["natoms"] == 4 and surface["info"]["natoms"] == 48
    post(
        client,
        "/api/edit",
        SURFACE,
        {"op": "substitute", "indices": [0], "element": "Pt"},
    )
    assert (
        client.get("/api/state", params={"ws": BULK}).json()["info"]["formula"] == "Cu4"
    )
    assert not client.get("/api/state", params={"ws": BULK}).json()["can_undo"]
    # Cleaving without a target opens a new, named surface window.
    state = post(client, "/api/cleave", BULK, {"miller": [1, 0, 0], "layers": 2}).json()
    assert state["workspace"] == {
        "id": "surface-2",
        "name": "Cu(100)",
        "kind": "surface",
    }
    assert [w["id"] for w in state["windows"]] == [BULK, SURFACE, MOLECULE, "surface-2"]


def test_build_edit_undo_roundtrip(slab_client):
    client = slab_client
    state = client.get("/api/state", params={"ws": SURFACE}).json()
    n_slab = state["info"]["natoms"]
    assert state["info"]["n_fixed"] == n_slab // 3
    assert state["scene"]["sites"] == []

    state = post(client, "/api/sites", SURFACE, {"show": True, "height": 1.8}).json()
    assert len(state["scene"]["sites"]) > 0
    assert state["sites"] == {"show": True, "distinct": False, "height": 1.8}
    # Sites are a surface-window feature.
    assert (
        post(client, "/api/sites", BULK, {"show": True}).json()["scene"]["sites"] == []
    )

    state = post(
        client, "/api/edit", SURFACE, {"op": "place", "site": 0, "adsorbate": "CO"}
    ).json()
    assert state["info"]["natoms"] == n_slab + 2 and state["info"]["n_adsorbates"] == 1
    pair = [n_slab, n_slab + 1]
    before = {atom["index"]: atom for atom in state["scene"]["atoms"]}

    state = post(
        client,
        "/api/edit",
        SURFACE,
        {"op": "move", "indices": pair, "vector": [0, 0, 0.5]},
    ).json()
    assert state["scene"]["atoms"][n_slab]["z"] == pytest.approx(
        before[n_slab]["z"] + 0.5
    )

    state = post(
        client,
        "/api/edit",
        SURFACE,
        {"op": "rotate", "indices": pair, "angle": 90, "axis": [1, 0, 0]},
    ).json()
    first, second = (state["scene"]["atoms"][i] for i in pair)
    assert first["z"] == pytest.approx(
        second["z"], abs=1e-6
    )  # bond now lies in the surface plane

    post(client, "/api/edit", SURFACE, {"op": "bond", "indices": pair})
    post(client, "/api/edit", SURFACE, {"op": "com", "indices": pair})
    state = post(
        client, "/api/edit", SURFACE, {"op": "wall", "indices": pair, "height": 30.0}
    ).json()
    info = state["info"]
    assert (info["n_bond_constraints"], info["n_com_constraints"], info["n_walls"]) == (
        1,
        1,
        2,
    )
    assert state["scene"]["constraints"]["bonds"] == [pair]

    state = post(
        client,
        "/api/edit",
        SURFACE,
        {"op": "substitute", "indices": [0], "element": "Pt"},
    ).json()
    assert state["scene"]["atoms"][0]["elem"] == "Pt"
    state = post(client, "/api/edit", SURFACE, {"op": "release", "indices": [0]}).json()
    assert not state["scene"]["atoms"][0]["fixed"]
    state = post(client, "/api/edit", SURFACE, {"op": "clear_constraints"}).json()
    assert state["info"]["n_bond_constraints"] == 0

    assert state["can_undo"]
    state = post(client, "/api/undo", SURFACE).json()
    assert state["info"]["n_bond_constraints"] == 1
    state = post(client, "/api/edit", SURFACE, {"op": "delete", "indices": pair}).json()
    assert state["info"]["natoms"] == n_slab

    for bad in (
        {"op": "bond", "indices": [1]},
        {"op": "nope"},
        {"op": "move", "indices": [], "vector": [0, 0, 1]},
    ):
        assert post(client, "/api/edit", SURFACE, bad).status_code == 400


def test_items_can_be_switched_off_and_removed(slab_client):
    client = slab_client
    post(client, "/api/sites", SURFACE, {"show": True})
    post(client, "/api/edit", SURFACE, {"op": "place", "site": 0, "adsorbate": "CO"})
    state = post(
        client, "/api/edit", SURFACE, {"op": "place", "site": 30, "adsorbate": "OH"}
    ).json()
    n_slab = 48
    post(client, "/api/edit", SURFACE, {"op": "bond", "indices": [n_slab, n_slab + 1]})
    state = post(
        client,
        "/api/edit",
        SURFACE,
        {"op": "wall", "indices": [n_slab + 3], "height": 30.0},
    ).json()
    assert [(item["key"], item["checked"]) for item in state["items"]] == [
        ("substrate", True), ("fixed", True), ("ads:0", True), ("ads:1", True), ("bond:0", True), ("wall:0", True),
    ]  # fmt: skip
    assert state["role"] == "adsorbed"

    state = post(
        client, "/api/items", SURFACE, {"key": "ads:1", "checked": False}
    ).json()
    state = post(
        client, "/api/items", SURFACE, {"key": "bond:0", "checked": False}
    ).json()
    assert [item["checked"] for item in state["items"]] == [
        True,
        True,
        True,
        False,
        False,
        True,
    ]
    assert [atom["excluded"] for atom in state["scene"]["atoms"][n_slab:]] == [
        False,
        False,
        True,
        True,
    ]
    assert (
        state["scene"]["constraints"]["bonds"] == []
    )  # switched-off constraints are not drawn
    assert state["info"]["natoms"] == n_slab + 4  # the window still holds everything
    assert (
        post(
            client, "/api/items", SURFACE, {"key": "substrate", "checked": False}
        ).status_code
        == 400
    )
    assert (
        post(
            client, "/api/items", SURFACE, {"key": "ads:9", "checked": False}
        ).status_code
        == 400
    )

    # Only the checked items leave the window.
    exported = client.get(
        "/api/export", params={"ws": SURFACE, "format": "extxyz"}
    ).text
    assert int(exported.splitlines()[0]) == n_slab + 2
    state = post(client, "/api/save", SURFACE).json()
    assert (
        state["current_id"] is None
    )  # the window (with unchecked items) differs from the stored row
    stored = client.app.state.session.project.store
    record = stored.records()[-1]
    assert (record.natoms, record.role) == (n_slab + 2, "adsorbed")
    assert ops_constraints(stored.get(record.id))["bonds"] == []

    state = post(
        client, "/api/items", SURFACE, {"key": "ads:1", "checked": True}
    ).json()
    assert not any(atom["excluded"] for atom in state["scene"]["atoms"])
    state = post(
        client, "/api/edit", SURFACE, {"op": "remove_item", "key": "ads:0"}
    ).json()
    assert state["info"]["natoms"] == n_slab + 2
    assert [item["key"] for item in state["items"]] == [
        "substrate",
        "fixed",
        "ads:0",
        "wall:0",
    ]
    assert post(client, "/api/undo", SURFACE).json()["info"]["natoms"] == n_slab + 4


def test_global_references_load_and_drive_energetics(client):
    from ase.build import bulk as ase_bulk
    from ase.calculators.emt import EMT

    from gs_gmlip.importers import to_interface
    from gs_gmlip.store import global_store

    cu = ase_bulk("Cu", "fcc", a=3.59, cubic=True)
    cu.calc = EMT()
    cu.get_potential_energy()
    global_store().add(
        to_interface(cu), stage="imported", role="bulk", level="emt", label="EMT Cu"
    )

    (reference,) = client.get("/api/references", params={"role": "bulk"}).json()[
        "references"
    ]
    assert (reference["scope"], reference["formula"], reference["level"]) == (
        "global",
        "Cu4",
        "emt",
    )
    assert reference["cell_lengths"] == pytest.approx([3.59] * 3)
    assert reference["energy_per_atom"] == pytest.approx(cu.get_potential_energy() / 4)

    listing = client.get(
        "/api/records", params={"scope": "global", "role": "bulk"}
    ).json()
    assert listing["scope"] == "global" and [
        row["role"] for row in listing["rows"]
    ] == ["bulk"]
    assert client.get("/api/records", params={"scope": "nope"}).status_code == 400

    state = client.post(
        f"/api/records/{reference['id']}/load", params={"ws": BULK, "scope": "global"}
    ).json()
    assert (state["origin"], state["level"], state["role"], state["current_id"]) == (
        "global #1",
        "emt",
        "bulk",
        None,
    )
    assert (
        client.get("/api/records").json()["rows"] == []
    )  # nothing was copied into the project yet

    post(
        client,
        "/api/cleave",
        BULK,
        {"miller": [1, 1, 1], "layers": 4, "supercell": [2, 2], "target": SURFACE},
    )
    assert (
        "has no energy"
        in client.get("/api/energetics", params={"ws": SURFACE}).json()["missing"][0]
    )
    job_id = post(
        client, "/api/jobs", SURFACE, {"kind": "singlepoint", "calculator": "emt"}
    ).json()["job_id"]
    _wait(client)
    (row,) = client.get("/api/records", params={"job_id": job_id}).json()["rows"]
    assert (row["role"], row["level"]) == ("surface", "emt")
    post(client, f"/api/records/{row['id']}/load", SURFACE)
    result = client.get("/api/energetics", params={"ws": SURFACE}).json()
    assert 0.5 < result["values"]["surface energy (J/m²)"] < 2.5
    assert (
        result["references"] == {"bulk Cu": "global #1 Cu4"} and result["missing"] == []
    )


def test_vacuum_adjustment(slab_client):
    client = slab_client
    before = client.get("/api/state", params={"ws": SURFACE}).json()
    z = [atom["z"] for atom in before["scene"]["atoms"]]
    thickness = max(z) - min(z)
    assert before["info"]["cell_lengths"][2] == pytest.approx(thickness + 24.0)

    state = post(
        client,
        "/api/edit",
        SURFACE,
        {"op": "vacuum", "vacuum": 16.0, "position": "bottom"},
    ).json()
    z = [atom["z"] for atom in state["scene"]["atoms"]]
    assert state["info"]["cell_lengths"][2] == pytest.approx(thickness + 16.0)
    assert min(z) == pytest.approx(1.0)
    assert state["info"]["n_fixed"] == before["info"]["n_fixed"]

    state = post(client, "/api/edit", SURFACE, {"op": "vacuum", "vacuum": 16.0}).json()
    assert min(atom["z"] for atom in state["scene"]["atoms"]) == pytest.approx(8.0)
    assert post(client, "/api/undo", SURFACE).json()["info"]["cell_lengths"][
        2
    ] == pytest.approx(thickness + 16.0)
    assert (
        post(
            client,
            "/api/edit",
            SURFACE,
            {"op": "vacuum", "vacuum": 16.0, "position": "top"},
        ).status_code
        == 400
    )


def test_molecule_window_library_and_placement(slab_client, tmp_path):
    client = slab_client
    state = post(client, "/api/molecule/create", MOLECULE, {"name": "CH3OH"}).json()
    assert state["info"]["formula"] == "CH4O"
    assert state["info"]["cell_lengths"] == pytest.approx([15.0] * 3)
    assert (
        post(
            client, "/api/molecule/create", MOLECULE, {"name": "not-a-molecule"}
        ).status_code
        == 400
    )

    builtin = {m["name"]: m for m in state["molecules"]}
    assert builtin["COOH"]["source"] == "intermediate" and builtin["COOH"][
        "reactions"
    ] == ["CO2RR"]
    assert builtin["OOH"]["reactions"] == ["OER", "ORR"]
    assert "initial guess" in builtin["COOH"]["note"]

    oxygen = next(a["index"] for a in state["scene"]["atoms"] if a["elem"] == "O")
    state = post(
        client,
        "/api/library",
        MOLECULE,
        {
            "name": "methanol",
            "scope": "project",
            "bind_index": oxygen,
            "reactions": ["CO2RR"],
            "note": "G2",
        },
    ).json()
    entry = state["molecules"][0]
    assert (entry["name"], entry["source"], entry["reactions"], entry["note"]) == (
        "methanol",
        "project",
        ["CO2RR"],
        "G2",
    )
    assert (tmp_path / "proj" / "molecules.json").is_file()
    assert post(client, "/api/library", MOLECULE, {"name": " "}).status_code == 400
    assert (
        post(
            client, "/api/library", MOLECULE, {"name": "x", "scope": "nope"}
        ).status_code
        == 400
    )

    # The library molecule is placed on the surface through its binding atom.
    n_slab = client.get("/api/state", params={"ws": SURFACE}).json()["info"]["natoms"]
    position = [1.0, 1.0, 30.0]
    state = post(
        client,
        "/api/edit",
        SURFACE,
        {"op": "place", "adsorbate": "methanol", "position": position},
    ).json()
    assert state["info"]["natoms"] == n_slab + 6
    placed = state["scene"]["atoms"][n_slab:]
    bound = next(a for a in placed if a["elem"] == "O")
    assert [bound["x"], bound["y"], bound["z"]] == pytest.approx(position)
    assert (
        post(
            client,
            "/api/edit",
            SURFACE,
            {"op": "place", "adsorbate": "nope", "position": position},
        ).status_code
        == 400
    )

    # Seed intermediates open in a molecule window for editing or optimization.
    state = post(client, "/api/library/OCCO/open", MOLECULE).json()
    assert state["info"]["formula"] == "C2O2"
    assert (
        client.delete("/api/library/methanol", params={"ws": MOLECULE}).status_code
        == 200
    )
    assert (
        client.delete("/api/library/methanol", params={"ws": MOLECULE}).status_code
        == 400
    )


def test_global_library_is_shared_between_projects(client, tmp_path):
    post(client, "/api/molecule/create", MOLECULE, {"name": "H2O"})
    post(client, "/api/library", MOLECULE, {"name": "shared-water", "scope": "global"})
    post(
        client,
        "/api/library",
        MOLECULE,
        {"name": "session-water", "scope": "temporary"},
    )

    meta = client.post("/api/project", json={"path": str(tmp_path / "other")}).json()
    assert meta["project"] == str((tmp_path / "other").resolve())
    assert meta["recent_projects"][:2] == [
        str((tmp_path / "other").resolve()),
        str((tmp_path / "proj").resolve()),
    ]
    assert (tmp_path / "other" / "jobs").is_dir()
    names = {
        m["name"]: m["source"] for m in client.get("/api/state").json()["molecules"]
    }
    assert names["shared-water"] == "global"
    assert "session-water" not in names
    assert client.post("/api/project", json={"path": ""}).status_code == 400


def test_save_export_and_upload(slab_client):
    client = slab_client
    assert client.get("/api/state", params={"ws": SURFACE}).json()["current_id"] is None
    state = post(client, "/api/save", SURFACE, {"label": "clean"}).json()
    assert state["current_id"] == 1 and state["lineage"] == [1]
    assert (
        post(client, "/api/save", SURFACE).json()["current_id"] == 1
    )  # idempotent while unmodified

    poscar = client.get("/api/export", params={"ws": SURFACE, "format": "vasp"})
    assert poscar.status_code == 200 and "Cu" in poscar.text
    assert "attachment" in poscar.headers["content-disposition"]
    assert (
        client.get("/api/export", params={"ws": SURFACE, "format": "nope"}).status_code
        == 400
    )

    response = client.post(
        "/api/upload",
        params={"name": "POSCAR.vasp", "ws": "surface-1"},
        content=poscar.text.encode(),
    )
    state = response.json()
    assert (
        response.status_code == 200
        and state["current_id"] is None
        and state["info"]["natoms"] == 48
    )
    xyz = b"3\nwater\nO 0 0 0\nH 0 0.76 0.59\nH 0 -0.76 0.59\n"
    state = client.post(
        "/api/upload", params={"name": "w.xyz", "ws": MOLECULE}, content=xyz
    ).json()
    assert state["info"]["formula"] == "H2O" and state["info"][
        "cell_lengths"
    ] == pytest.approx([15.0] * 3)
    assert (
        client.post(
            "/api/upload", params={"name": "bad.cif", "ws": BULK}, content=b"not a cif"
        ).status_code
        == 400
    )


def test_bulk_cell_relax_then_cleave_from_relaxed_bulk(client):
    post(
        client, "/api/bulk/create", BULK, {"element": "Cu", "lattice": "fcc", "a": 3.70}
    )
    response = post(
        client, "/api/jobs", BULK, {"kind": "cell_relax", "calculator": "emt"}
    )
    assert response.status_code == 200
    (job,) = _wait(client)
    assert job["state"] == "done", client.get(f"/api/jobs/{job['job_id']}/log").json()[
        "log"
    ]

    rows = client.get("/api/records", params={"stage": "cell_relaxed"}).json()["rows"]
    state = post(client, f"/api/records/{rows[-1]['id']}/load", BULK).json()
    assert state["info"]["cell_lengths"] == pytest.approx([3.59] * 3, abs=0.01)
    assert state["lineage"] == [1, rows[-1]["id"]]
    state = post(
        client,
        "/api/cleave",
        BULK,
        {"miller": [1, 0, 0], "layers": 2, "target": SURFACE},
    ).json()
    assert state["workspace"]["id"] == SURFACE and state["current_id"] is None
    assert state["info"]["cell_lengths"][0] == pytest.approx(3.59, abs=0.01)


def test_gcmc_job_ranking_and_graph(slab_client):
    client = slab_client
    response = post(
        client,
        "/api/jobs",
        SURFACE,
        {
            "kind": "gcmc",
            "calculator": "emt",
            "params": {
                "n_steps": 10,
                "relax_steps": 3,
                "chemical_potentials": {"O": -2.0},
            },
            "label": "screen",
        },
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    assert response.json()["state"]["current_id"] == 1  # the input was stored first
    (job,) = _wait(client)
    assert job["state"] == "done" and job["n_structures"] >= 2

    ranked = client.get(
        "/api/records", params={"job_id": job_id, "order": "rank"}
    ).json()
    assert ranked["metric"] == "score"
    values = [row["value"] for row in ranked["rows"]]
    assert values == sorted(values)
    assert [row["rank"] for row in ranked["rows"]] == list(range(1, len(values) + 1))
    assert ranked["rows"][0]["delta"] == 0.0
    assert all(
        row["delta"] == pytest.approx(row["score"] - values[0])
        for row in ranked["rows"]
    )
    by_time = client.get("/api/records", params={"job_id": job_id}).json()["rows"]
    assert [row["id"] for row in by_time] == sorted(row["id"] for row in by_time)

    graph = client.get("/api/graph").json()
    nodes = {node["id"]: node for node in graph["nodes"]}
    assert nodes["s1"]["type"] == "structure" and nodes["s1"]["current"]
    job_node = nodes[f"j{job_id}"]
    assert (job_node["type"], job_node["label"], job_node["state"]) == (
        "job",
        "gcmc",
        "done",
    )
    assert job_node["n_structures"] == len(by_time)
    assert {"source": "s1", "target": f"j{job_id}", "kind": "input"} in graph["edges"]
    outputs = [edge for edge in graph["edges"] if edge["source"] == f"j{job_id}"]
    # Search intermediates are folded into the job node; only the best candidate is drawn.
    assert [edge["target"] for edge in outputs] == [f"s{ranked['rows'][0]['id']}"]

    assert "GCMC finished" in client.get(f"/api/jobs/{job_id}/log").json()["log"]
    assert (
        client.delete(
            f"/api/records/{by_time[-1]['id']}", params={"ws": SURFACE}
        ).status_code
        == 200
    )
    assert (
        len(client.get("/api/records", params={"job_id": job_id}).json()["rows"])
        == len(by_time) - 1
    )


def test_graph_shows_edits_between_structures(slab_client):
    client = slab_client
    post(client, "/api/save", SURFACE)
    post(
        client,
        "/api/edit",
        SURFACE,
        {"op": "substitute", "indices": [0], "element": "Pt"},
    )
    post(client, "/api/save", SURFACE)
    graph = client.get("/api/graph").json()
    assert {"source": "s1", "target": "s2", "kind": "edit"} in graph["edges"]


def test_submit_rejects_unknown_task_and_calculator(slab_client):
    assert (
        post(
            slab_client, "/api/jobs", SURFACE, {"kind": "nope", "calculator": "emt"}
        ).status_code
        == 400
    )
    assert (
        post(
            slab_client, "/api/jobs", SURFACE, {"kind": "relax", "calculator": "nope"}
        ).status_code
        == 400
    )
    assert (
        post(
            slab_client, "/api/jobs", MOLECULE, {"kind": "relax", "calculator": "emt"}
        ).status_code
        == 400
    )  # empty


def test_tasks_are_offered_by_category_and_model_type(slab_client):
    client = slab_client
    meta = client.get("/api/meta").json()
    assert meta["task_categories"][:4] == [
        "single point",
        "optimization",
        "molecular dynamics",
        "global search",
    ]
    tasks = {task["kind"]: task for task in meta["tasks"]}
    assert (
        tasks["cell_relax"]["applies_to"] == ["bulk"]
        and tasks["gcmc"]["category"] == "global search"
    )
    calculators = {entry["name"]: entry for entry in meta["calculators"]}
    assert (
        calculators["mace"]["defaults"]["model_path"] == ""
        and "model_path" in calculators["matgl"]["defaults"]
    )

    # A task is refused for a model type it does not apply to.
    response = post(
        client, "/api/jobs", SURFACE, {"kind": "cell_relax", "calculator": "emt"}
    )
    assert (
        response.status_code == 400
        and "is for bulk models" in response.json()["detail"]
    )
    assert (
        post(
            client, "/api/jobs", BULK, {"kind": "gcmc", "calculator": "emt"}
        ).status_code
        == 400
    )


def test_wait_endpoint_returns_when_jobs_finish(slab_client):
    client = slab_client
    job_id = post(
        client, "/api/jobs", SURFACE, {"kind": "singlepoint", "calculator": "emt"}
    ).json()["job_id"]
    result = client.get("/api/jobs/wait", params={"timeout": 60}).json()
    (job,) = result["jobs"]
    assert (job["job_id"], job["state"], job["active"], job["category"]) == (
        job_id,
        "done",
        False,
        "single point",
    )
    assert job["last_structure_id"] == result["n_structures"] == 2
    assert (
        client.get("/api/jobs/wait", params={"timeout": 0}).json()["jobs"][0]["state"]
        == "done"
    )


def test_vasp_preview_layers_tags(slab_client):
    result = post(
        slab_client,
        "/api/vasp/preview",
        SURFACE,
        {
            "preset": "relax-r2SCAN",
            "precision": "high",
            "dispersion": "D4",
            "user_incar_settings": {"NSW": 250},
        },
    ).json()
    incar = result["incar"]
    assert result["slab"] and result["selective_dynamics"]
    assert (
        incar["ISIF"],
        incar["IDIPOL"],
        incar["ENCUT"],
        incar["IVDW"],
        incar["NSW"],
    ) == (2, 3, 680.0, 13, 250)
    assert not post(
        slab_client, "/api/vasp/preview", BULK, {"preset": "relax-PBE"}
    ).json()["slab"]
    assert (
        post(slab_client, "/api/vasp/preview", SURFACE, {"preset": "nope"}).status_code
        == 400
    )


def test_session_without_server(tmp_path):
    session = Session(Project(tmp_path / "proj"))
    session.create_bulk(BULK, "Cu", "fcc", 3.61)
    target = session.cleave(BULK, [1, 1, 1], target_id=SURFACE, layers=3)
    assert target == SURFACE
    assert session.state(SURFACE)["info"]["formula"] == "Cu12"
    assert session.state()["workspace"]["id"] == BULK  # the first window is the default
    with pytest.raises(ValueError, match="unknown edit operation"):
        session.edit(SURFACE, "nope")


# ---------------------------------------------------------------- desktop launcher


def test_quit_is_only_offered_when_enabled(client):
    assert client.get("/api/meta").json()["can_quit"] is False
    assert client.get("/api/meta").json()["app"] == "gs_gmlip"
    assert client.post("/api/quit").status_code == 403
    stopped = []
    client.app.state.quit = lambda: stopped.append(True)
    assert client.get("/api/meta").json()["can_quit"] is True
    assert client.post("/api/quit").json() == {"stopping": True}
    assert stopped == [True]


def test_macos_app_bundle(tmp_path):
    import plistlib
    import sys

    from gs_gmlip.app.bundle import build_macos_app

    if sys.platform != "darwin":
        with pytest.raises(RuntimeError, match="macOS"):
            build_macos_app(tmp_path)
        return
    bundle = build_macos_app(
        tmp_path, python="/some/env with space/bin/python", port=8123
    )
    assert bundle == tmp_path.resolve() / "GS-GMLIP.app"
    info = plistlib.loads((bundle / "Contents" / "Info.plist").read_bytes())
    assert info["CFBundleExecutable"] == "GS-GMLIP" and info["LSUIElement"] is True
    launcher = bundle / "Contents" / "MacOS" / "GS-GMLIP"
    text = launcher.read_text()
    assert "PYTHON='/some/env with space/bin/python'" in text
    assert "-m gs_gmlip.cli app --port 8123 --allow-quit" in text
    assert launcher.stat().st_mode & 0o111


def test_run_app_reuses_a_running_app(monkeypatch):
    from gs_gmlip.app import server

    opened = []
    monkeypatch.setattr(server, "_port_free", lambda host, port: False)
    monkeypatch.setattr(server, "_running_app", lambda url: True)
    monkeypatch.setattr("webbrowser.open", opened.append)
    server.run_app(port=8080)  # returns instead of starting a second server
    assert opened == ["http://127.0.0.1:8080"]
