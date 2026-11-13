"""JSON Lines reference files: round trip, the packaged library, and its use in the app."""

from __future__ import annotations

import json

import pytest
from ase.build import bulk, molecule
from ase.calculators.singlepoint import SinglePointCalculator

from gm2_surf.cli import main as cli_main
from gm2_surf.importers import to_interface
from gm2_surf.reference import (
    ReferenceLibrary,
    builtin_references,
    import_jsonl,
    read_jsonl,
    write_jsonl,
)
from gm2_surf.store import Project, StructureStore, global_store


@pytest.fixture
def store(tmp_path) -> StructureStore:
    store = StructureStore(tmp_path / "source.db")
    copper = bulk("Cu", cubic=True)
    copper.calc = SinglePointCalculator(copper, energy=-14.5)
    store.add(
        to_interface(copper),
        stage="imported",
        role="bulk",
        level="vasp/PBE/pw450",
        label="ref",
        tags={"xc": "PBE", "source": "/private/path/file.db"},
    )
    store.add(to_interface(molecule("H2O")), stage="built", role="molecule")
    return store


def test_jsonl_round_trip_keeps_structure_energy_and_labels(store, tmp_path):
    path = tmp_path / "refs.jsonl"
    assert write_jsonl(store, path) == 2
    first, second = read_jsonl(path)
    assert (
        first["formula"] == "Cu4"
        and first["energy"] == -14.5
        and first["level"] == "vasp/PBE/pw450"
    )
    assert first["tags"] == {"xc": "PBE"}  # origin paths are not exported
    assert second["energy"] is None and second["pbc"] == [False, False, False]
    assert all(
        json.loads(line) for line in path.read_text().splitlines()
    )  # one object per line

    target = StructureStore(tmp_path / "target.db")
    assert import_jsonl(path, target) == 2
    copy = target.records(role="bulk")[0]
    assert (copy.formula, copy.energy, copy.level, copy.label) == (
        "Cu4",
        -14.5,
        "vasp/PBE/pw450",
        "ref",
    )
    assert target.get(copy.id).interface.cell.lengths() == pytest.approx(
        store.get(1).interface.cell.lengths()
    )


def test_write_jsonl_filters_by_role_and_id(store, tmp_path):
    assert write_jsonl(store, tmp_path / "bulk.jsonl", roles=["bulk"]) == 1
    assert write_jsonl(store, tmp_path / "one.jsonl", structure_ids=[2]) == 1
    assert read_jsonl(tmp_path / "one.jsonl")[0]["formula"] == "H2O"


def test_library_reads_like_a_store(store, tmp_path):
    path = tmp_path / "refs.jsonl"
    write_jsonl(store, path)
    library = ReferenceLibrary([path])
    assert len(library) == 2 and [rec.id for rec in library.records()] == [1, 2]
    assert [
        rec.formula for rec in library.records(role="bulk", level="vasp/PBE/pw450")
    ] == ["Cu4"]
    assert library.records(job_id="some-job") == []
    assert library.get(1).interface.get_potential_energy() == -14.5
    with pytest.raises(KeyError):
        library.record(3)


def test_packaged_references_cover_bulk_lattices_and_gas_molecules():
    library = builtin_references()
    bulks = library.records(role="bulk")
    assert {rec.formula for rec in bulks} == {"Cu4", "Ag4", "Au4", "Pd4", "Pt4"}
    assert {"vasp/r2SCAN/pw450", "vasp/PBE/pw450", "gpaw/PBE/pw600"} <= {
        rec.level for rec in bulks
    }
    assert all(rec.energy is not None and rec.level for rec in library.records())
    assert {"H2", "H2O", "CO", "CO2"} <= {
        rec.formula for rec in library.records(role="molecule")
    }
    copper = next(
        rec
        for rec in bulks
        if rec.formula == "Cu4" and rec.level == "vasp/r2SCAN/pw450"
    )
    assert (
        3.4 < library.get(copper.id).interface.cell.lengths()[0] < 3.8
    )  # fcc Cu lattice constant in Å


def test_export_and_import_commands(store, tmp_path, capsys):
    project = Project(tmp_path / "proj")
    store.copy_to(project.store, [1, 2])
    out = tmp_path / "out.jsonl"
    assert cli_main(["export", str(out)]) == 2  # a source must be named
    assert (
        cli_main(["export", str(out), "--project", str(project.root), "--role", "bulk"])
        == 0
    )
    assert "wrote 1 structures" in capsys.readouterr().out
    assert cli_main(["import", str(out), "--global"]) == 0
    assert [rec.formula for rec in global_store().records()] == ["Cu4"]


def test_app_lists_and_loads_builtin_references(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from gm2_surf.app.server import create_app

    client = TestClient(create_app(tmp_path / "proj"))
    rows = client.get(
        "/api/records", params={"scope": "builtin", "role": "bulk"}
    ).json()["rows"]
    assert len(rows) == len(builtin_references().records(role="bulk"))
    target = next(
        row
        for row in rows
        if row["formula"] == "Pt4" and row["level"] == "vasp/PBE/pw450"
    )
    state = client.post(
        f"/api/records/{target['id']}/load",
        params={"ws": "bulk-1", "scope": "builtin"},
        json={},
    ).json()
    assert state["info"]["formula"] == "Pt4" and state["level"] == "vasp/PBE/pw450"
    assert (
        state["current_id"] is None
    )  # an unsaved copy: nothing is written to the project
    references = client.get("/api/references", params={"role": "bulk"}).json()
    assert any(item["scope"] == "builtin" for item in references["references"])
    assert len(Project(tmp_path / "proj").store) == 0
