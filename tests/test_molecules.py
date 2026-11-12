"""Tests for the seed intermediates, the tiered molecule library, and user-level config."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import molecule

from gm2_surf.config import gm2_home, recent_projects, remember_project
from gm2_surf.interface import operations as ops
from gm2_surf.interface.intermediates import (
    INTERMEDIATE_NAMES,
    INTERMEDIATES,
    REACTIONS,
    build_intermediate,
    get_intermediate,
)
from gm2_surf.interface.molecules import MoleculeLibrary
from gm2_surf.jobs import JobSpec, run_task
from gm2_surf.store import Project

EXPECTED_FORMULAS = {
    "H": "H1", "OH": "H1 O1", "OOH": "H1 O2", "COOH": "H1 C1 O2", "OCHO": "H1 C1 O2", "CO": "C1 O1",
    "CHO": "H1 C1 O1", "COH": "H1 C1 O1", "CHOH": "H2 C1 O1", "CH2OH": "H3 C1 O1", "CH3O": "H3 C1 O1",
    "OCCO": "C2 O2", "OCCHO": "H1 C2 O2", "CCO": "C2 O1", "CHCO": "H1 C2 O1", "CH3CH2O": "H5 C2 O1",
}  # fmt: skip


def test_seed_set_covers_each_reaction_with_unique_names():
    assert len(set(INTERMEDIATE_NAMES)) == len(INTERMEDIATE_NAMES) >= 35
    for reaction in REACTIONS:
        assert any(reaction in item.reactions for item in INTERMEDIATES)
    assert all(set(item.reactions) <= set(REACTIONS) for item in INTERMEDIATES)
    by_name = {item.name: item for item in INTERMEDIATES}
    assert {"OH", "O", "OOH"} <= {n for n, i in by_name.items() if "OER" in i.reactions}
    assert {"COOH", "CO", "CHO", "OCHO"} <= {
        n for n, i in by_name.items() if "CO2RR" in i.reactions
    }
    assert "H" in {n for n, i in by_name.items() if "HER" in i.reactions}


@pytest.mark.parametrize("item", INTERMEDIATES, ids=lambda item: item.name)
def test_every_intermediate_is_a_sane_oriented_molecule(item):
    built = build_intermediate(item)
    coords = built.cart_coords
    # The binding atom is at the origin and is the element the definition names.
    at_origin = int(np.argmin(np.linalg.norm(coords, axis=1)))
    assert np.allclose(coords[at_origin], 0.0, atol=1e-8)
    assert str(built[at_origin].specie) == item.bind[0]
    if len(built) > 1:
        distances = built.distance_matrix + np.eye(len(built)) * 10
        assert (
            0.7 < distances.min() < 1.3
        )  # no fused atoms, shortest contact is a real bond
        # Connected: every atom has a neighbour within bonding distance.
        assert (distances.min(axis=1) < 1.7).all()
        assert (
            coords[:, 2].max() >= -1e-8
        )  # the body does not sit entirely below the binding atom
    assert "initial guess" in item.note


@pytest.mark.parametrize(("name", "formula"), EXPECTED_FORMULAS.items())
def test_intermediate_compositions(name, formula):
    assert get_intermediate(name).composition.formula == formula


def test_hydrogen_removal_targets_the_named_atom():
    cooh, ocho = get_intermediate("COOH"), get_intermediate("OCHO")
    # COOH keeps the O-H and binds through C; formate keeps the C-H and binds through O.
    assert any(0.9 < cooh.get_distance(i, j) < 1.05 for i in range(4) for j in range(4)
               if {str(cooh[i].specie), str(cooh[j].specie)} == {"O", "H"})  # fmt: skip
    assert any(1.05 < ocho.get_distance(i, j) < 1.15 for i in range(4) for j in range(4)
               if {str(ocho[i].specie), str(ocho[j].specie)} == {"C", "H"})  # fmt: skip
    with pytest.raises(KeyError):
        get_intermediate("nope")


def test_library_tiers_and_precedence(tmp_path):
    library = MoleculeLibrary(tmp_path / "proj", global_dir=tmp_path / "global")
    (tmp_path / "proj").mkdir()
    sources = {entry["name"]: entry["source"] for entry in library.entries()}
    assert sources["COOH"] == "intermediate" and sources["NO"] == "builtin"
    assert (
        sources["CO"] == "intermediate"
    )  # seed intermediates shadow the plain built-ins
    assert "CO" in library and "nope" not in library

    water = molecule("H2O")
    library.add("water", water, scope="global", reactions=["OER"], note="ref")
    library.add("water", molecule("H2O2"), scope="project")
    library.add("water", molecule("CH4"), scope="temporary")
    assert len(library.get("water")) == 5  # temporary wins
    names = [entry["name"] for entry in library.entries()]
    assert names.count("water") == 1 and names[0] == "water"

    again = MoleculeLibrary(tmp_path / "proj", global_dir=tmp_path / "global")
    assert (
        len(again.get("water")) == 4
    )  # project beats global once the session copy is gone
    elsewhere = MoleculeLibrary(tmp_path / "other", global_dir=tmp_path / "global")
    assert len(elsewhere.get("water")) == 3
    entry = next(e for e in elsewhere.entries() if e["name"] == "water")
    assert (entry["source"], entry["reactions"], entry["note"]) == (
        "global",
        ["OER"],
        "ref",
    )

    # A user molecule may override a seed intermediate.
    library.add("COOH", molecule("HCOOH"), scope="temporary")
    assert len(library.get("COOH")) == 5
    library.remove("COOH")
    assert len(library.get("COOH")) == 4
    library.remove("water")
    assert "water" not in MoleculeLibrary(
        tmp_path / "other", global_dir=tmp_path / "global"
    )
    with pytest.raises(KeyError):
        library.remove("water")


def test_library_binding_atom_and_validation(tmp_path):
    library = MoleculeLibrary(tmp_path)
    methanol = molecule("CH3OH")
    oxygen = [atom.index for atom in methanol if atom.symbol == "O"][0]
    library.add("methanol", methanol, bind_index=oxygen)
    stored = library.get("methanol")
    assert str(stored[oxygen].specie) == "O" and np.allclose(
        stored.cart_coords[oxygen], 0.0
    )
    library.add("auto", methanol)
    assert library.get("auto").cart_coords[:, 2].min() == pytest.approx(
        0.0
    )  # lowest atom binds
    assert len(library.atoms("methanol")) == 6
    for bad in (
        {"name": " "},
        {"name": "x", "scope": "nope"},
        {"name": "x", "bind_index": 99},
    ):
        with pytest.raises(ValueError):
            library.add(atoms=methanol, **bad)
    with pytest.raises(KeyError):
        library.get("nope")


def test_search_tasks_use_project_library_molecules(tmp_path):
    project = Project(tmp_path / "proj")
    MoleculeLibrary(project.root).add("my-oxygen", molecule("O"), scope="project")
    slab = ops.build_slab(
        ops.make_bulk("Cu", "fcc", 3.61), (1, 1, 1), layers=3, fix_layers=1
    )
    row_id = project.store.add(slab, stage="built")
    context = run_task(
        JobSpec(
            kind="ga",
            structure_id=row_id,
            job_id="ga-lib",
            params={
                "blocks": {"my-oxygen": 1, "OH": 1},
                "n_initial": 2,
                "population_size": 2,
                "n_steps": 1,
                "relax_steps": 2,
            },
        ),
        project.store,
        project.root,
    )
    assert context.n_structures >= 2
    assert {r.formula for r in project.store.records(job_id="ga-lib")} == {"Cu12HO2"}
    with pytest.raises(KeyError, match="unknown molecule"):
        run_task(
            JobSpec(
                kind="ga",
                structure_id=row_id,
                job_id="ga-bad",
                params={"blocks": {"nope": 1}},
            ),
            project.store,
            project.root,
        )


def test_recent_projects_are_remembered_most_recent_first(tmp_path):
    assert recent_projects() == []
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    remember_project(first)
    remember_project(second)
    remember_project(first)
    assert recent_projects() == [str(first.resolve()), str(second.resolve())]
    second.rmdir()
    assert recent_projects() == [str(first.resolve())]  # vanished folders are dropped
    assert gm2_home().name.startswith(
        "gm2_home"
    )  # the autouse fixture isolates the user directory


def test_home_falls_back_to_the_former_package_name(tmp_path, monkeypatch):
    monkeypatch.delenv("GM2_SURF_HOME")
    monkeypatch.setenv("GS_GMLIP_HOME", str(tmp_path / "configured"))
    assert (
        gm2_home() == tmp_path / "configured"
    )  # the former variable is still honoured

    monkeypatch.delenv("GS_GMLIP_HOME")
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    legacy = tmp_path / ".gs_gmlip"
    legacy.mkdir()
    (legacy / "hosts.json").write_text("{}")
    home = gm2_home()
    assert home == tmp_path / ".gm2_surf" and (home / "hosts.json").read_text() == "{}"
    assert not legacy.exists()  # moved once, not copied
