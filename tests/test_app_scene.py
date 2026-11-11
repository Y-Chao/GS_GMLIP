"""Tests for the viewer scene (no NiceGUI needed)."""

from __future__ import annotations

import json

import pytest

from gs_gmlip.app.scene import atom_role, build_scene
from gs_gmlip.interface import operations as ops


@pytest.fixture(scope="module")
def system():
    slab = ops.build_slab(
        ops.make_bulk("Cu", "fcc", 3.61), (1, 1, 1), layers=3, fix_layers=1
    )
    sites = ops.find_adsorption_sites(slab)
    return ops.place_adsorbate(slab, "CO", sites[0].coords), sites


def test_scene_is_json_and_marks_roles(system):
    interface, sites = system
    n_atoms, n_substrate = len(interface.interface), len(interface.substrate)
    scene = build_scene(interface, selection=[0, n_atoms - 1], sites=sites)
    json.dumps(scene)
    assert len(scene["atoms"]) == n_atoms
    assert [a["role"] for a in scene["atoms"]] == ["substrate"] * n_substrate + [
        "adsorbate"
    ] * 2
    assert [a["index"] for a in scene["atoms"] if a["selected"]] == [0, n_atoms - 1]
    assert [a["index"] for a in scene["atoms"] if a["fixed"]] == interface.fix
    assert scene["bonds"] == [[n_substrate, n_substrate + 1]]
    assert len(scene["sites"]) == len(sites)
    assert scene["ghosts"] == []


def test_fixed_atoms_are_tinted(system):
    interface, _ = system
    atoms = build_scene(interface)["atoms"]
    fixed_color = atoms[interface.fix[0]]["color"]
    free_color = atoms[interface.relax[0]]["color"]
    assert fixed_color != free_color


def test_ghost_images_are_in_plane_only(system):
    interface, _ = system
    n_atoms = len(interface.interface)
    scene = build_scene(interface, repeat=1)
    assert len(scene["ghosts"]) == 8 * n_atoms
    z_real = sorted(round(a["z"], 6) for a in scene["atoms"])
    z_ghost = sorted({round(g["z"], 6) for g in scene["ghosts"]})
    assert set(z_ghost) == set(z_real)


def test_atom_role(system):
    interface, _ = system
    assert atom_role(interface, 0) == "substrate"
    assert atom_role(interface, len(interface.interface) - 1) == "adsorbate"
