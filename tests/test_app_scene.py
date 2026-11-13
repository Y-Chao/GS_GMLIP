"""Tests for the viewer scene (plain Python)."""

from __future__ import annotations

import json

import pytest

from gs_gmlip.app.scene import atom_role, build_scene, element_color, find_bonds
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
    scene = build_scene(interface, sites)
    json.dumps(scene)
    assert set(scene) == {"atoms", "bonds", "cell", "sites", "constraints"}
    assert len(scene["atoms"]) == n_atoms
    assert [a["role"] for a in scene["atoms"]] == ["substrate"] * n_substrate + [
        "adsorbate"
    ] * 2
    assert [a["index"] for a in scene["atoms"] if a["fixed"]] == interface.fix
    assert len(scene["sites"]) == len(sites)
    assert scene["constraints"] == {"bonds": [], "com": [], "walls": []}


def test_fixed_atoms_are_tinted_and_light_elements_use_pmol_colours(system):
    interface, _ = system
    atoms = build_scene(interface)["atoms"]
    assert atoms[interface.fix[0]]["color"] != atoms[interface.relax[0]]["color"]
    assert atoms[interface.relax[0]]["color"] == element_color("Cu")
    assert element_color("C") == "#d9d9d9"
    assert element_color("H") == "#fafafa"
    assert element_color("N") == "#7f7fbf"
    carbon = next(a for a in atoms if a["elem"] == "C")
    assert carbon["color"] == "#d9d9d9"


def test_bonds_include_adsorbate_and_never_cross_the_cell(system):
    interface, _ = system
    n_substrate = len(interface.substrate)
    bonds = find_bonds(interface)
    assert [n_substrate, n_substrate + 1] in bonds  # C-O
    assert all(i < j for i, j in bonds)
    # Every drawn bond is a real short contact in the unwrapped cell.
    atoms = interface.interface
    assert max(atoms.get_distance(i, j, mic=False) for i, j in bonds) < 3.1
    # Cu-Cu nearest neighbours inside the cell are bonded.
    assert any(i < n_substrate and j < n_substrate for i, j in bonds)


def test_constraints_appear_in_scene(system):
    interface, _ = system
    n_atoms = len(interface.interface)
    constrained = ops.add_bond_constraint(interface, n_atoms - 2, n_atoms - 1)
    constrained = ops.add_com_constraint(constrained, [n_atoms - 2, n_atoms - 1])
    constrained = ops.add_wall(constrained, [n_atoms - 1], height=19.0)
    constraints = build_scene(constrained)["constraints"]
    assert constraints["bonds"] == [[n_atoms - 2, n_atoms - 1]]
    assert constraints["walls"] == [{"index": n_atoms - 1, "height": 19.0}]
    (group,) = constraints["com"]
    assert group["indices"] == [n_atoms - 2, n_atoms - 1]
    carbon_oxygen = constrained.interface[[n_atoms - 2, n_atoms - 1]]
    assert group["center"] == pytest.approx(list(carbon_oxygen.get_center_of_mass()))


def test_atom_role(system):
    interface, _ = system
    assert atom_role(interface, 0) == "substrate"
    assert atom_role(interface, len(interface.interface) - 1) == "adsorbate"
