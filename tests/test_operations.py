"""Tests for the pymatgen-based builder operations."""

from __future__ import annotations

import numpy as np
import pytest

from gm2_surf.interface import operations as ops


@pytest.fixture(scope="module")
def cu_bulk():
    return ops.make_bulk("Cu", "fcc", 3.61)


@pytest.fixture
def cu111(cu_bulk):
    return ops.build_slab(
        cu_bulk, (1, 1, 1), layers=3, vacuum=12.0, supercell=(2, 2), fix_layers=1
    )


@pytest.mark.parametrize(
    ("lattice", "n_atoms"),
    [("fcc", 4), ("bcc", 2), ("sc", 1), ("diamond", 8), ("hcp", 2)],
)
def test_make_bulk_atom_counts(lattice, n_atoms):
    assert len(ops.make_bulk("Cu", lattice, 3.0)) == n_atoms


def test_make_bulk_rejects_unknown_lattice():
    with pytest.raises(ValueError, match="unknown lattice"):
        ops.make_bulk("Cu", "nope", 3.0)


@pytest.mark.parametrize("miller", [(1, 1, 1), (1, 0, 0), (2, 1, 1)])
def test_build_slab_is_oriented_with_exact_vacuum(cu_bulk, miller):
    vacuum = 12.0
    slab = ops.build_slab(cu_bulk, miller, layers=3, vacuum=vacuum).interface
    cell = np.asarray(slab.cell)
    # a along x, b in the xy plane, c along z.
    assert np.allclose(cell[0, 1:], 0.0, atol=1e-8)
    assert np.allclose(cell[1, 2], 0.0, atol=1e-8)
    assert np.allclose(cell[2, :2], 0.0, atol=1e-8)
    thickness = np.ptp(slab.positions[:, 2])
    assert cell[2, 2] == pytest.approx(thickness + vacuum, abs=1e-6)
    # Slab is centred in the cell.
    assert slab.positions[:, 2].mean() == pytest.approx(cell[2, 2] / 2, abs=1e-6)
    # Nearest-neighbour distance of fcc Cu survives the rebuild.
    distances = slab.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    assert distances.min() == pytest.approx(3.61 / np.sqrt(2), abs=1e-3)


def test_build_slab_supercell_and_fixed_layers(cu_bulk):
    small = ops.build_slab(cu_bulk, (1, 1, 1), layers=3)
    big = ops.build_slab(cu_bulk, (1, 1, 1), layers=3, supercell=(2, 3), fix_layers=1)
    assert len(big.interface) == 6 * len(small.interface)
    per_layer = len(big.interface) // big.get_substrate_layers()
    assert len(big.fix) == per_layer
    assert list(big.interface.constraints[0].index) == big.fix
    # Fixed atoms are the lowest ones.
    z = big.interface.positions[:, 2]
    assert z[big.fix].max() < z[big.relax].min()


def test_adsorption_sites_cover_all_kinds_inside_cell(cu111):
    sites = ops.find_adsorption_sites(cu111, distance=2.0)
    kinds = {site.kind for site in sites}
    assert kinds == {"ontop", "bridge", "hollow"}
    # 2x2 of a 4-atom surface cell -> 16 ontop sites.
    assert sum(site.kind == "ontop" for site in sites) == 16
    top = cu111.interface.positions[:, 2].max()
    scaled = cu111.interface.cell.scaled_positions(
        np.array([site.coords for site in sites])
    )
    assert ((scaled[:, :2] > -1e-6) & (scaled[:, :2] < 1 + 1e-6)).all()
    ontop_z = [site.coords[2] for site in sites if site.kind == "ontop"]
    assert np.allclose(ontop_z, top + 2.0, atol=1e-6)


def test_symmetry_distinct_sites_are_fewer(cu111):
    assert len(ops.find_adsorption_sites(cu111, symmetry_reduce=True)) < len(
        ops.find_adsorption_sites(cu111)
    )


@pytest.mark.parametrize("name", ops.ADSORBATES)
def test_adsorbate_binds_through_lowest_atom_at_origin(name):
    mol = ops.get_adsorbate(name)
    distances = np.linalg.norm(mol.cart_coords, axis=1)
    bind = int(np.argmin(distances))
    # The binding atom sits at the origin.
    assert np.isclose(distances[bind], 0.0, atol=1e-8)
    if len(mol) > 1:
        # The rest of the molecule points away from the surface: its centroid is
        # straight above the binding atom, or coincides with it for molecules
        # that are laid flat (e.g. CO2, CH4).
        centroid = np.delete(mol.cart_coords, bind, axis=0).mean(axis=0)
        assert centroid[2] > -1e-6
        assert (
            np.allclose(centroid[:2], 0.0, atol=1e-6) or np.linalg.norm(centroid) < 1e-3
        )


def test_co_binds_through_carbon():
    mol = ops.get_adsorbate("CO")
    by_symbol = {str(site.specie): site.coords for site in mol}
    assert np.allclose(by_symbol["C"], 0.0, atol=1e-8)
    assert by_symbol["O"][2] == pytest.approx(1.15, abs=0.05)


def test_place_adsorbate_is_pure_and_keeps_constraints(cu111):
    site = ops.find_adsorption_sites(cu111)[0]
    n_before = len(cu111.interface)
    result = ops.place_adsorbate(cu111, "CO", site.coords)
    assert len(cu111.interface) == n_before
    assert len(result.interface) == n_before + 2
    assert result.adsList == [[n_before, n_before + 1]]
    assert result.fix == cu111.fix
    assert list(result.interface.constraints[0].index) == cu111.fix
    carbon = result.interface[[a.index for a in result.interface if a.symbol == "C"][0]]
    assert np.allclose(carbon.position, site.coords, atol=1e-8)


def test_place_adsorbate_rotation_about_normal(cu111):
    site = ops.find_adsorption_sites(cu111)[0]
    flat = ops.place_adsorbate(cu111, "CO2", site.coords)
    turned = ops.place_adsorbate(cu111, "CO2", site.coords, rotation=90.0)
    axis_flat = flat.interface.positions[-1] - flat.interface.positions[-2]
    axis_turned = turned.interface.positions[-1] - turned.interface.positions[-2]
    assert abs(axis_flat[2]) < 1e-6 and abs(axis_turned[2]) < 1e-6
    assert abs(axis_flat @ axis_turned) < 1e-6


def test_delete_atoms_remaps_substrate_and_fix(cu111):
    with_co = ops.place_adsorbate(
        cu111, "CO", ops.find_adsorption_sites(cu111)[0].coords
    )
    n_substrate = len(with_co.substrate)
    fixed_atom = with_co.fix[0]
    result = ops.delete_atoms(with_co, [fixed_atom, len(with_co.interface) - 1])
    assert len(result.substrate) == n_substrate - 1
    assert len(result.interface) == len(with_co.interface) - 2
    assert len(result.fix) == len(with_co.fix) - 1
    assert max(result.fix) < len(result.substrate)
    assert len(with_co.interface) == n_substrate + 2  # input untouched


def test_substitute_atoms(cu111):
    result = ops.substitute_atoms(cu111, [5], "Pt")
    assert result.interface[5].symbol == "Pt"
    assert result.substrate[5].symbol == "Pt"
    assert cu111.interface[5].symbol == "Cu"
    assert result.fix == cu111.fix


def test_translate_atoms(cu111):
    result = ops.translate_atoms(cu111, [0, 1], [0.1, 0.0, 0.3])
    shift = result.interface.positions - cu111.interface.positions
    assert np.allclose(shift[:2], [0.1, 0.0, 0.3], atol=1e-8)
    assert np.allclose(shift[2:], 0.0, atol=1e-8)


def test_set_fixed_and_fix_bottom_layers(cu111):
    free_atom = cu111.relax[0]
    fixed = ops.set_fixed(cu111, [free_atom], True)
    assert free_atom in fixed.fix
    assert free_atom not in fixed.relax
    released = ops.set_fixed(fixed, fixed.fix, False)
    assert released.fix == []
    assert released.interface.constraints == []
    assert len(ops.fix_bottom_layers(cu111, 2).fix) == 2 * len(cu111.fix)


def test_edit_rejects_bad_index(cu111):
    with pytest.raises(IndexError):
        ops.delete_atoms(cu111, [10_000])


def test_rotate_atoms_about_centroid(cu111):
    with_co = ops.place_adsorbate(
        cu111, "CO", ops.find_adsorption_sites(cu111)[0].coords
    )
    pair = with_co.adsList[0]
    before = with_co.interface.positions[pair]
    after = ops.rotate_atoms(with_co, pair, 90.0, axis=(1, 0, 0)).interface.positions[
        pair
    ]
    # Centroid and bond length are preserved; the bond turns from z into the xy plane.
    assert np.allclose(after.mean(axis=0), before.mean(axis=0), atol=1e-8)
    assert np.linalg.norm(after[1] - after[0]) == pytest.approx(
        np.linalg.norm(before[1] - before[0])
    )
    assert abs((after[1] - after[0])[2]) < 1e-8
    # A full turn is the identity; other atoms never move.
    full = ops.rotate_atoms(with_co, pair, 360.0).interface.positions
    assert np.allclose(full, with_co.interface.positions, atol=1e-8)


def test_rotate_atoms_about_explicit_anchor(cu111):
    moved = ops.rotate_atoms(cu111, [0], 180.0, axis=(0, 0, 1), anchor=(0, 0, 0))
    original = cu111.interface.positions[0]
    assert np.allclose(
        moved.interface.positions[0], [-original[0], -original[1], original[2]]
    )


def test_constraints_accumulate_and_survive_edits(cu111):
    with_co = ops.place_adsorbate(
        cu111, "CO", ops.find_adsorption_sites(cu111)[0].coords
    )
    first, second = with_co.adsList[0]
    result = ops.add_bond_constraint(with_co, first, second)
    result = ops.add_bond_constraint(result, second, first)  # duplicate is ignored
    result = ops.add_bond_constraint(result, 0, first)
    result = ops.add_com_constraint(result, [first, second])
    result = ops.add_wall(result, [first, second], height=20.0)
    summary = ops.describe_constraints(result)
    assert summary["bonds"] == [[first, second], [0, first]]
    assert summary["com"] == [[first, second]]
    assert [wall["index"] for wall in summary["walls"]] == [first, second]
    assert ops.describe_constraints(with_co) == {"bonds": [], "com": [], "walls": []}

    for edited in (
        ops.translate_atoms(result, [first], [0.0, 0.0, 0.1]),
        ops.substitute_atoms(result, [3], "Pt"),
        ops.set_fixed(result, [result.relax[0]], True),
        ops.fix_bottom_layers(result, 2),
        ops.place_adsorbate(result, "O", (1.0, 1.0, 20.0)),
    ):
        assert ops.describe_constraints(edited) == summary
        assert edited.fix  # frozen atoms are kept alongside
    # Deleting an atom re-indexes the surviving constraints and drops those that used it.
    after_delete = ops.describe_constraints(ops.delete_atoms(result, [0]))
    assert after_delete["bonds"] == [
        [first - 1, second - 1]
    ]  # the (0, first) bond is gone
    assert after_delete["com"] == [[first - 1, second - 1]]
    assert [wall["index"] for wall in after_delete["walls"]] == [first - 1, second - 1]
    without_first = ops.describe_constraints(ops.delete_atoms(result, [first]))
    assert without_first["bonds"] == [] and without_first["com"] == [[second - 1]]
    assert [wall["index"] for wall in without_first["walls"]] == [second - 1]
    cleared = ops.clear_constraints(result)
    assert ops.extra_constraints(cleared) == []
    assert cleared.fix == result.fix


def test_wall_pushes_back_only_above_the_plane(cu111):
    from ase.calculators.emt import EMT

    with_o = ops.place_adsorbate(cu111, "O", ops.find_adsorption_sites(cu111)[0].coords)
    oxygen = len(with_o.interface) - 1
    z = with_o.interface.positions[oxygen, 2]

    def force_z(height):
        atoms = ops.add_wall(
            with_o, [oxygen], height=height, spring_constant=4.0
        ).interface
        atoms.calc = EMT()
        return atoms.get_forces()[oxygen, 2]

    free = with_o.interface.copy()
    free.calc = EMT()
    unconstrained = free.get_forces()[oxygen, 2]
    assert force_z(z + 1.0) == pytest.approx(unconstrained)
    assert force_z(z - 0.5) == pytest.approx(unconstrained - 4.0 * 0.5)


def test_bond_constraint_needs_two_atoms(cu111):
    with pytest.raises(ValueError, match="two different atoms"):
        ops.add_bond_constraint(cu111, 1, 1)


@pytest.mark.parametrize("position", ["center", "bottom"])
def test_adjust_vacuum_moves_everything_rigidly(cu111, position):
    with_co = ops.place_adsorbate(
        cu111, "CO", ops.find_adsorption_sites(cu111)[0].coords
    )
    first, second = with_co.adsList[0]
    start = ops.add_bond_constraint(with_co, first, second)
    start = ops.add_wall(
        start, [second], height=start.interface.positions[second, 2] + 2.0
    )
    before = start.interface
    thickness = np.ptp(before.positions[:, 2])

    result = ops.adjust_vacuum(start, 20.0, position=position)
    after = result.interface
    assert after.cell[2, 2] == pytest.approx(thickness + 20.0)
    assert np.allclose(after.cell[:2], before.cell[:2])
    lowest = after.positions[:, 2].min()
    expected_lowest = ops.BOTTOM_MARGIN if position == "bottom" else 10.0
    assert lowest == pytest.approx(expected_lowest)
    assert after.positions[:, 2].max() == pytest.approx(expected_lowest + thickness)

    # A rigid shift along z only: in-plane coordinates and all interatomic vectors survive.
    shift = after.positions - before.positions
    assert np.allclose(shift[:, :2], 0.0, atol=1e-9)
    assert np.allclose(shift[:, 2], shift[0, 2], atol=1e-9)

    # Partition, frozen atoms, and constraints are kept; the wall moves with the atoms.
    assert result.fix == start.fix and result.adsList == start.adsList
    assert list(after.constraints[0].index) == start.fix
    summary, original = (
        ops.describe_constraints(result),
        ops.describe_constraints(start),
    )
    assert summary["bonds"] == original["bonds"]
    assert summary["walls"][0]["height"] == pytest.approx(
        original["walls"][0]["height"] + shift[0, 2]
    )
    assert len(start.interface) == len(before) and np.allclose(
        start.interface.positions, before.positions
    )


def test_adjust_vacuum_roundtrip_and_errors(cu111):
    wide = ops.adjust_vacuum(cu111, 30.0, position="bottom")
    back = ops.adjust_vacuum(wide, 12.0, position="center")
    assert np.allclose(back.interface.positions, cu111.interface.positions, atol=1e-8)
    assert np.allclose(back.interface.cell, cu111.interface.cell, atol=1e-8)
    with pytest.raises(ValueError, match="unknown position"):
        ops.adjust_vacuum(cu111, 15.0, position="top")
    with pytest.raises(ValueError, match="positive"):
        ops.adjust_vacuum(cu111, 0.0)
    with pytest.raises(ValueError, match="margin"):
        ops.adjust_vacuum(cu111, 0.5, position="bottom")


def test_list_and_remove_items(cu111):
    sites = ops.find_adsorption_sites(cu111)
    system = ops.place_adsorbate(cu111, "CO", sites[0].coords)
    system = ops.place_adsorbate(system, "OH", sites[20].coords)
    n_slab = len(cu111.interface)
    system = ops.add_bond_constraint(system, n_slab, n_slab + 1)
    system = ops.add_com_constraint(system, [n_slab + 2, n_slab + 3])
    system = ops.add_wall(system, [n_slab + 1, n_slab + 3], height=25.0)

    items = {item["key"]: item for item in ops.list_items(system)}
    assert list(items) == [
        "substrate",
        "fixed",
        "ads:0",
        "ads:1",
        "bond:0",
        "com:0",
        "wall:0",
    ]
    assert not items["substrate"]["removable"] and all(
        items[k]["removable"] for k in list(items)[1:]
    )
    assert (
        items["ads:0"]["atoms"] == [n_slab, n_slab + 1]
        and "CO" in items["ads:0"]["label"]
    )
    assert (
        items["wall:0"]["atoms"] == [n_slab + 1, n_slab + 3]
        and "25.00" in items["wall:0"]["label"]
    )
    assert items["fixed"]["atoms"] == system.fix

    # Removing the first adsorbate drops its bond and its wall atom; the rest is re-indexed.
    less = ops.remove_items(system, ["ads:0", "nope", "substrate"])
    keys = [item["key"] for item in ops.list_items(less)]
    assert keys == ["substrate", "fixed", "ads:0", "com:0", "wall:0"]
    assert len(less.interface) == n_slab + 2 and len(less.substrate) == n_slab
    summary = ops.describe_constraints(less)
    assert summary["com"] == [[n_slab, n_slab + 1]]
    assert [wall["index"] for wall in summary["walls"]] == [n_slab + 1]

    only_constraints = ops.remove_items(system, ["fixed", "bond:0", "com:0", "wall:0"])
    assert [item["key"] for item in ops.list_items(only_constraints)] == [
        "substrate",
        "ads:0",
        "ads:1",
    ]
    assert only_constraints.fix == [] and only_constraints.interface.constraints == []
    assert len(system.interface) == n_slab + 4 and system.fix  # input untouched
