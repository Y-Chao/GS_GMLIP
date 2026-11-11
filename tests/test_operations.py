"""Tests for the pymatgen-based builder operations."""

from __future__ import annotations

import numpy as np
import pytest

from gs_gmlip.interface import operations as ops


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
    assert np.allclose(mol.cart_coords.min(axis=0)[2], 0.0, atol=0.5)
    # The binding atom sits at the origin.
    assert np.isclose(np.linalg.norm(mol.cart_coords, axis=1).min(), 0.0, atol=1e-8)


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
