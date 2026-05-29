"""Tests for the Interface class."""

from __future__ import annotations

import pytest
from ase import Atoms
from ase.build import bulk
from ase.constraints import FixAtoms
from pymatgen.core import Structure

from gs_gmlip.interface import Interface


def _slab():
    # two-layer 2x2 H "slab" stand-in (cheap, no calculator needed)
    pos = [[x, y, z] for z in (0.0, 2.0) for x in (0.0, 2.0) for y in (0.0, 2.0)]
    return Atoms("H8", positions=pos, cell=[4, 4, 12], pbc=[True, True, False])


def test_init_bare_substrate():
    intf = Interface(substrate=_slab())
    assert len(intf.substrate) == 8
    assert len(intf.interface) == 8
    assert intf.adsList == []
    assert intf.clusterList == []
    assert intf.fix == []
    assert sorted(intf.relax) == list(range(8))


def test_init_with_adsorbate_classifies():
    slab = _slab()
    full = slab + Atoms("CO", positions=[[2.0, 2.0, 4.0], [2.0, 2.0, 5.13]])
    intf = Interface(substrate=slab, interface=full)
    assert intf.clusterList == []
    assert sorted(intf.adsList[0]) == [8, 9]


def test_init_fixlist_validation():
    with pytest.raises(ValueError):
        Interface(substrate=_slab(), fixlist=[99])


def test_init_reads_existing_fixatoms_constraint():
    slab = _slab()
    slab.set_constraint(FixAtoms(indices=[0, 1, 2, 3]))
    intf = Interface(substrate=slab)
    assert intf.fix == [0, 1, 2, 3]
    # relax is the complement
    assert intf.relax == [4, 5, 6, 7]


def test_repr_contains_interface():
    intf = Interface(substrate=_slab())
    assert "Interface" in repr(intf)


def test_fix_is_flat_int_list_from_fixlist():
    intf = Interface(substrate=_slab(), fixlist=[2, 0, 1])
    assert intf.fix == [0, 1, 2]  # sorted, deduped, plain ints
    assert all(isinstance(i, int) for i in intf.fix)


def test_fix_matches_between_construction_paths():
    # fixlist path and constraint path should yield identical flat fix lists
    from ase.constraints import FixAtoms as _Fix

    a = Interface(substrate=_slab(), fixlist=[0, 1, 2, 3])
    slab = _slab()
    slab.set_constraint(_Fix(indices=[0, 1, 2, 3]))
    b = Interface(substrate=slab)
    assert a.fix == b.fix == [0, 1, 2, 3]


def test_relaxlist_explicit():
    intf = Interface(substrate=_slab(), relaxlist=[4, 5])
    assert intf.relax == [4, 5]


def test_cached_properties_present_and_cached():
    slab = _slab()
    full = slab + Atoms("CO", positions=[[2.0, 2.0, 4.0], [2.0, 2.0, 5.13]])
    intf = Interface(substrate=slab, interface=full)
    bm1 = intf.bondmatrix
    assert bm1.shape == (10, 10)
    assert intf.bondmatrix is bm1  # cached: identical object on 2nd access
    assert isinstance(intf.possible_bond, dict)
    import numpy as np
    assert isinstance(intf.fingerprint, np.ndarray)


def test_reset_cache_clears_entries():
    intf = Interface(substrate=_slab())
    _ = intf.bondmatrix
    assert "bondmatrix" in intf.__dict__
    intf._reset_cache()
    assert "bondmatrix" not in intf.__dict__


def test_get_substrate_layers_two():
    intf = Interface(substrate=_slab())  # _slab has z=0 and z=2 -> 2 layers
    assert intf.get_substrate_layers() == 2


def test_fix_substrate_int_fixes_bottom_layer():
    intf = Interface(substrate=_slab())
    intf.fix_substrate(layers=1)  # fix bottom 1 layer (z=0 -> indices 0..3)
    assert sorted(intf.fix) == [0, 1, 2, 3]
    assert any(isinstance(c, FixAtoms) for c in intf.substrate.constraints)


def test_fix_substrate_float_fixes_below_height():
    intf = Interface(substrate=_slab())
    intf.fix_substrate(layers=1.0)  # float -> fix atoms with z <= 1.0 (the z=0 layer)
    assert sorted(intf.fix) == [0, 1, 2, 3]


def test_fix_substrate_resets_cache():
    intf = Interface(substrate=_slab())
    _ = intf.bondmatrix
    assert "bondmatrix" in intf.__dict__
    intf.fix_substrate(layers=1)
    assert "bondmatrix" not in intf.__dict__


def test_build_from_bulk_sets_substrate():
    intf = Interface()
    intf.build_from_bulk(bulk("Pt", "fcc", a=3.92), (1, 1, 1), layer=3, vacuum=12.0)
    assert len(intf.substrate) > 0
    assert len(intf.interface) == len(intf.substrate)


def test_build_primitive_surface_sets_substrate():
    intf = Interface()
    intf.build_primitive_surface_from_bulk(
        bulk("Pt", "fcc", a=3.92), (1, 1, 1), layer=3, vacuum=12.0
    )
    assert len(intf.substrate) > 0
    assert len(intf.interface) == len(intf.substrate)


def test_fix_substrate_zero_fixes_nothing():
    intf = Interface(substrate=_slab())
    intf.fix_substrate(layers=0)
    assert intf.fix == []
    assert intf.relax == list(range(8))


def test_fix_substrate_all_layers_fixes_everything():
    intf = Interface(substrate=_slab())  # 2 layers
    intf.fix_substrate(layers=5)  # more than n_layers -> clamp to all
    assert intf.fix == list(range(8))
    assert intf.relax == []


def test_fix_substrate_leaves_adsorbates_free():
    # the interface FixAtoms must cover ONLY substrate indices; adsorbates stay free
    slab = _slab()
    full = slab + Atoms("CO", positions=[[2.0, 2.0, 4.0], [2.0, 2.0, 5.13]])
    intf = Interface(substrate=slab, interface=full)
    intf.fix_substrate(layers=1)  # fix bottom substrate layer (indices 0-3)
    fixed_on_interface = set()
    for c in intf.interface.constraints:
        if isinstance(c, FixAtoms):
            fixed_on_interface.update(int(i) for i in c.index)
    assert fixed_on_interface == {0, 1, 2, 3}
    # adsorbate indices 8,9 are NOT fixed
    assert 8 not in fixed_on_interface and 9 not in fixed_on_interface


def test_build_from_bulk_clears_adsorbate_state():
    slab = _slab()
    full = slab + Atoms("CO", positions=[[2.0, 2.0, 4.0], [2.0, 2.0, 5.13]])
    intf = Interface(substrate=slab, interface=full)
    assert intf.adsList != []  # has an adsorbate before rebuild
    intf.build_from_bulk(bulk("Pt", "fcc", a=3.92), (1, 1, 1), layer=3, vacuum=12.0)
    assert intf.adsList == []
    assert intf.clusterList == []
    assert intf.fix == []


def test_wrap_resets_cache_and_wraps():
    slab = _slab()
    slab.positions[0] += slab.cell[0]  # push atom 0 outside cell along a
    intf = Interface(substrate=slab.copy(), interface=slab.copy())
    intf.interface.set_pbc([True, True, False])
    _ = intf.bondmatrix
    intf.wrap()
    assert "bondmatrix" not in intf.__dict__
    # atom 0 wrapped back inside the a-cell extent
    assert intf.interface.get_positions()[0, 0] < slab.cell[0, 0]


def test_align_bottom_places_adsorbate_exactly_two_above_slab_top():
    slab = _slab()  # slab spans z=0..2, top = 2.0
    full = slab + Atoms("C", positions=[[2.0, 2.0, 8.0]])
    intf = Interface(substrate=slab, interface=full)
    intf.align_interface(loc="bottom")
    # appended min-z should sit exactly slab_top + 2.0 = 4.0
    assert intf.interface.get_positions()[8, 2] == pytest.approx(4.0)


def test_align_center_places_mean_at_slab_midpoint():
    slab = _slab()  # z midpoint = (0+2)/2 = 1.0
    full = slab + Atoms("C", positions=[[2.0, 2.0, 8.0]])
    intf = Interface(substrate=slab, interface=full)
    intf.align_interface(loc="center")
    assert intf.interface.get_positions()[8, 2] == pytest.approx(1.0)


def test_align_leaves_substrate_positions_unchanged():
    slab = _slab()
    full = slab + Atoms("C", positions=[[2.0, 2.0, 8.0]])
    intf = Interface(substrate=slab, interface=full)
    before = intf.interface.get_positions()[:8].copy()
    intf.align_interface(loc="bottom")
    after = intf.interface.get_positions()[:8]
    assert (before == after).all()


def test_align_preserves_adsorbate_internal_spacing():
    # a 2-atom adsorbate must translate rigidly (internal z-gap preserved)
    slab = _slab()
    full = slab + Atoms("CO", positions=[[2.0, 2.0, 8.0], [2.0, 2.0, 9.13]])
    intf = Interface(substrate=slab, interface=full)
    before_gap = (
        full.get_positions()[9, 2] - full.get_positions()[8, 2]
    )
    intf.align_interface(loc="bottom")
    pos = intf.interface.get_positions()
    after_gap = pos[9, 2] - pos[8, 2]
    assert after_gap == pytest.approx(before_gap)


def test_align_interface_invalid_loc_raises():
    with pytest.raises(ValueError):
        # no appended atoms -> early return, so add one to reach the loc check
        slab = _slab()
        full = slab + Atoms("C", positions=[[2.0, 2.0, 8.0]])
        bad = Interface(substrate=slab, interface=full)
        bad.align_interface(loc="nonsense")


def test_align_interface_noop_without_adsorbate():
    intf = Interface(substrate=_slab())
    # no appended atoms: align should be a no-op and not raise
    intf.align_interface(loc="bottom")
    assert len(intf.interface) == 8


def test_to_ase_part_selection():
    slab = _slab()
    full = slab + Atoms("CO", positions=[[2.0, 2.0, 4.0], [2.0, 2.0, 5.13]])
    intf = Interface(substrate=slab, interface=full)
    assert len(intf.to_ase()) == 10
    assert len(intf.to_ase(part="substrate")) == 8


def test_to_ase_returns_copy():
    intf = Interface(substrate=_slab())
    a = intf.to_ase()
    a.positions[0] += 5.0
    assert not (a.get_positions()[0] == intf.interface.get_positions()[0]).all()


def test_to_pymatgen_returns_structure():
    slab = _slab()
    slab.set_pbc(True)
    intf = Interface(substrate=slab)
    struct = intf.to_pymatgen()
    assert isinstance(struct, Structure)
    sub = intf.to_pymatgen(part="substrate")
    assert isinstance(sub, Structure)


def test_to_ase_invalid_part_raises():
    intf = Interface(substrate=_slab())
    with pytest.raises(ValueError):
        intf.to_ase(part="bogus")


def test_dict_roundtrip():
    slab = _slab()
    full = slab + Atoms("CO", positions=[[2.0, 2.0, 4.0], [2.0, 2.0, 5.13]])
    intf = Interface(substrate=slab, interface=full)
    d = intf.to_dict()
    intf2 = Interface.from_dict(d)
    assert len(intf2.interface) == 10
    assert len(intf2.substrate) == 8
    assert intf2.adsList == intf.adsList
    assert intf2.fix == intf.fix


def test_to_dict_is_json_serializable():
    import json

    slab = _slab()
    full = slab + Atoms("CO", positions=[[2.0, 2.0, 4.0], [2.0, 2.0, 5.13]])
    intf = Interface(substrate=slab, interface=full)
    s = json.dumps(intf.to_dict())  # must not raise
    assert isinstance(s, str)


def test_from_ase_splits_at_n_substrate():
    slab = _slab()
    full = slab + Atoms("CO", positions=[[2.0, 2.0, 4.0], [2.0, 2.0, 5.13]])
    intf = Interface.from_ase(full, n_substrate=8)
    assert len(intf.substrate) == 8
    assert sorted(intf.adsList[0]) == [8, 9]


def test_from_pymatgen_roundtrip():
    slab = _slab()
    slab.set_pbc(True)
    intf = Interface(substrate=slab)
    struct = intf.to_pymatgen()
    rebuilt = Interface.from_pymatgen(struct, n_substrate=len(struct))
    assert len(rebuilt.substrate) == 8


def test_copy_is_independent():
    intf = Interface(substrate=_slab())
    clone = intf.copy()
    clone.interface.positions[0] += 1.0
    assert not (
        clone.interface.get_positions()[0] == intf.interface.get_positions()[0]
    ).all()


def test_copy_preserves_grouped_lists():
    slab = _slab()
    full = slab + Atoms("CO", positions=[[2.0, 2.0, 4.0], [2.0, 2.0, 5.13]])
    intf = Interface(substrate=slab, interface=full)
    clone = intf.copy()
    assert clone.adsList == intf.adsList
    assert clone.fix == intf.fix


def test_dict_roundtrip_preserves_empty_fix_over_constrained_slab():
    # an interface with explicit fix=[] but whose Atoms carries a FixAtoms
    # constraint must round-trip to fix=[] (not re-derive from the constraint)
    slab = _slab()
    slab.set_constraint(FixAtoms(indices=[0, 1]))
    intf = Interface(substrate=slab, fixlist=[])  # explicit empty fix
    assert intf.fix == []
    intf2 = Interface.from_dict(intf.to_dict())
    assert intf2.fix == []


def test_copy_preserves_empty_fix_over_constrained_slab():
    slab = _slab()
    slab.set_constraint(FixAtoms(indices=[0, 1]))
    intf = Interface(substrate=slab, fixlist=[])
    assert intf.fix == []
    assert intf.copy().fix == []


def test_dict_roundtrip_bare_slab():
    intf = Interface(substrate=_slab())
    intf2 = Interface.from_dict(intf.to_dict())
    assert len(intf2.interface) == 8
    assert len(intf2.substrate) == 8
    assert intf2.adsList == []
    assert intf2.clusterList == []
    assert intf2.fix == []
    assert intf2.relax == list(range(8))


def test_write_read_roundtrip(tmp_path):
    slab = _slab()
    full = slab + Atoms("CO", positions=[[2.0, 2.0, 4.0], [2.0, 2.0, 5.13]])
    intf = Interface(substrate=slab, interface=full)
    path = tmp_path / "intf.traj"
    intf.write(str(path))
    loaded = Interface.read(str(path))
    assert len(loaded.interface) == 10
    assert len(loaded.substrate) == 8
    assert loaded.adsList == intf.adsList
    assert loaded.fix == intf.fix


def test_db_roundtrip(tmp_path):
    from ase.db import connect

    slab = _slab()
    full = slab + Atoms("CO", positions=[[2.0, 2.0, 4.0], [2.0, 2.0, 5.13]])
    intf = Interface(substrate=slab, interface=full)
    db = connect(str(tmp_path / "test.db"))
    intf.write_db(db, name="trial")
    row = db.get(name="trial")
    loaded = Interface.from_db_row(row)
    assert len(loaded.substrate) == 8
    assert loaded.adsList == intf.adsList
    assert loaded.fix == intf.fix


def test_db_roundtrip_bare_slab(tmp_path):
    from ase.db import connect

    intf = Interface(substrate=_slab())
    db = connect(str(tmp_path / "bare.db"))
    intf.write_db(db, name="bare")
    loaded = Interface.from_db_row(db.get(name="bare"))
    assert loaded.adsList == []
    assert loaded.clusterList == []
    assert loaded.fix == []


def test_write_read_preserves_relax_and_custom_grouping(tmp_path):
    slab = _slab()
    full = slab + Atoms("CO", positions=[[2.0, 2.0, 4.0], [2.0, 2.0, 5.13]])
    # force a custom (non-derived) grouping + fix to prove they round-trip verbatim
    intf = Interface(
        substrate=slab,
        interface=full,
        fixlist=[0, 1],
        adsList=[[8], [9]],
        clusterList=[],
    )
    path = tmp_path / "custom.traj"
    intf.write(str(path))
    loaded = Interface.read(str(path))
    assert loaded.fix == [0, 1]
    assert loaded.relax == intf.relax
    assert loaded.adsList == [[8], [9]]
    assert loaded.clusterList == []
    # leftover metadata key removed from reconstructed interface.info
    assert "gs_gmlip_interface" not in loaded.interface.info


def test_read_plain_file_without_metadata(tmp_path):
    from ase.io import write as _w

    # a plain Atoms file with NO gs_gmlip metadata -> everything is substrate
    plain = _slab()
    p = tmp_path / "plain.traj"
    _w(str(p), plain)
    loaded = Interface.read(str(p))
    assert len(loaded.substrate) == 8
    assert len(loaded.interface) == 8
    assert loaded.adsList == []
    assert loaded.clusterList == []


def test_to_dict_includes_metadata_and_interface():
    intf = Interface(substrate=_slab())
    d = intf.to_dict()
    # to_dict is _metadata() plus the encoded interface
    for key in (
        "n_substrate",
        "fix",
        "relax",
        "adsList",
        "clusterList",
        "split_mol_on_cluster",
        "interface",
    ):
        assert key in d


def test_add_adsorbate_appends_and_reclassifies():
    intf = Interface(substrate=_slab())
    co = Atoms("CO", positions=[[2.0, 2.0, 4.0], [2.0, 2.0, 5.13]])
    new_idx = intf.add_adsorbate(co)
    assert new_idx == [8, 9]
    assert len(intf.interface) == 10
    assert sorted(intf.adsList[0]) == [8, 9]


def test_add_adsorbate_with_offset():
    intf = Interface(substrate=_slab())
    o = Atoms("O", positions=[[0.0, 0.0, 0.0]])
    new_idx = intf.add_adsorbate(o, offset=[2.0, 2.0, 4.0])
    assert new_idx == [8]
    # the atom landed at the offset position
    assert (intf.interface.get_positions()[8] == [2.0, 2.0, 4.0]).all()


def test_add_adsorbate_resets_cache():
    intf = Interface(substrate=_slab())
    _ = intf.bondmatrix
    assert "bondmatrix" in intf.__dict__
    intf.add_adsorbate(Atoms("O", positions=[[2.0, 2.0, 4.0]]))
    assert "bondmatrix" not in intf.__dict__


def test_remove_group_removes_atoms_and_reclassifies():
    slab = _slab()
    full = slab + Atoms("CO", positions=[[2.0, 2.0, 4.0], [2.0, 2.0, 5.13]])
    intf = Interface(substrate=slab, interface=full)
    assert sorted(intf.adsList[0]) == [8, 9]
    intf.remove_group([8, 9])
    assert len(intf.interface) == 8
    assert intf.adsList == []


def test_remove_group_rejects_substrate_indices():
    intf = Interface(substrate=_slab())
    with pytest.raises(ValueError):
        intf.remove_group([0])  # cannot remove substrate atoms


def test_remove_group_resets_cache():
    slab = _slab()
    full = slab + Atoms("CO", positions=[[2.0, 2.0, 4.0], [2.0, 2.0, 5.13]])
    intf = Interface(substrate=slab, interface=full)
    _ = intf.bondmatrix
    intf.remove_group([8, 9])
    assert "bondmatrix" not in intf.__dict__


def test_add_adsorbate_after_fix_preserves_constraint_and_frees_adsorbate():
    # load-bearing: after fixing the substrate then adding an adsorbate,
    # the substrate FixAtoms must remain and the new atoms must stay free
    intf = Interface(substrate=_slab())
    intf.fix_substrate(layers=1)  # fixes substrate indices 0..3
    intf.add_adsorbate(Atoms("CO", positions=[[2.0, 2.0, 4.0], [2.0, 2.0, 5.13]]))
    fixed = set()
    for c in intf.interface.constraints:
        if isinstance(c, FixAtoms):
            fixed.update(int(i) for i in c.index)
    assert fixed == {0, 1, 2, 3}  # substrate still fixed
    assert 8 not in fixed and 9 not in fixed  # adsorbate free


def test_remove_group_out_of_range_raises():
    slab = _slab()
    full = slab + Atoms("CO", positions=[[2.0, 2.0, 4.0], [2.0, 2.0, 5.13]])
    intf = Interface(substrate=slab, interface=full)
    with pytest.raises(ValueError):
        intf.remove_group([99])


def test_remove_partial_adsorbate_reclassifies_remainder():
    # remove one atom of a CO; the remaining lone C is re-classified
    slab = _slab()
    full = slab + Atoms("CO", positions=[[2.0, 2.0, 4.0], [2.0, 2.0, 5.13]])
    intf = Interface(substrate=slab, interface=full)
    intf.remove_group([9])  # drop the O, leaving lone C at index 8
    assert len(intf.interface) == 9
    # lone non-metal C with no metal nearby -> a size-1 adsorbate group
    assert intf.adsList == [[8]]
    assert intf.clusterList == []


def test_phase2_stubs_raise_not_implemented():
    intf = Interface(substrate=_slab())
    with pytest.raises(NotImplementedError):
        intf.find_all_ads_sites()
    with pytest.raises(NotImplementedError):
        intf.find_simple_ads_sites()
    with pytest.raises(NotImplementedError):
        intf.identify_symmetry()


def test_from_dict_defaults_n_substrate_when_missing():
    # a dict lacking n_substrate should not KeyError; everything treated as substrate
    intf = Interface(substrate=_slab())
    d = intf.to_dict()
    del d["n_substrate"]  # simulate a hand-built / partial dict
    rebuilt = Interface.from_dict(d)
    assert len(rebuilt.substrate) == 8  # defaulted to len(atoms)
    assert len(rebuilt.interface) == 8
