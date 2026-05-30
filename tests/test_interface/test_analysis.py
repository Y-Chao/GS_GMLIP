"""Tests for gs_gmlip.interface.analysis."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from gs_gmlip.interface.analysis import (
    build_neighbor_graph,
    connected_components,
    is_metal,
)
from gs_gmlip.interface.analysis import classify_appended
from gs_gmlip.interface.analysis import (
    compute_bondmatrix,
    compute_possible_bond,
    expected_bonds,
)
from gs_gmlip.interface.analysis import detect_layers, fingerprint


def test_is_metal():
    assert is_metal("Pt") is True
    assert is_metal("Fe") is True
    assert is_metal("O") is False
    assert is_metal("C") is False
    assert is_metal("Si") is False  # metalloid -> not a metal


def test_build_neighbor_graph_diatomic():
    co = Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.13]], cell=[10, 10, 10])
    g = build_neighbor_graph(co)
    assert g.has_edge(0, 1)


def test_connected_components_two_molecules():
    atoms = Atoms(
        "COOO",
        positions=[
            [0, 0, 0],
            [0, 0, 1.13],
            [5, 5, 5],
            [5, 5, 6.2],
        ],
        cell=[10, 10, 10],
    )
    g = build_neighbor_graph(atoms)
    assert not g.has_edge(0, 2)
    assert not g.has_edge(1, 3)
    comps = connected_components(g)
    comps_sorted = sorted([sorted(c) for c in comps])
    assert comps_sorted == [[0, 1], [2, 3]]


def test_build_neighbor_graph_periodic_wrap():
    # Two O atoms across a periodic boundary: 0.6 Å apart via the image,
    # so they should bond under MIC even though their raw separation is large.
    atoms = Atoms(
        "OO",
        positions=[[0.3, 0.0, 0.0], [4.7, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    g = build_neighbor_graph(atoms)
    assert g.has_edge(0, 1)


def test_build_neighbor_graph_single_atom():
    atoms = Atoms("O", positions=[[0, 0, 0]], cell=[10, 10, 10])
    g = build_neighbor_graph(atoms)
    assert list(g.nodes()) == [0]
    assert g.number_of_edges() == 0


def _pt4():
    # compact Pt4 cluster, all within bonding distance
    return Atoms(
        "Pt4",
        positions=[[0, 0, 0], [2.6, 0, 0], [1.3, 2.25, 0], [1.3, 0.75, 2.1]],
        cell=[15, 15, 15],
    )


def test_classify_bare_metal_cluster():
    ads, clu = classify_appended(_pt4())
    assert ads == []
    assert sorted(clu[0]) == [0, 1, 2, 3]


def test_classify_free_molecule_no_metal():
    h2o = Atoms(
        "OHH",
        positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]],
        cell=[15, 15, 15],
    )
    ads, clu = classify_appended(h2o)
    assert clu == []
    assert sorted(ads[0]) == [0, 1, 2]


def test_classify_co_on_cluster_splits():
    atoms = _pt4()
    atoms += Atoms("CO", positions=[[0, 0, 2.0], [0, 0, 3.13]])
    ads, clu = classify_appended(atoms, split_mol_on_cluster=True)
    assert sorted(ads[0]) == [4, 5]
    assert sorted(clu[0]) == [0, 1, 2, 3]


def test_classify_atomic_o_on_cluster_joins_cluster():
    atoms = _pt4()
    atoms += Atoms("O", positions=[[0, 0, 1.9]])
    ads, clu = classify_appended(atoms, split_mol_on_cluster=True)
    assert ads == []
    assert 4 in clu[0]


def test_classify_no_split_keeps_whole_component():
    atoms = _pt4()
    atoms += Atoms("CO", positions=[[0, 0, 2.0], [0, 0, 3.13]])
    ads, clu = classify_appended(atoms, split_mol_on_cluster=False)
    assert ads == []
    assert sorted(clu[0]) == [0, 1, 2, 3, 4, 5]


def test_classify_two_adsorbates_on_one_cluster():
    # two separate CO molecules adsorbed on one Pt4 -> two adsorbate groups, one cluster
    atoms = _pt4()
    atoms += Atoms("CO", positions=[[0, 0, 2.0], [0, 0, 3.13]])
    atoms += Atoms("CO", positions=[[2.6, 0, 2.0], [2.6, 0, 3.13]])
    ads, clu = classify_appended(atoms, split_mol_on_cluster=True)
    groups = sorted(sorted(g) for g in ads)
    assert groups == [[4, 5], [6, 7]]
    assert sorted(clu[0]) == [0, 1, 2, 3]


def test_classify_lone_nonmetal_no_metal_is_adsorbate():
    # a single O with no metal anywhere -> a size-1 adsorbate group (NOT a cluster)
    o = Atoms("O", positions=[[0, 0, 0]], cell=[15, 15, 15])
    ads, clu = classify_appended(o)
    assert clu == []
    assert ads == [[0]]


def test_classify_empty_returns_empty():
    from ase import Atoms as _A
    ads, clu = classify_appended(_A())
    assert ads == []
    assert clu == []


def test_expected_bonds_closed_shell():
    assert expected_bonds(1) == 1   # H -> |1-2|
    assert expected_bonds(8) == 2   # O -> |8-10|
    assert expected_bonds(7) == 3   # N -> |7-10|
    assert expected_bonds(6) == 4   # C -> min(|6-2|,|6-10|)
    assert expected_bonds(78) == 8  # Pt -> |78-86|


def test_compute_bondmatrix_symmetric_binary():
    co = Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.13]], cell=[10, 10, 10])
    m = compute_bondmatrix(co)
    assert m.shape == (2, 2)
    assert m[0, 1] == 1 and m[1, 0] == 1
    assert m[0, 0] == 0


def test_compute_possible_bond_water():
    h2o = Atoms(
        "OHH",
        positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]],
        cell=[10, 10, 10],
    )
    pb = compute_possible_bond(h2o)
    assert pb == {0: 0, 1: 0, 2: 0}


def test_compute_possible_bond_lone_oxygen():
    o = Atoms("O", positions=[[0, 0, 0]], cell=[10, 10, 10])
    pb = compute_possible_bond(o)
    assert pb == {0: 2}


def test_compute_possible_bond_methane_saturated():
    # CH4: central C has 4 bonds (expected 4 -> 0 free); each H has 1 bond (expected 1 -> 0)
    ch4 = Atoms(
        "CH4",
        positions=[
            [0.00, 0.00, 0.00],
            [0.63, 0.63, 0.63],
            [-0.63, -0.63, 0.63],
            [-0.63, 0.63, -0.63],
            [0.63, -0.63, -0.63],
        ],
        cell=[10, 10, 10],
    )
    pb = compute_possible_bond(ch4)
    assert pb[0] == 0  # carbon saturated
    assert all(pb[i] == 0 for i in range(1, 5))  # each H saturated


def test_detect_layers_three_layers():
    z = [0.0, 0.05, 0.0, 2.0, 2.0, 1.95, 4.0, 4.0, 4.0]
    atoms = Atoms("H9", positions=[[0, 0, zi] for zi in z], cell=[10, 10, 10])
    labels = detect_layers(atoms, tol=0.5)
    assert labels.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_detect_layers_single_atom():
    atoms = Atoms("H", positions=[[0, 0, 1.0]], cell=[10, 10, 10])
    assert detect_layers(atoms).tolist() == [0]


def test_detect_layers_tol_boundary_not_split():
    # gap exactly == tol must NOT start a new layer (strict > comparison)
    atoms = Atoms("H2", positions=[[0, 0, 0.0], [0, 0, 0.5]], cell=[10, 10, 10])
    assert detect_layers(atoms, tol=0.5).tolist() == [0, 0]


def test_detect_layers_cumulative_drift_lumps_one_layer():
    # documents the known limitation: each consecutive gap (0.4) <= tol (0.5),
    # so all atoms collapse into one layer even though total spread is 1.2 Å
    atoms = Atoms(
        "H4",
        positions=[[0, 0, 0.0], [0, 0, 0.4], [0, 0, 0.8], [0, 0, 1.2]],
        cell=[10, 10, 10],
    )
    assert detect_layers(atoms, tol=0.5).tolist() == [0, 0, 0, 0]


def test_fingerprint_is_deterministic_and_order_invariant():
    a = Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.13]], cell=[10, 10, 10])
    b = Atoms("OC", positions=[[0, 0, 1.13], [0, 0, 0]], cell=[10, 10, 10])
    fa, fb = fingerprint(a), fingerprint(b)
    assert isinstance(fa, np.ndarray)
    assert np.allclose(fa, fb)


def test_classify_co_on_cu100_real_slab(cu100_with_co, cu100_slab):
    # Cu(100) + CO: appended is just the CO (indices in the FULL structure are after the slab).
    # classify_appended takes the appended region only; check it as if isolated.
    appended = cu100_with_co[len(cu100_slab):]
    ads, clu = classify_appended(appended)
    assert ads == [[0, 1]]
    assert clu == []


def test_classify_two_co_on_cu100_two_adsorbate_groups(cu100_with_two_co, cu100_slab):
    appended = cu100_with_two_co[len(cu100_slab):]
    ads, clu = classify_appended(appended)
    assert sorted(ads) == [[0, 1], [2, 3]]
    assert clu == []


def test_classify_lone_co_on_graphene_is_cluster(cocn_sheet):
    # Treating the graphene+N as substrate, just the Co (last atom) as appended.
    appended = cocn_sheet[158:]
    ads, clu = classify_appended(appended)
    assert ads == []
    assert clu == [[0]]


def test_classify_full_cocn_documents_cutoff_limitation(cocn_sheet):
    # Passing the whole Co-N4 sheet as appended highlights the neighbor-graph
    # cutoff limitation: the Co atom is NOT bonded to the N at this geometry under
    # the simple covalent-radius cutoff, so it separates out as a size-1 cluster
    # while the graphene+N network is one large no-metal adsorbate component.
    ads, clu = classify_appended(cocn_sheet)
    # exactly one ads group (the 158-atom C+N network) and one size-1 cluster (the Co)
    assert len(ads) == 1 and len(ads[0]) == 158
    assert len(clu) == 1 and len(clu[0]) == 1


def test_neighbor_graph_on_periodic_cu_slab(cu100_slab):
    # Real periodic slab: every Cu in the bottom layer should bond to at least
    # one neighbor in the same layer (4 in-plane Cu-Cu bonds at ~2.7 Å).
    g = build_neighbor_graph(cu100_slab)
    # bottom-layer atoms: z near 10.0
    z = cu100_slab.get_positions()[:, 2]
    bottom = [i for i, zi in enumerate(z) if abs(zi - z.min()) < 0.1]
    assert len(bottom) == 36
    # each bottom atom should have at least one in-plane neighbor
    assert all(g.degree(i) >= 1 for i in bottom)
