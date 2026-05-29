"""Tests for gs_gmlip.interface.analysis."""

from __future__ import annotations

from ase import Atoms

from gs_gmlip.interface.analysis import (
    build_neighbor_graph,
    connected_components,
    is_metal,
)
from gs_gmlip.interface.analysis import classify_appended


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
