"""Tests for gs_gmlip.interface.analysis."""

from __future__ import annotations

from ase import Atoms

from gs_gmlip.interface.analysis import (
    build_neighbor_graph,
    connected_components,
    is_metal,
)


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
