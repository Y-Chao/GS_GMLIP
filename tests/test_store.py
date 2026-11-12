"""Tests for the provenance-aware structure store."""

from __future__ import annotations

import numpy as np
import pytest
from ase.calculators.emt import EMT

from gm2_surf.interface import operations as ops
from gm2_surf.store import Project, StructureStore


@pytest.fixture
def slab():
    bulk = ops.make_bulk("Cu", "fcc", 3.61)
    slab = ops.build_slab(bulk, (1, 1, 1), layers=3, supercell=(1, 1), fix_layers=1)
    return ops.place_adsorbate(slab, "CO", ops.find_adsorption_sites(slab)[0].coords)


@pytest.fixture
def store(tmp_path):
    return StructureStore(tmp_path / "structures.db")


def test_empty_store(store):
    assert len(store) == 0
    assert store.records() == []


def test_roundtrip_preserves_partition_and_constraints(store, slab):
    row_id = store.add(slab, stage="built", label="42")
    loaded = store.get(row_id)
    assert np.allclose(loaded.interface.positions, slab.interface.positions)
    assert len(loaded.substrate) == len(slab.substrate)
    assert loaded.fix == slab.fix
    assert loaded.adsList == slab.adsList
    assert list(loaded.interface.constraints[0].index) == slab.fix
    record = store.record(row_id)
    assert record.stage == "built"
    assert record.label == "42"  # numeric-looking labels are allowed
    assert record.energy is None
    assert record.parent_id is None


def test_energy_and_forces_are_frozen(store, slab):
    slab.interface.calc = EMT()
    energy = slab.interface.get_potential_energy()
    row_id = store.add(slab, stage="singlepoint", calculator="emt")
    record = store.record(row_id)
    assert record.energy == pytest.approx(energy)
    assert record.calculator == "emt"
    assert record.fmax is not None
    loaded = store.get(row_id)
    assert loaded.interface.get_potential_energy() == pytest.approx(energy)


def test_filters_and_lineage(store, slab):
    root = store.add(slab, stage="built")
    step = store.add(slab, stage="relax_step", parent_id=root, job_id="relax-a", step=5)
    final = store.add(slab, stage="relaxed", parent_id=root, job_id="relax-a", step=9)
    child = store.add(slab, stage="md", parent_id=final, job_id="md-b", step=0)

    assert [r.id for r in store.records()] == [root, step, final, child]
    assert [r.id for r in store.records(job_id="relax-a")] == [step, final]
    assert [r.id for r in store.records(stage="md")] == [child]
    assert [r.id for r in store.records(parent_id=root)] == [step, final]
    assert [r.id for r in store.records(limit=2)] == [final, child]
    assert [r.id for r in store.lineage(child)] == [root, final, child]
    assert store.record(step).step == 5


def test_delete(store, slab):
    first = store.add(slab, stage="built")
    second = store.add(slab, stage="built")
    store.delete([first])
    assert [r.id for r in store.records()] == [second]
    with pytest.raises(KeyError):
        store.get(first)


def test_project_layout(tmp_path):
    project = Project(tmp_path / "proj")
    assert project.jobs_dir.is_dir()
    assert project.store.path == project.root / "structures.db"


def test_score_is_stored_and_optional(store, slab):
    plain = store.add(slab, stage="gcmc")
    scored = store.add(slab, stage="gcmc", score=-1.25)
    assert store.record(plain).score is None
    assert store.record(scored).score == pytest.approx(-1.25)


def test_extra_constraints_roundtrip(store, slab):
    n_atoms = len(slab.interface)
    constrained = ops.add_bond_constraint(slab, n_atoms - 2, n_atoms - 1)
    constrained = ops.add_com_constraint(constrained, [n_atoms - 2, n_atoms - 1])
    constrained = ops.add_wall(
        constrained, [n_atoms - 1], height=18.0, spring_constant=3.0
    )
    loaded = store.get(store.add(constrained, stage="built"))
    assert loaded.fix == slab.fix
    assert ops.describe_constraints(loaded) == {
        "bonds": [[n_atoms - 2, n_atoms - 1]],
        "com": [[n_atoms - 2, n_atoms - 1]],
        "walls": [{"index": n_atoms - 1, "height": 18.0}],
    }
