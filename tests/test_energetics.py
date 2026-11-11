"""Tests for role/level labels, the normalizing importer, and derived energies."""

from __future__ import annotations

import ase.db
import numpy as np
import pytest
from ase.build import bulk, fcc111, molecule
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms

from gs_gmlip import energetics
from gs_gmlip.cli import main as cli_main
from gs_gmlip.evaluator import level_of_theory
from gs_gmlip.importers import import_ase_db, infer_role, level_from_row, to_interface
from gs_gmlip.interface import operations as ops
from gs_gmlip.jobs import JobSpec, run_task
from gs_gmlip.store import Project, StructureStore, global_store


def evaluated(atoms):
    atoms.calc = EMT()
    atoms.get_potential_energy()
    return atoms


def boxed(name):
    atoms = molecule(name)
    atoms.center(vacuum=6.0)
    return evaluated(atoms)


# ---------------------------------------------------------------- labels


def test_level_of_theory_labels():
    assert level_of_theory("emt") == "emt"
    assert level_of_theory("matgl") == "matgl/TensorNet-PES-MatPES-r2SCAN-2025.2"
    assert level_of_theory("matgl", {"model": "X"}) == "matgl/X"
    assert level_of_theory("vasp", {"xc": "PBE", "encut": 520}) == "vasp/PBE/encut520"
    assert level_of_theory("not-registered", {"model": "m"}) == "not-registered/m"


def test_store_labels_filters_and_validation(tmp_path):
    store = StructureStore(tmp_path / "s.db")
    slab = ops.build_slab(ops.make_bulk("Cu", "fcc", 3.61), (1, 1, 1), layers=3)
    first = store.add(
        slab,
        stage="built",
        role="surface",
        level="emt",
        tags={"potential": 3.4, "note": "x"},
    )
    second = store.add(slab, stage="built", role="bulk", level="gpaw/PBE/pw600")
    third = store.add(slab, stage="built")
    record = store.record(first)
    assert (record.role, record.level, record.tags) == (
        "surface",
        "emt",
        {"potential": 3.4, "note": "x"},
    )
    assert store.record(third).role == "" and store.record(third).tags == {}
    assert [r.id for r in store.records(role="bulk")] == [second]
    assert [r.id for r in store.records(level="emt")] == [first]
    assert [r.id for r in store.records(role="surface", level="gpaw/PBE/pw600")] == []
    with pytest.raises(ValueError, match="unknown role"):
        store.add(slab, stage="built", role="nope")


def test_jobs_label_outputs_with_role_and_level(tmp_path):
    project = Project(tmp_path / "proj")
    slab = ops.build_slab(
        ops.make_bulk("Cu", "fcc", 3.61), (1, 1, 1), layers=3, fix_layers=1
    )
    surface_id = project.store.add(slab, stage="built", role="surface")
    unlabelled_id = project.store.add(slab, stage="built")
    run_task(
        JobSpec(kind="singlepoint", structure_id=surface_id, job_id="sp-s"),
        project.store,
        project.root,
    )
    run_task(
        JobSpec(kind="singlepoint", structure_id=unlabelled_id, job_id="sp-u"),
        project.store,
        project.root,
    )
    run_task(
        JobSpec(
            kind="gcmc",
            structure_id=surface_id,
            job_id="gc-s",
            params={"n_steps": 8, "relax_steps": 2},
        ),
        project.store,
        project.root,
    )
    assert [(r.role, r.level) for r in project.store.records(job_id="sp-s")] == [
        ("surface", "emt")
    ]
    assert [(r.role, r.level) for r in project.store.records(job_id="sp-u")] == [
        ("", "emt")
    ]
    searched = project.store.records(job_id="gc-s")
    # The bare starting surface stays a surface; candidates with adsorbates become "adsorbed".
    assert {(r.role, r.natoms > searched[0].natoms) for r in searched} <= {
        ("surface", False),
        ("adsorbed", True),
    }
    assert searched[0].role == "surface" and any(r.role == "adsorbed" for r in searched)


# ---------------------------------------------------------------- importer


def test_infer_role():
    assert infer_role(bulk("Cu", "fcc", a=3.6, cubic=True)) == "bulk"
    slab = fcc111("Cu", size=(2, 2, 3), a=3.6, vacuum=8.0)
    slab.pbc = True
    assert infer_role(slab) == "surface"
    with_co = slab + molecule("CO")
    with_co.positions[-2:] += [1.0, 1.0, slab.positions[:, 2].max() + 2.0]
    assert infer_role(with_co) == "adsorbed"
    water = molecule("H2O")
    water.center(vacuum=6.0)
    assert infer_role(water) == "molecule"
    assert infer_role(molecule("H2O")) == "molecule"  # no cell at all


def test_to_interface_puts_metals_first_and_keeps_energy_and_constraints():
    atoms = molecule("CO") + fcc111("Cu", size=(1, 1, 2), a=3.6, vacuum=8.0)
    atoms.set_constraint(FixAtoms(indices=[2]))  # the first Cu
    forces = np.arange(len(atoms) * 3, dtype=float).reshape(-1, 3)
    atoms.calc = SinglePointCalculator(atoms, energy=-1.5, forces=forces)
    interface = to_interface(atoms)
    assert interface.interface.get_chemical_symbols() == ["Cu", "Cu", "O", "C"]
    assert len(interface.substrate) == 2 and interface.adsList == [[2, 3]]
    assert interface.fix == [0]
    assert interface.interface.get_potential_energy() == -1.5
    assert np.allclose(
        interface.interface.get_forces(apply_constraint=False)[0], forces[2]
    )
    plain = to_interface(evaluated(bulk("Cu", "fcc", a=3.6, cubic=True)))
    assert len(plain.substrate) == 4 and plain.interface.calc is not None


@pytest.fixture
def source_db(tmp_path):
    """A small foreign database mimicking different ways of recording settings."""
    path = tmp_path / "campaign.db"
    with ase.db.connect(str(path)) as db:
        cu = bulk("Cu", "fcc", a=3.6, cubic=True)
        cu.calc = SinglePointCalculator(cu, energy=-14.7)
        db.write(
            cu,
            xc="PBE",
            encut=600,
            kpts="8-8-8",
            data={"calc_params": {"mode": {"name": "pw", "ecut": 600}}},
        )
        d3 = bulk("Cu", "fcc", a=3.58, cubic=True)
        db.write(d3, xc="PBE-D3", encut=600, kpts="8-8-8", sp_energy=-16.7)
        slab = fcc111("Cu", size=(2, 2, 3), a=3.6, vacuum=8.0)
        slab.pbc = True
        slab += molecule("CO")
        slab.positions[-2:] += [1.0, 1.0, 14.0]
        slab.calc = SinglePointCalculator(slab, energy=-30.0)
        db.write(slab, pot=3.4, q=0.26, reac_name="CO_to_CHO")
    return path


def test_level_from_row_normalizes_settings(source_db):
    with ase.db.connect(str(source_db)) as db:
        rows = list(db.select())
    level, settings = level_from_row(rows[0])
    assert level == "unknown/PBE/pw600"  # SinglePointCalculator rows carry no code name
    assert settings["basis"] == "pw600" and settings["kpoints"] == "k8-8-8"
    assert (
        level_from_row(rows[1])[0] == ""
    )  # no calculator: no level unless a code is given


def test_functional_comes_from_calculator_settings(tmp_path):
    path = tmp_path / "vasp.db"
    cu = bulk("Cu", "fcc", a=3.6, cubic=True)
    with ase.db.connect(str(path)) as db:
        for parameters, key in (
            (
                {"xc": "pbe", "gga": "PE", "encut": 450, "kpts": [8, 8, 8]},
                "pbc",
            ),  # mislabelled key
            (
                {"xc": "pbe", "ivdw": 11, "encut": 450, "kpts": [8, 8, 8]},
                "pbe",
            ),  # D3 only in IVDW
            (
                {"xc": "PBE", "encut": 450, "kpts": [1, 1, 1]},
                "PBE-D3",
            ),  # key adds the suffix
        ):
            cu.calc = SinglePointCalculator(cu, energy=-14.0)
            cu.calc.name = "vasp"
            cu.calc.parameters = parameters
            db.write(cu, xc=key)
    with ase.db.connect(str(path)) as db:
        results = [level_from_row(row) for row in db.select()]
    assert [level for level, _ in results] == [
        "vasp/PBE/pw450",
        "vasp/PBE-D3/pw450",
        "vasp/PBE-D3/pw450",
    ]
    assert (
        results[0][1]["xc_key"] == "pbc"
        and results[1][1]["xc_key"] == "pbe"
        and "xc_key" not in results[2][1]
    )
    # The k-point mesh is recorded but is not part of the level.
    assert [settings["kpoints"] for _, settings in results] == [
        "k8-8-8",
        "k8-8-8",
        "k1-1-1",
    ]


def test_canonical_functional_names():
    from gs_gmlip.importers import canonical_xc

    names = ("pbe", "PBE", "rpbe-d3", "beef-vdw", "r2scan", "hse06", "MGGA_X_R2SCAN")
    expected = ["PBE", "PBE", "RPBE-D3", "BEEF-vdW", "r2SCAN", "HSE06", "MGGA_X_R2SCAN"]
    assert [canonical_xc(name) for name in names] == expected
    assert canonical_xc("PBE-D3BJ") == "PBE-D3BJ" and canonical_xc("") == ""


def test_same_geometry_with_different_labels_is_kept(tmp_path):
    path = tmp_path / "pairs.db"
    cu = bulk("Cu", "fcc", a=3.6, cubic=True)
    cu.calc = SinglePointCalculator(cu, energy=-14.0)
    with ase.db.connect(str(path)) as db:
        db.write(cu, reac_id="pair_0", type="is")
        db.write(
            cu, reac_id="pair_1", type="is"
        )  # the same state reused by another reaction
        db.write(cu, reac_id="pair_1", type="is")  # a true copy
    store = StructureStore(tmp_path / "dest.db")
    assert import_ase_db(path, store) == {"imported": 2, "skipped": 1}
    assert [r.tags["reac_id"] for r in store.records()] == ["pair_0", "pair_1"]


def test_geometric_copies_are_skipped(source_db, tmp_path):
    copy = tmp_path / "copy.db"
    with ase.db.connect(str(source_db)) as source, ase.db.connect(str(copy)) as target:
        for row in source.select():
            target.write(
                row.toatoms(), key_value_pairs=row.key_value_pairs, data=row.data
            )  # new unique ids
    store = StructureStore(tmp_path / "dest.db")
    assert import_ase_db(source_db, store, energy_key="sp_energy") == {
        "imported": 3,
        "skipped": 0,
    }
    assert import_ase_db(copy, store, energy_key="sp_energy") == {
        "imported": 0,
        "skipped": 3,
    }


def test_import_normalizes_and_is_repeatable(source_db, tmp_path):
    store = StructureStore(tmp_path / "dest.db")
    counts = import_ase_db(
        source_db, store, energy_key="sp_energy", code="gpaw", label_key="reac_name"
    )
    assert counts == {"imported": 3, "skipped": 0}
    assert import_ase_db(source_db, store, energy_key="sp_energy", code="gpaw") == {
        "imported": 0,
        "skipped": 3,
    }
    plain, dispersion, adsorbed = store.records()
    assert (plain.role, plain.level, plain.energy) == (
        "bulk",
        "gpaw/PBE/pw600",
        -14.7,
    )
    assert (dispersion.role, dispersion.level, dispersion.energy) == (
        "bulk",
        "gpaw/PBE-D3/pw600",
        -16.7,
    )
    assert (
        adsorbed.role == "adsorbed"
        and adsorbed.label == "CO_to_CHO"
        and adsorbed.stage == "imported"
    )
    assert adsorbed.tags["potential"] == 3.4 and adsorbed.tags["net_charge"] == 0.26
    assert adsorbed.tags["source"] == "campaign.db" and plain.tags["cutoff"] == 600
    assert (
        plain.tags["xc"] == "PBE" and dispersion.tags["xc"] == "PBE-D3"
    )  # normalized, not the raw key
    loaded = store.get(adsorbed.id)
    assert len(loaded.substrate) == 12 and loaded.adsList == [[12, 13]]
    with pytest.raises(ValueError, match="unknown role"):
        import_ase_db(source_db, store, role="nope")
    with pytest.raises(FileNotFoundError):
        import_ase_db(tmp_path / "missing.db", store)


def test_import_command_writes_to_the_global_store(source_db, capsys):
    assert cli_main(["import", str(source_db), "--role", "bulk"]) == 0
    assert "imported 3" in capsys.readouterr().out
    assert [r.role for r in global_store().records()] == ["bulk"] * 3


def test_import_command_reports_a_missing_file_without_a_traceback(capsys):
    assert cli_main(["import", "path/to/file.db"]) == 1
    assert "no such database: path/to/file.db" in capsys.readouterr().err
    assert global_store().records() == []


# ---------------------------------------------------------------- formulas


def test_energy_formulas():
    assert energetics.formation_energy(
        -10.0, {"Cu": 2, "O": 1}, {"Cu": -3.0, "O": -2.0}
    ) == pytest.approx(-2.0)
    with pytest.raises(KeyError, match="no chemical potential"):
        energetics.formation_energy(-10.0, {"Cu": 2, "O": 1}, {"Cu": -3.0})
    assert energetics.surface_energy(
        -39.0, {"Cu": 10}, {"Cu": -4.0}, area=5.0
    ) == pytest.approx(0.1)
    assert energetics.surface_energy(
        -39.0, {"Cu": 10}, {"Cu": -4.0}, area=5.0, n_surfaces=1
    ) == pytest.approx(0.2)
    assert energetics.adsorption_energy(-50.0, -40.0, -9.2) == pytest.approx(-0.8)
    assert energetics.grand_potential(-50.0, {"O": 2}, {"O": -5.0}) == pytest.approx(
        -40.0
    )
    assert energetics.grand_potential(
        -50.0, {}, {}, charge=0.5, electron_potential=4.4
    ) == pytest.approx(-47.8)
    potentials = energetics.che_chemical_potentials(e_h2=-6.8, e_h2o=-14.2, e_co=-14.8)
    assert potentials == pytest.approx({"H": -3.4, "O": -7.4, "C": -7.4})
    assert "C" not in energetics.che_chemical_potentials(-6.8, -14.2)
    assert energetics.EV_PER_A2_TO_J_PER_M2 == pytest.approx(16.0218, abs=1e-4)


# ---------------------------------------------------------------- analysis with stores


@pytest.fixture
def stores(tmp_path):
    reference = StructureStore(tmp_path / "global.db")
    for name in ("H2", "H2O", "CO"):
        reference.add(
            to_interface(boxed(name)), stage="imported", role="molecule", level="emt"
        )
    reference.add(
        to_interface(evaluated(bulk("Cu", "fcc", a=3.59, cubic=True))),
        stage="imported",
        role="bulk",
        level="emt",
    )
    # A higher-energy bulk and a bulk at another level must not be picked.
    reference.add(
        to_interface(evaluated(bulk("Cu", "fcc", a=3.80, cubic=True))),
        stage="imported",
        role="bulk",
        level="emt",
    )
    other = bulk("Cu", "fcc", a=3.59, cubic=True)
    other.calc = SinglePointCalculator(other, energy=-999.0)
    reference.add(to_interface(other), stage="imported", role="bulk", level="gpaw/PBE")
    return [("project", StructureStore(tmp_path / "project.db")), ("global", reference)]


def relaxed_slab():
    slab = ops.build_slab(
        ops.make_bulk("Cu", "fcc", 3.59), (1, 1, 1), layers=4, supercell=(2, 2)
    )
    evaluated(slab.interface)
    return slab


def test_surface_energy_from_matching_bulk(stores):
    slab = relaxed_slab()
    result = energetics.analyze(slab, "surface", "emt", stores)
    atoms = slab.interface
    bulk_per_atom = (
        evaluated(bulk("Cu", "fcc", a=3.59, cubic=True)).get_potential_energy() / 4
    )
    area = np.linalg.norm(np.cross(atoms.cell[0], atoms.cell[1]))
    expected = (atoms.get_potential_energy() - len(atoms) * bulk_per_atom) / (2 * area)
    assert result.values["surface energy (eV/Å²)"] == pytest.approx(expected)
    assert result.values["surface energy (J/m²)"] == pytest.approx(
        expected * 16.02176634
    )
    assert (
        0.5 < result.values["surface energy (J/m²)"] < 2.0
    )  # physically sensible for Cu(111)
    assert result.references == {"bulk Cu": "global #4 Cu4"}
    assert result.missing == [] and result.energy_per_atom == pytest.approx(
        atoms.get_potential_energy() / len(atoms)
    )


def test_adsorption_energy_needs_clean_surface_and_gas_references(stores):
    slab = relaxed_slab()
    site = ops.find_adsorption_sites(slab)[0]
    with_co = ops.place_adsorbate(slab, "CO", site.coords)
    evaluated(with_co.interface)

    missing = energetics.analyze(with_co, "adsorbed", "emt", stores)
    assert missing.values == {} and "no clean-surface reference" in missing.missing[0]

    stores[0][1].add(slab, stage="built", role="surface", level="emt")
    result = energetics.analyze(with_co, "adsorbed", "emt", stores)
    e_co = boxed("CO").get_potential_energy()
    expected = (
        with_co.interface.get_potential_energy()
        - slab.interface.get_potential_energy()
        - e_co
    )
    assert result.values["adsorption energy vs gas molecules (eV)"] == pytest.approx(
        expected
    )
    # For CO the CHE reference (mu_C + mu_O) is E(CO) itself.
    assert result.values[
        "adsorbate formation energy vs H2, H2O, CO (eV)"
    ] == pytest.approx(expected)
    assert result.references["clean surface"] == "project #1 Cu64"
    assert set(result.references) == {"clean surface", "gas H2", "gas H2O", "gas CO"}

    # OH has no gas-phase entry: only the CHE-referenced formation energy is available.
    with_oh = ops.place_adsorbate(slab, "OH", site.coords)
    evaluated(with_oh.interface)
    result = energetics.analyze(with_oh, "adsorbed", "emt", stores)
    assert list(result.values) == ["adsorbate formation energy vs H2, H2O, CO (eV)"]
    mu = energetics.che_chemical_potentials(
        boxed("H2").get_potential_energy(), boxed("H2O").get_potential_energy()
    )
    expected = (
        with_oh.interface.get_potential_energy()
        - slab.interface.get_potential_energy()
        - mu["O"]
        - mu["H"]
    )
    assert result.values[
        "adsorbate formation energy vs H2, H2O, CO (eV)"
    ] == pytest.approx(expected)


def test_analysis_reports_what_is_missing(stores):
    slab = relaxed_slab()
    assert (
        "no bulk reference for Cu at level other"
        in energetics.analyze(slab, "surface", "other", stores).missing[0]
    )
    assert (
        "no level-of-theory label"
        in energetics.analyze(slab, "surface", "", stores).missing[0]
    )
    assert "no role label" in energetics.analyze(slab, "", "emt", stores).missing[0]
    bare = ops.build_slab(ops.make_bulk("Cu", "fcc", 3.59), (1, 1, 1), layers=3)
    assert (
        "has no energy" in energetics.analyze(bare, "surface", "emt", stores).missing[0]
    )
    assert energetics.analyze(bare, "surface", "emt", stores).as_dict()["values"] == {}
