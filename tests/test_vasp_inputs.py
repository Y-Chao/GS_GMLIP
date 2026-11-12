"""Tests for VASP inputs built from pymatgen input sets."""

from __future__ import annotations

import pytest
from pymatgen.io.vasp.sets import MatPESStaticSet, MPMDSet, MPRelaxSet, MPScanRelaxSet

from gm2_surf.evaluator import vasp
from gm2_surf.interface import operations as ops
from gm2_surf.jobs import JobSpec, run_task
from gm2_surf.store import Project


@pytest.fixture(scope="module")
def bulk():
    return ops.make_bulk("Cu", "fcc", 3.61)


@pytest.fixture(scope="module")
def slab(bulk):
    slab = ops.build_slab(bulk, (1, 1, 1), layers=3, vacuum=15.0, fix_layers=1)
    return ops.place_adsorbate(slab, "CO", ops.find_adsorption_sites(slab)[0].coords)


def test_slab_detection(bulk, slab):
    assert vasp.vacuum_thickness(bulk) < 3.0
    assert not vasp.is_slab(bulk)
    assert vasp.vacuum_thickness(slab) == pytest.approx(15.0 - 2.0 - 1.15, abs=0.1)
    assert vasp.is_slab(slab)


@pytest.mark.parametrize(
    ("preset", "input_set", "tag", "value"),
    [
        ("static-r2SCAN", MatPESStaticSet, "METAGGA", "R2scan"),
        ("static-PBE", MatPESStaticSet, "NSW", 0),
        ("relax-r2SCAN", MPScanRelaxSet, "METAGGA", "R2scan"),
        ("relax-PBE", MPRelaxSet, "IBRION", 2),
        ("md-PBE", MPMDSet, "IBRION", 0),
    ],
)
def test_presets_use_pymatgen_sets(bulk, preset, input_set, tag, value):
    built = vasp.vasp_input_set(bulk, preset=preset)
    assert isinstance(built, input_set)
    assert built.incar[tag] == value
    assert "METAGGA" in built.incar or "PBE" in preset


def test_bulk_keeps_the_set_untouched(bulk):
    resolved = vasp.resolve_incar(bulk, preset="relax-PBE")
    reference = MPRelaxSet(bulk).incar
    assert not resolved["slab"] and not resolved["selective_dynamics"]
    assert resolved["incar"] == dict(reference)
    assert resolved["incar"]["ISIF"] == 3 and "LDIPOL" not in resolved["incar"]
    assert resolved["kpoints"] is not None


def test_tag_layering_order_on_a_slab(slab):
    base = vasp.resolve_incar(slab, preset="relax-r2SCAN")["incar"]
    assert (base["ISIF"], base["LDIPOL"], base["IDIPOL"]) == (2, True, 3)
    assert len(base["DIPOL"]) == 3 and 0.3 < base["DIPOL"][2] < 0.7
    assert (
        base["ENCUT"] == 680.0 and "IVDW" not in base
    )  # MPScanRelaxSet default cutoff

    layered = vasp.resolve_incar(
        slab,
        preset="relax-r2SCAN",
        precision="low",
        dispersion="D4",
        user_incar_settings={"ENCUT": 450, "ISIF": 0, "LVTOT": None},
    )["incar"]
    # precision < surface < dispersion < user settings
    assert layered["EDIFF"] == 1e-4 and layered["KSPACING"] == 0.4
    assert layered["ENCUT"] == 450 and layered["ISIF"] == 0
    assert layered["IVDW"] == 13 and layered["LDIPOL"] is True
    # D4 parameters matched to the functional (pymatgen capitalises string values; VASP ignores case)
    assert layered["DFTD4_XC"].lower() == "r2scan"
    pbe = vasp.resolve_incar(slab, preset="relax-PBE", dispersion="D4")["incar"]
    assert pbe["IVDW"] == 13 and "DFTD4_XC" not in pbe  # VASP derives PBE from GGA
    d3 = vasp.resolve_incar(slab, preset="relax-r2SCAN", dispersion="D3")["incar"]
    assert (d3["IVDW"], d3["SDFTD3_XC"].lower()) == (15, "r2scan")
    assert "LVTOT" not in layered

    forced_bulk = vasp.resolve_incar(slab, preset="relax-r2SCAN", surface=False)
    assert not forced_bulk["slab"] and forced_bulk["incar"]["ISIF"] == 3
    static = vasp.resolve_incar(slab, preset="static-r2SCAN")["incar"]
    assert static["LDIPOL"] is True and "ISIF" not in static


def test_md_set_kwargs(bulk):
    built = vasp.vasp_input_set(
        bulk,
        preset="md-PBE",
        set_kwargs={"start_temp": 500.0, "end_temp": 500.0, "nsteps": 50},
    )
    assert (built.incar["TEBEG"], built.incar["NSW"]) == (500.0, 50)


@pytest.mark.parametrize(
    "kwargs", [{"preset": "nope"}, {"precision": "nope"}, {"dispersion": "nope"}]
)
def test_unknown_choices_are_rejected(bulk, kwargs):
    with pytest.raises(ValueError, match="unknown"):
        vasp.vasp_input_set(bulk, **kwargs)


def test_write_inputs_with_selective_dynamics(slab, tmp_path):
    directory = vasp.write_vasp_inputs(
        slab, tmp_path / "calc", preset="static-r2SCAN", potcar_spec=True
    )
    assert sorted(path.name for path in directory.iterdir()) == [
        "INCAR",
        "POSCAR",
        "POTCAR.spec",
    ]
    poscar = (directory / "POSCAR").read_text()
    assert "Selective dynamics" in poscar
    assert poscar.count("F F F") == len(slab.fix)
    assert poscar.count("T T T") == len(slab.interface) - len(slab.fix)
    assert (directory / "POTCAR.spec").read_text().split() == ["Cu_pv", "C", "O"]
    assert "LDIPOL = True" in (directory / "INCAR").read_text()


def test_vasp_inputs_task_writes_into_job_directory(slab, tmp_path):
    project = Project(tmp_path / "proj")
    row_id = project.store.add(slab, stage="built")
    messages = []
    context = run_task(
        JobSpec(
            kind="vasp_inputs",
            structure_id=row_id,
            job_id="vasp-a",
            params={"preset": "relax-PBE", "user_incar_settings": {"NSW": 300}},
        ),
        project.store,
        tmp_path / "job",
        lambda context, progress, message: messages.append(message),
    )
    assert context.n_structures == 0
    assert "NSW = 300" in (tmp_path / "job" / "vasp" / "INCAR").read_text()
    assert (tmp_path / "job" / "vasp" / "KPOINTS").is_file()
    assert "VASP inputs in" in messages[-1]
