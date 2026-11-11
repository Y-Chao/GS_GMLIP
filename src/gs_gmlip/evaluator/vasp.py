"""VASP inputs from pymatgen's predefined input sets, with editable tags.

Inputs always start from a pymatgen ``VaspInputSet`` chosen by task and
functional; nothing is hand-rolled. On top of the set, tags are layered in a
fixed, inspectable order (later wins):

1. the pymatgen set (task- and functional-specific),
2. a precision level (``PRECISION_TAGS``),
3. surface tags when the structure is a slab (``surface_tags``),
4. a dispersion correction (``DISPERSION_TAGS``),
5. explicit user INCAR settings.

``resolve_incar`` returns the result of this layering so it can be shown and
edited before anything is written.

Note on functional choice for surfaces: r2SCAN (the default here, matching
MatPES and the default foundation potential) and PBE can order adsorption
sites differently and neither includes dispersion; pick the preset to match
the reference data you compare against, and keep the dispersion setting
identical to the MLIP leg of the workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.sets import (
    MatPESStaticSet,
    MPMDSet,
    MPRelaxSet,
    MPScanRelaxSet,
    VaspInputSet,
)

from gs_gmlip.interface.core import Interface

# A structure counts as a slab when the largest empty gap along c exceeds this (Å).
SLAB_VACUUM_THRESHOLD = 6.0


@dataclass(frozen=True)
class VaspPreset:
    """A task- and functional-specific pymatgen input set.

    Attributes:
        name: Preset key, ``<task>-<functional>``.
        task: ``static``, ``relax``, or ``md``.
        functional: ``r2SCAN`` or ``PBE``.
        input_set: The pymatgen ``VaspInputSet`` class.
        kwargs: Keyword arguments for the set.
        description: One-line description.
    """

    name: str
    task: str
    functional: str
    input_set: type[VaspInputSet]
    kwargs: dict[str, Any] = field(default_factory=dict)
    description: str = ""


VASP_PRESETS: dict[str, VaspPreset] = {
    preset.name: preset
    for preset in (
        VaspPreset(
            "static-r2SCAN",
            "static",
            "r2SCAN",
            MatPESStaticSet,
            {"xc_functional": "R2SCAN"},
            "MatPES r2SCAN single point (matches MatPES-r2SCAN training data)",
        ),
        VaspPreset(
            "static-PBE",
            "static",
            "PBE",
            MatPESStaticSet,
            {"xc_functional": "PBE"},
            "MatPES PBE single point (matches MatPES-PBE training data)",
        ),
        VaspPreset(
            "relax-r2SCAN",
            "relax",
            "r2SCAN",
            MPScanRelaxSet,
            {},
            "Materials Project r2SCAN relaxation",
        ),
        VaspPreset(
            "relax-PBE",
            "relax",
            "PBE",
            MPRelaxSet,
            {},
            "Materials Project PBE/PBE+U relaxation",
        ),
        VaspPreset(
            "md-PBE",
            "md",
            "PBE",
            MPMDSet,
            {"start_temp": 300.0, "end_temp": 300.0, "nsteps": 1000},
            "Materials Project PBE NVT molecular dynamics",
        ),
    )
}

DEFAULT_PRESET = "static-r2SCAN"

# Precision levels: tags layered over the set. "normal" keeps the set as is.
PRECISION_TAGS: dict[str, dict[str, Any]] = {
    "low": {"ENCUT": 400, "EDIFF": 1e-4, "PREC": "Normal", "KSPACING": 0.4},
    "normal": {},
    "high": {"ENCUT": 680, "EDIFF": 1e-6, "PREC": "Accurate", "KSPACING": 0.15},
}

# Dispersion corrections. D4 (IVDW = 13) needs VASP >= 6.2 built with DFT-D4 and
# takes its damping parameters from the functional set in the INCAR.
DISPERSION_TAGS: dict[str, dict[str, Any]] = {"none": {}, "D4": {"IVDW": 13}}


def _to_structure(structure: Interface | Structure | Atoms) -> Structure:
    if isinstance(structure, Interface):
        return structure.to_pymatgen()
    if isinstance(structure, Atoms):
        return AseAtomsAdaptor.get_structure(structure)
    return structure


def vacuum_thickness(structure: Interface | Structure | Atoms) -> float:
    """Largest empty gap (Å) along the c axis, measured along the cell normal."""
    pmg = _to_structure(structure)
    fractional = np.sort(pmg.frac_coords[:, 2] % 1.0)
    gaps = np.diff(np.concatenate([fractional, [fractional[0] + 1.0]]))
    matrix = pmg.lattice.matrix
    normal = np.cross(matrix[0], matrix[1])
    height = abs(matrix[2] @ normal) / np.linalg.norm(normal)
    return float(gaps.max() * height)


def is_slab(structure: Interface | Structure | Atoms) -> bool:
    """Whether the structure has a vacuum layer along c."""
    return vacuum_thickness(structure) > SLAB_VACUUM_THRESHOLD


def surface_tags(structure: Interface | Structure | Atoms, task: str) -> dict[str, Any]:
    """Tags a slab calculation needs on top of a bulk-oriented input set.

    A dipole correction along c is switched on, and relaxations are restricted
    to ionic positions (``ISIF = 2``) so the vacuum does not collapse.

    Args:
        structure: The slab.
        task: ``static``, ``relax``, or ``md``.
    """
    pmg = _to_structure(structure)
    weights = np.array([site.specie.Z for site in pmg], dtype=float)
    center = (pmg.frac_coords * weights[:, None]).sum(axis=0) / weights.sum()
    tags: dict[str, Any] = {
        "LDIPOL": True,
        "IDIPOL": 3,
        "DIPOL": [round(float(x), 4) for x in center],
    }
    if task == "relax":
        tags["ISIF"] = 2
    return tags


def _layers(
    structure: Structure,
    preset: VaspPreset,
    precision: str,
    surface: bool | None,
    dispersion: str,
    user_incar_settings: dict[str, Any] | None,
) -> dict[str, Any]:
    if precision not in PRECISION_TAGS:
        raise ValueError(
            f"unknown precision {precision!r}; choose from {list(PRECISION_TAGS)}"
        )
    if dispersion not in DISPERSION_TAGS:
        raise ValueError(
            f"unknown dispersion {dispersion!r}; choose from {list(DISPERSION_TAGS)}"
        )
    settings: dict[str, Any] = dict(PRECISION_TAGS[precision])
    if is_slab(structure) if surface is None else surface:
        settings.update(surface_tags(structure, preset.task))
    settings.update(DISPERSION_TAGS[dispersion])
    settings.update(user_incar_settings or {})
    return settings


def vasp_input_set(
    structure: Interface | Structure | Atoms,
    preset: str = DEFAULT_PRESET,
    precision: str = "normal",
    surface: bool | None = None,
    dispersion: str = "none",
    user_incar_settings: dict[str, Any] | None = None,
    set_kwargs: dict[str, Any] | None = None,
) -> VaspInputSet:
    """Build the pymatgen input set for a structure.

    Frozen atoms of an ``Interface`` become selective dynamics in the POSCAR.

    Args:
        structure: Structure to compute.
        preset: Key of ``VASP_PRESETS``.
        precision: Key of ``PRECISION_TAGS``.
        surface: Apply ``surface_tags``; ``None`` detects a slab automatically.
        dispersion: Key of ``DISPERSION_TAGS``.
        user_incar_settings: INCAR tags that override everything else. A value
            of ``None`` removes a tag.
        set_kwargs: Extra keyword arguments for the pymatgen set (e.g.
            ``start_temp``, ``end_temp``, ``nsteps`` for ``md-PBE``, or
            ``user_kpoints_settings``).

    Raises:
        ValueError: For an unknown preset, precision, or dispersion.
    """
    if preset not in VASP_PRESETS:
        raise ValueError(
            f"unknown VASP preset {preset!r}; choose from {list(VASP_PRESETS)}"
        )
    chosen = VASP_PRESETS[preset]
    pmg = _to_structure(structure)
    settings = _layers(pmg, chosen, precision, surface, dispersion, user_incar_settings)
    return chosen.input_set(
        pmg, user_incar_settings=settings, **{**chosen.kwargs, **(set_kwargs or {})}
    )


def resolve_incar(
    structure: Interface | Structure | Atoms, **kwargs: Any
) -> dict[str, Any]:
    """The INCAR tags that ``vasp_input_set`` with the same arguments would write.

    Returns:
        Dict with ``incar`` (tag -> value), ``kpoints`` (explicit mesh, or
        ``None`` when ``KSPACING`` is used), ``potcar`` (POTCAR symbols),
        ``slab`` (whether surface tags were applied), and ``selective_dynamics``.
    """
    input_set = vasp_input_set(structure, **kwargs)
    surface = kwargs.get("surface")
    kpoints = input_set.kpoints
    return {
        "incar": {
            key: _jsonable(value)
            for key, value in input_set.incar.items()
            if value is not None
        },
        "kpoints": None if kpoints is None else [list(row) for row in kpoints.kpts],
        "potcar": list(input_set.potcar_symbols),
        "slab": is_slab(structure) if surface is None else bool(surface),
        "selective_dynamics": input_set.poscar.selective_dynamics is not None,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(item) for item in value]
    return value


def potcars_available() -> bool:
    """Whether pymatgen is configured with a POTCAR directory (``PMG_VASP_PSP_DIR``)."""
    from pymatgen.core import SETTINGS

    directory = SETTINGS.get("PMG_VASP_PSP_DIR")
    return bool(directory) and Path(directory).is_dir()


def write_vasp_inputs(
    structure: Interface | Structure | Atoms,
    directory: str | Path,
    potcar_spec: bool | None = None,
    **kwargs: Any,
) -> Path:
    """Write INCAR, KPOINTS (if any), POSCAR, and POTCAR into ``directory``.

    Args:
        structure: Structure to compute.
        directory: Output directory (created if missing).
        potcar_spec: Write ``POTCAR.spec`` (symbols only) instead of a POTCAR.
            ``None`` writes a real POTCAR when pymatgen has a POTCAR directory
            configured, and the spec file otherwise.
        **kwargs: Passed to ``vasp_input_set``.

    Returns:
        The output directory.
    """
    directory = Path(directory)
    input_set = vasp_input_set(structure, **kwargs)
    if potcar_spec is None:
        potcar_spec = not potcars_available()
    input_set.write_input(
        str(directory), potcar_spec=potcar_spec, make_dir_if_not_present=True
    )
    return directory
