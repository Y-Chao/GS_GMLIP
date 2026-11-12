"""Calculator registry: any ASE calculator (DFT, MLIP, classical) by name.

Search methods and tasks never import a backend directly; they ask the
registry for a calculator by name. Backends are imported lazily, so an
uninstalled MLIP package only matters when that calculator is requested.

Register a custom backend with::

    register_calculator("my_calc", lambda **params: MyCalculator(**params))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable

from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator

from gm2_surf.evaluator.base import BaseEvaluator
from gm2_surf.interface.core import Interface

CalculatorFactory = Callable[..., Calculator]


@dataclass(frozen=True)
class CalculatorEntry:
    """A registered calculator backend.

    Attributes:
        name: Registry key.
        factory: Callable building the ASE calculator from keyword parameters.
        requires: Importable module the backend needs, ``None`` if always available.
        description: One-line description shown in the app.
        defaults: Default keyword parameters, overridden by user parameters.
        choices: Allowed values per parameter (drives dropdowns in the app).
    """

    name: str
    factory: CalculatorFactory
    requires: str | None = None
    description: str = ""
    defaults: dict[str, Any] = field(default_factory=dict)
    choices: dict[str, list[str]] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        """Whether the backend's package can be imported."""
        return self.requires is None or find_spec(self.requires) is not None


_REGISTRY: dict[str, CalculatorEntry] = {}


def register_calculator(
    name: str,
    factory: CalculatorFactory,
    *,
    requires: str | None = None,
    description: str = "",
    defaults: dict[str, Any] | None = None,
    choices: dict[str, list[str]] | None = None,
) -> None:
    """Register (or replace) a calculator backend under ``name``."""
    _REGISTRY[name] = CalculatorEntry(
        name, factory, requires, description, dict(defaults or {}), dict(choices or {})
    )


def calculator_entries() -> dict[str, CalculatorEntry]:
    """All registered backends, keyed by name."""
    return dict(_REGISTRY)


def get_calculator(name: str, **params: Any) -> Calculator:
    """Build a calculator by registry name.

    Args:
        name: Registry key.
        **params: Backend parameters, merged over the entry's defaults.

    Raises:
        KeyError: If ``name`` is not registered.
        ImportError: If the backend's package is not installed.
    """
    if name not in _REGISTRY:
        raise KeyError(f"unknown calculator {name!r}; registered: {sorted(_REGISTRY)}")
    entry = _REGISTRY[name]
    if not entry.available:
        raise ImportError(f"calculator {name!r} needs the '{entry.requires}' package")
    return entry.factory(**{**entry.defaults, **params})


def level_of_theory(name: str, params: dict[str, Any] | None = None) -> str:
    """Label identifying the level of theory of energies from a calculator.

    Energies are only comparable (for surface, adsorption, or formation
    energies) between structures carrying the same label. The label is the
    registry name plus the parameters that change the potential energy
    surface: the model for foundation potentials, the functional for DFT.

    Args:
        name: Registered calculator name.
        params: Calculator parameters; registry defaults are filled in.
    """
    entry = _REGISTRY.get(name)
    merged = {**(entry.defaults if entry else {}), **(params or {})}
    parts = [name]
    if merged.get("model_path"):
        # A user-trained model is identified by its file or directory name and
        # the folder it sits in; the pretrained ``model`` name does not apply.
        path = Path(str(merged["model_path"])).expanduser()
        parts.append("custom:" + "/".join(path.parts[-2:]))
        merged.pop("model", None)
    for key in ("model", "xc", "encut", "kspacing"):
        if key in merged and merged[key] not in (None, ""):
            value = merged[key]
            parts.append(str(value) if key in ("model", "xc") else f"{key}{value}")
    return "/".join(parts)


class CalculatorEvaluator(BaseEvaluator):
    """Evaluator for any registered calculator.

    The calculator is built once and reused (loading an MLIP per structure is
    prohibitive). After each evaluation the results are frozen onto the
    structure as a ``SinglePointCalculator``, so structures kept in a
    population never trigger a recalculation through the shared calculator.

    Args:
        name: Registry key of the calculator.
        params: Backend parameters.
        fmax: Force convergence criterion for relaxation (eV/Å).
        relax_steps: Maximum number of optimizer steps.
    """

    def __init__(
        self,
        name: str,
        params: dict[str, Any] | None = None,
        fmax: float = 0.05,
        relax_steps: int = 200,
    ):
        super().__init__(fmax=fmax, relax_steps=relax_steps)
        self.name = name
        self.params = dict(params or {})
        self._calculator: Calculator | None = None

    def get_calculator(self) -> Calculator:
        """Return the shared calculator, building it on first use."""
        if self._calculator is None:
            self._calculator = get_calculator(self.name, **self.params)
        return self._calculator

    def evaluate(
        self,
        interface: Interface,
        relax: bool = False,
        fmax: float | None = None,
        steps: int | None = None,
    ) -> Interface:
        """Evaluate (optionally relax), then freeze energy and forces on the structure."""
        interface = super().evaluate(interface, relax=relax, fmax=fmax, steps=steps)
        atoms = interface.interface
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=float(atoms.get_potential_energy()),
            forces=atoms.get_forces(apply_constraint=False),
        )
        interface._reset_cache()
        return interface


# ---------------------------------------------------------------------------
# Built-in backends (imports are deferred to the factory call)
# ---------------------------------------------------------------------------


def _emt(**params: Any) -> Calculator:
    from ase.calculators.emt import EMT

    return EMT(**params)


# MatGL foundation potentials (PES models). r2SCAN TensorNet is the default;
# PBE-trained variants are listed for Materials Project compatibility only.
MATGL_DEFAULT_MODEL = "TensorNet-PES-MatPES-r2SCAN-2025.2"
MATGL_MODELS: tuple[str, ...] = (
    "TensorNet-PES-MatPES-r2SCAN-2025.2",
    "TensorNet-PES-MatPES-r2SCAN-2025.2-m",
    "TensorNet-PES-MatPES-PBE-2025.2",
    "TensorNet-PES-MatPES-PBE-2025.2-m",
    "QET-PES-MatPES-r2SCAN-2025.2",
    "QET-PES-MatPES-PBE-2025.2",
    "CHGNet-PES-MatPES-r2SCAN-2025.2.10",
    "CHGNet-PES-MatPES-r2SCAN-1M-2026.9",
    "CHGNet-PES-MatPES-PBE-2025.2.10",
    "CHGNet-PES-MatPES-PBE-1M-2026.9",
    "M3GNet-PES-MatPES-r2SCAN-2025.2",
    "M3GNet-PES-MatPES-PBE-2025.2",
)


def matgl_models() -> list[str]:
    """PES models offered for the ``matgl`` calculator.

    The installed MatGL's own list of pretrained PES models is used when it
    can be queried; otherwise the static ``MATGL_MODELS`` list.
    """
    try:
        import matgl

        found = [
            name for name in matgl.get_available_pretrained_models() if "-PES-" in name
        ]
    except Exception:  # noqa: BLE001 - not installed, or offline
        found = []
    models = found or list(MATGL_MODELS)
    if MATGL_DEFAULT_MODEL in models:
        models.remove(MATGL_DEFAULT_MODEL)
        models.insert(0, MATGL_DEFAULT_MODEL)
    return models


def resolve_model_path(model_path: str | Path) -> Path:
    """Absolute path of a user-trained model, checked to exist.

    Args:
        model_path: A model file (MACE ``.model``) or a model directory
            (MatGL), ``~`` allowed.

    Raises:
        FileNotFoundError: If nothing exists at the path.
    """
    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"no model at {path}")
    return path


def _matgl(
    model: str = MATGL_DEFAULT_MODEL, model_path: str = "", **params: Any
) -> Calculator:
    """MatGL calculator from a pretrained model name or a user-trained model directory."""
    import matgl
    from matgl.ext.ase import PESCalculator

    # ASE optimizers and barostats expect stress in eV/Å^3 (PESCalculator defaults to GPa).
    params.setdefault("stress_unit", "eV/A3")
    source = str(resolve_model_path(model_path)) if model_path else model
    return PESCalculator(potential=matgl.load_model(source), **params)


def _mace(model: str = "medium", model_path: str = "", **params: Any) -> Calculator:
    """MACE calculator from a foundation model name or a user-trained ``.model`` file."""
    if model_path:
        from mace.calculators import MACECalculator

        params.setdefault("device", "cpu")
        return MACECalculator(model_paths=str(resolve_model_path(model_path)), **params)
    from mace.calculators import mace_mp

    return mace_mp(model=model, **params)


def _vasp(**params: Any) -> Calculator:
    from ase.calculators.vasp import Vasp

    return Vasp(**params)


register_calculator(
    "emt",
    _emt,
    description="ASE effective-medium theory (Al, Cu, Ag, Au, Ni, Pd, Pt + H, C, N, O); for testing",
)
register_calculator(
    "matgl",
    _matgl,
    requires="matgl",
    description="MatGL potential: a pretrained model (default r2SCAN TensorNet) or, with model_path, "
    "the directory of a model you trained",
    defaults={"model": MATGL_DEFAULT_MODEL, "model_path": ""},
    choices={"model": list(MATGL_MODELS)},
)
register_calculator(
    "mace",
    _mace,
    requires="mace",
    description="MACE potential: a foundation model (mace_mp) or, with model_path, the .model file "
    "of a model you trained",
    defaults={"model": "medium", "model_path": "", "default_dtype": "float64"},
)
register_calculator(
    "vasp",
    _vasp,
    description="VASP through ASE (needs ASE_VASP_COMMAND and VASP_PP_PATH, which the cluster presets of the "
    "Hosts dialog set from the registered VASP module); set xc explicitly",
)
