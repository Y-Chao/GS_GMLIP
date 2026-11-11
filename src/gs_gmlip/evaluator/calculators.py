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
from typing import Any, Callable

from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator

from gs_gmlip.evaluator.base import BaseEvaluator
from gs_gmlip.interface.core import Interface

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
    """

    name: str
    factory: CalculatorFactory
    requires: str | None = None
    description: str = ""
    defaults: dict[str, Any] = field(default_factory=dict)

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
) -> None:
    """Register (or replace) a calculator backend under ``name``."""
    _REGISTRY[name] = CalculatorEntry(
        name, factory, requires, description, dict(defaults or {})
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


def _matgl(model: str, **params: Any) -> Calculator:
    import matgl
    from matgl.ext.ase import PESCalculator

    return PESCalculator(potential=matgl.load_model(model), **params)


def _mace(**params: Any) -> Calculator:
    from mace.calculators import mace_mp

    return mace_mp(**params)


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
    description="MatGL foundation potential (default TensorNet-MatPES-r2SCAN)",
    defaults={"model": "TensorNet-MatPES-r2SCAN-v2025.1-PES"},
)
register_calculator(
    "mace",
    _mace,
    requires="mace",
    description="MACE foundation potential (mace_mp)",
    defaults={"model": "medium", "default_dtype": "float64"},
)
register_calculator(
    "vasp",
    _vasp,
    description="VASP through ASE (needs ASE_VASP_COMMAND and VASP_PP_PATH); set xc explicitly",
)
