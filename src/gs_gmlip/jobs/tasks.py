"""Tasks: what a job actually does.

Each task reads its input structure from the store, runs, and writes every
intermediate and final structure back with provenance. Register new tasks
(SSW, GOFEE, minima hopping, Sella, ...) with ``@register_task``; the app
picks them up from ``TASKS`` without further wiring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS, FIRE, LBFGS

from gs_gmlip.evaluator.calculators import CalculatorEvaluator, get_calculator
from gs_gmlip.interface.core import Interface
from gs_gmlip.interface.operations import adsorbate_atoms
from gs_gmlip.jobs.spec import JobSpec
from gs_gmlip.search.ga.runner import GARunner
from gs_gmlip.search.gcmc.runner import GCMCRunner
from gs_gmlip.store import StructureStore

logger = logging.getLogger(__name__)

OPTIMIZERS = {"BFGS": BFGS, "LBFGS": LBFGS, "FIRE": FIRE}

ProgressCallback = Callable[["TaskContext", float, str], None]


@dataclass
class TaskContext:
    """Everything a task needs: its spec, the store, and a progress channel.

    Attributes:
        spec: The job specification.
        store: Project structure store.
        workdir: Job directory (scratch space for the task).
        on_progress: Optional callback ``(context, progress in [0, 1], message)``.
        params: Task parameters with defaults filled in.
        n_structures: Structures written so far.
        last_structure_id: Most recent row id written.
    """

    spec: JobSpec
    store: StructureStore
    workdir: Path
    on_progress: ProgressCallback | None = None
    params: dict[str, Any] = field(default_factory=dict)
    n_structures: int = 0
    last_structure_id: int | None = None

    def save(
        self, interface: Interface, stage: str, method: str, step: int | None = None
    ) -> int:
        """Write a structure produced by this job to the store."""
        self.last_structure_id = self.store.add(
            interface,
            stage=stage,
            method=method,
            calculator=self.spec.calculator,
            parent_id=self.spec.structure_id,
            job_id=self.spec.job_id,
            step=step,
            label=self.spec.label,
        )
        self.n_structures += 1
        return self.last_structure_id

    def report(self, progress: float, message: str) -> None:
        """Publish progress (fraction complete and a short message)."""
        if self.on_progress is not None:
            self.on_progress(self, progress, message)

    def load_input(self) -> Interface:
        """Load the job's input structure."""
        return self.store.get(self.spec.structure_id)

    def calculator(self):
        """Build the job's calculator."""
        return get_calculator(self.spec.calculator, **self.spec.calculator_params)


@dataclass(frozen=True)
class Task:
    """A registered task.

    Attributes:
        kind: Registry key, used as ``JobSpec.kind``.
        run: The task function.
        defaults: Default parameters (also drives the app's parameter form).
        description: One-line description shown in the app.
    """

    kind: str
    run: Callable[[TaskContext], None]
    defaults: dict[str, Any]
    description: str


TASKS: dict[str, Task] = {}


def register_task(kind: str, defaults: dict[str, Any], description: str = ""):
    """Decorator registering a task function under ``kind``."""

    def decorator(func: Callable[[TaskContext], None]):
        TASKS[kind] = Task(kind, func, dict(defaults), description)
        return func

    return decorator


def run_task(
    spec: JobSpec,
    store: StructureStore,
    workdir: Path,
    on_progress: ProgressCallback | None = None,
) -> TaskContext:
    """Run a job in the current process.

    Args:
        spec: The job specification.
        store: Project structure store.
        workdir: Scratch directory for the task.
        on_progress: Optional callback ``(context, progress, message)``.

    Returns:
        The finished context (counts and last structure id).

    Raises:
        KeyError: If ``spec.kind`` is not a registered task.
    """
    if spec.kind not in TASKS:
        raise KeyError(f"unknown task {spec.kind!r}; registered: {sorted(TASKS)}")
    task = TASKS[spec.kind]
    context = TaskContext(
        spec=spec,
        store=store,
        workdir=Path(workdir),
        on_progress=on_progress,
        params={**task.defaults, **spec.params},
    )
    task.run(context)
    return context


def _max_force(atoms) -> float:
    return float(np.linalg.norm(atoms.get_forces(), axis=1).max())


@register_task(
    "singlepoint", defaults={}, description="Energy and forces of the structure as is"
)
def singlepoint(context: TaskContext) -> None:
    """Evaluate energy and forces without moving atoms."""
    interface = context.load_input()
    interface.interface.calc = context.calculator()
    energy = interface.interface.get_potential_energy()
    context.save(interface, stage="singlepoint", method="singlepoint")
    context.report(1.0, f"E = {energy:.4f} eV")


@register_task(
    "relax",
    defaults={"optimizer": "BFGS", "fmax": 0.05, "steps": 200, "write_interval": 5},
    description="Local geometry optimization (fixed atoms stay fixed)",
)
def relax(context: TaskContext) -> None:
    """Relax the structure, storing every ``write_interval``-th optimizer step."""
    params = context.params
    fmax, steps = float(params["fmax"]), int(params["steps"])
    interval = max(1, int(params["write_interval"]))
    method = str(params["optimizer"])
    interface = context.load_input()
    atoms = interface.interface
    atoms.calc = context.calculator()
    optimizer = OPTIMIZERS[method](atoms, logfile=None)

    def observe() -> None:
        step = optimizer.nsteps
        if step > 0 and step % interval == 0:
            context.save(interface, stage="relax_step", method=method, step=step)
        context.report(
            min(step / max(steps, 1), 0.99),
            f"step {step}: E = {atoms.get_potential_energy():.4f} eV, fmax = {_max_force(atoms):.4f} eV/Å",
        )

    optimizer.attach(observe, interval=1)
    optimizer.run(fmax=fmax, steps=steps)
    interface.substrate = atoms[: len(interface.substrate)]
    context.save(interface, stage="relaxed", method=method, step=optimizer.nsteps)
    residual = _max_force(atoms)
    state = "converged" if residual <= fmax else "not converged"
    context.report(
        1.0, f"{state} after {optimizer.nsteps} steps, fmax = {residual:.4f} eV/Å"
    )


@register_task(
    "md",
    defaults={
        "temperature": 300.0,
        "timestep_fs": 1.0,
        "steps": 200,
        "friction_per_fs": 0.01,
        "write_interval": 10,
        "seed": 0,
    },
    description="NVT Langevin molecular dynamics",
)
def molecular_dynamics(context: TaskContext) -> None:
    """Run Langevin MD, storing every ``write_interval``-th frame."""
    params = context.params
    steps = int(params["steps"])
    interval = max(1, int(params["write_interval"]))
    temperature = float(params["temperature"])
    rng = np.random.default_rng(int(params["seed"]))
    interface = context.load_input()
    atoms = interface.interface
    atoms.calc = context.calculator()
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=rng)
    dynamics = Langevin(
        atoms,
        timestep=float(params["timestep_fs"]) * units.fs,
        temperature_K=temperature,
        friction=float(params["friction_per_fs"]) / units.fs,
        rng=rng,
        logfile=None,
    )

    def observe() -> None:
        step = dynamics.nsteps
        context.save(interface, stage="md", method="Langevin", step=step)
        context.report(
            min(step / max(steps, 1), 0.99),
            f"step {step}: E = {atoms.get_potential_energy():.4f} eV, T = {atoms.get_temperature():.0f} K",
        )

    dynamics.attach(observe, interval=interval)
    dynamics.run(steps)
    context.report(1.0, f"{steps} MD steps at {temperature:.0f} K")


class _StoringEvaluator(CalculatorEvaluator):
    """Evaluator that writes every evaluated structure to the store.

    Used by the search tasks, so the database grows with each relaxation the
    search performs (accepted or not) instead of only at the end.
    """

    def __init__(
        self, context: TaskContext, stage: str, method: str, total: int, **kwargs
    ):
        super().__init__(
            context.spec.calculator, context.spec.calculator_params, **kwargs
        )
        self._context = context
        self._stage = stage
        self._method = method
        self._total = max(total, 1)
        self._count = 0

    def evaluate(self, interface, relax=False, fmax=None, steps=None):
        interface = super().evaluate(interface, relax=relax, fmax=fmax, steps=steps)
        self._count += 1
        self._context.save(
            interface, stage=self._stage, method=self._method, step=self._count
        )
        self._context.report(
            min(self._count / self._total, 0.99),
            f"evaluation {self._count}: E = {interface.interface.get_potential_energy():.4f} eV",
        )
        return interface


@register_task(
    "ga",
    defaults={
        "blocks": {"CO": 2},
        "population_size": 10,
        "n_initial": 10,
        "n_steps": 20,
        "fmax": 0.1,
        "relax_steps": 100,
        "seed": 0,
    },
    description="Genetic algorithm at fixed adsorbate composition",
)
def genetic_algorithm(context: TaskContext) -> None:
    """Run a GA search; every relaxed candidate is stored."""
    params = context.params
    blocks: dict[str, int] = dict(params["blocks"])
    n_steps = int(params["n_steps"])
    evaluator = _StoringEvaluator(
        context,
        stage="ga",
        method="GA",
        total=int(params["n_initial"]) + n_steps,
        fmax=float(params["fmax"]),
        relax_steps=int(params["relax_steps"]),
    )
    runner = GARunner(
        evaluator=evaluator,
        substrate=context.load_input(),
        blocks=[adsorbate_atoms(name) for name in blocks],
        counts=[int(count) for count in blocks.values()],
        workdir=str(context.workdir),
        population_size=int(params["population_size"]),
        n_initial=int(params["n_initial"]),
        fmax=float(params["fmax"]),
        relax_steps=int(params["relax_steps"]),
        seed=int(params["seed"]),
    )
    runner.setup()
    for _ in range(n_steps):
        info = runner.step()
        logger.info("GA step: %s", info)
    best = runner.population.best_energy() if runner.population.size else float("nan")
    context.report(
        1.0, f"GA finished: {context.n_structures} candidates, best E = {best:.4f} eV"
    )


@register_task(
    "gcmc",
    defaults={
        "blocks": ["O"],
        "chemical_potentials": {"O": -5.0},
        "temperature": 300.0,
        "n_steps": 50,
        "max_displacement": 1.0,
        "fmax": 0.1,
        "relax_steps": 100,
        "seed": 0,
    },
    description="Grand-canonical Monte Carlo at fixed chemical potentials",
)
def grand_canonical_mc(context: TaskContext) -> None:
    """Run a GCMC search; every evaluated proposal is stored."""
    params = context.params
    blocks: list[str] = list(params["blocks"])
    n_steps = int(params["n_steps"])
    evaluator = _StoringEvaluator(
        context,
        stage="gcmc",
        method="GCMC",
        total=n_steps + 1,
        fmax=float(params["fmax"]),
        relax_steps=int(params["relax_steps"]),
    )
    runner = GCMCRunner(
        evaluator=evaluator,
        substrate=context.load_input(),
        blocks=[adsorbate_atoms(name) for name in blocks],
        block_names=blocks,
        chemical_potentials={
            k: float(v) for k, v in dict(params["chemical_potentials"]).items()
        },
        temperature=float(params["temperature"]),
        max_displacement=float(params["max_displacement"]),
        workdir=str(context.workdir),
        seed=int(params["seed"]),
    )
    runner.setup()
    for _ in range(n_steps):
        info = runner.step()
        logger.info("GCMC step: %s", info)
    context.report(
        1.0,
        f"GCMC finished: best E = {runner.best_energy:.4f} eV, "
        f"acceptance = {runner.ensemble.acceptance_rate:.2f}",
    )
