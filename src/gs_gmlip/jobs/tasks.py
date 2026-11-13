"""Tasks: what a job actually does.

Each task reads its input structure from the store, runs, and writes every
intermediate and final structure back with provenance. Register new tasks
(SSW, GOFEE, minima hopping, Sella, ...) with ``@register_task``; the app
picks them up from ``TASKS`` without further wiring.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from ase import units
from ase.filters import FrechetCellFilter
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import MTKNPT, IsotropicMTKNPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS, FIRE, LBFGS

from gs_gmlip.evaluator.calculators import (
    CalculatorEvaluator,
    get_calculator,
    level_of_theory,
)
from gs_gmlip.evaluator.vasp import DEFAULT_PRESET, write_vasp_inputs
from gs_gmlip.interface.core import Interface
from gs_gmlip.interface.molecules import MoleculeLibrary
from gs_gmlip.jobs.spec import JobSpec
from gs_gmlip.search.ga.runner import GARunner
from gs_gmlip.search.gcmc.runner import GCMCRunner
from gs_gmlip.search.asop import ASOPRunner
from gs_gmlip.search.minima_hopping import MinimaHoppingRunner
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
    _input_role: str | None = field(default=None, repr=False)

    def input_role(self) -> str:
        """Role label of the job's input structure (looked up once)."""
        if self._input_role is None:
            self._input_role = self.store.record(self.spec.structure_id).role
        return self._input_role

    def role_of(self, interface: Interface) -> str:
        """Role of a structure this job produced.

        Bulk and molecule inputs keep their role. A labelled surface input
        yields ``adsorbed`` structures as soon as adsorbates are present and
        ``surface`` otherwise, so search tasks label their candidates correctly.
        """
        base = self.input_role()
        if base in ("surface", "adsorbed"):
            return (
                "adsorbed"
                if (interface.adsList or interface.clusterList)
                else "surface"
            )
        return base

    def save(
        self,
        interface: Interface,
        stage: str,
        method: str,
        step: int | None = None,
        score: float | None = None,
    ) -> int:
        """Write a structure produced by this job to the store, labelled with role and level."""
        self.last_structure_id = self.store.add(
            interface,
            stage=stage,
            method=method,
            calculator=self.spec.calculator,
            parent_id=self.spec.structure_id,
            job_id=self.spec.job_id,
            step=step,
            label=self.spec.label,
            score=score,
            role=self.role_of(interface),
            level=level_of_theory(self.spec.calculator, self.spec.calculator_params),
        )
        self.n_structures += 1
        return self.last_structure_id

    def report(self, progress: float, message: str) -> None:
        """Publish progress (fraction complete and a short message)."""
        if self.on_progress is not None:
            self.on_progress(self, progress, message)

    def block(self, name: str):
        """Building block by name: a built-in adsorbate or a permanent library molecule."""
        return MoleculeLibrary(self.store.path.parent).atoms(name)

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
        category: What kind of calculation it is, one of ``TASK_CATEGORIES``.
        applies_to: Model types it makes sense for, from ``MODEL_TYPES``.
    """

    kind: str
    run: Callable[[TaskContext], None]
    defaults: dict[str, Any]
    description: str
    category: str = "other"
    applies_to: tuple[str, ...] = ("bulk", "surface", "molecule")


# Calculation categories, in the order the app lists them.
TASK_CATEGORIES: tuple[str, ...] = (
    "single point",
    "optimization",
    "molecular dynamics",
    "global search",
    "input files",
    "other",
)
# Model types a task can apply to (the kinds of window in the app).
MODEL_TYPES: tuple[str, ...] = ("bulk", "surface", "molecule")

TASKS: dict[str, Task] = {}


def register_task(
    kind: str,
    defaults: dict[str, Any],
    description: str = "",
    category: str = "other",
    applies_to: tuple[str, ...] = MODEL_TYPES,
):
    """Decorator registering a task function under ``kind``.

    Args:
        kind: Job kind.
        defaults: Default parameters.
        description: One-line description shown in the app.
        category: One of ``TASK_CATEGORIES``.
        applies_to: Model types the task is offered for, from ``MODEL_TYPES``.

    Raises:
        ValueError: For an unknown category or model type.
    """
    _check_task_labels(category, applies_to)

    def decorator(func: Callable[[TaskContext], None]):
        TASKS[kind] = Task(
            kind, func, dict(defaults), description, category, tuple(applies_to)
        )
        return func

    return decorator


def _check_task_labels(category: str, applies_to: tuple[str, ...]) -> None:
    if category not in TASK_CATEGORIES:
        raise ValueError(
            f"unknown task category {category!r}; choose from {TASK_CATEGORIES}"
        )
    unknown = set(applies_to) - set(MODEL_TYPES)
    if unknown:
        raise ValueError(
            f"unknown model types {sorted(unknown)}; choose from {MODEL_TYPES}"
        )


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
    "singlepoint",
    defaults={},
    description="Energy and forces of the structure as is",
    category="single point",
)
def singlepoint(context: TaskContext) -> None:
    """Evaluate energy and forces without moving atoms."""
    interface = context.load_input()
    interface.interface.calc = context.calculator()
    energy = interface.interface.get_potential_energy()
    context.save(interface, stage="singlepoint", method="singlepoint")
    context.report(1.0, f"E = {energy:.4f} eV")


@register_task(
    "vasp_inputs",
    defaults={
        "preset": DEFAULT_PRESET,
        "precision": "normal",
        "dispersion": "none",
        "user_incar_settings": {},
    },
    description="Write VASP inputs from a pymatgen input set into the job directory (no calculation)",
    category="input files",
)
def vasp_inputs(context: TaskContext) -> None:
    """Write INCAR/KPOINTS/POSCAR/POTCAR for the input structure; the calculator is not used."""
    params = context.params
    directory = write_vasp_inputs(
        context.load_input(),
        context.workdir / "vasp",
        preset=str(params["preset"]),
        precision=str(params["precision"]),
        dispersion=str(params["dispersion"]),
        user_incar_settings=dict(params["user_incar_settings"]),
    )
    written = ", ".join(sorted(path.name for path in directory.iterdir()))
    context.report(1.0, f"VASP inputs in {directory}: {written}")


@register_task(
    "relax",
    defaults={"optimizer": "BFGS", "fmax": 0.05, "steps": 200, "write_interval": 5},
    description="Local geometry optimization (fixed atoms stay fixed)",
    category="optimization",
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
    "cell_relax",
    defaults={
        "optimizer": "BFGS",
        "fmax": 0.02,
        "steps": 300,
        "write_interval": 5,
        "hydrostatic": False,
        "constant_volume": False,
        "pressure_GPa": 0.0,
    },
    description="Bulk: relax atomic positions and the cell (needs a calculator with stress)",
    category="optimization",
    applies_to=("bulk",),
)
def cell_relax(context: TaskContext) -> None:
    """Relax positions and lattice together, storing every ``write_interval``-th step.

    Meant for bulk cells: a slab's vacuum direction would collapse.
    """
    params = context.params
    fmax, steps = float(params["fmax"]), int(params["steps"])
    interval = max(1, int(params["write_interval"]))
    method = str(params["optimizer"])
    interface = context.load_input()
    atoms = interface.interface
    atoms.calc = context.calculator()
    cell_filter = FrechetCellFilter(
        atoms,
        hydrostatic_strain=bool(params["hydrostatic"]),
        constant_volume=bool(params["constant_volume"]),
        scalar_pressure=float(params["pressure_GPa"]) * units.GPa,
    )
    optimizer = OPTIMIZERS[method](cell_filter, logfile=None)

    def summary() -> str:
        a, b, c = atoms.cell.lengths()
        return (
            f"E = {atoms.get_potential_energy():.4f} eV, V = {atoms.get_volume():.3f} Å³, "
            f"a, b, c = {a:.4f}, {b:.4f}, {c:.4f} Å"
        )

    def observe() -> None:
        step = optimizer.nsteps
        if step > 0 and step % interval == 0:
            interface.substrate = atoms[: len(interface.substrate)]
            context.save(interface, stage="cell_relax_step", method=method, step=step)
        context.report(min(step / max(steps, 1), 0.99), f"step {step}: {summary()}")

    optimizer.attach(observe, interval=1)
    optimizer.run(fmax=fmax, steps=steps)
    interface.substrate = atoms[: len(interface.substrate)]
    context.save(interface, stage="cell_relaxed", method=method, step=optimizer.nsteps)
    context.report(1.0, f"cell relaxed in {optimizer.nsteps} steps: {summary()}")


@register_task(
    "nvt",
    defaults={
        "temperature": 300.0,
        "timestep_fs": 1.0,
        "steps": 200,
        "friction_per_fs": 0.01,
        "write_interval": 10,
        "seed": 0,
    },
    description="NVT molecular dynamics (Langevin thermostat); constraints are honoured",
    category="molecular dynamics",
)
def nvt(context: TaskContext) -> None:
    """Run Langevin NVT MD, storing every ``write_interval``-th frame."""
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
        context.save(interface, stage="md", method="NVT-Langevin", step=step)
        context.report(
            min(step / max(steps, 1), 0.99),
            f"step {step}: E = {atoms.get_potential_energy():.4f} eV, T = {atoms.get_temperature():.0f} K",
        )

    dynamics.attach(observe, interval=interval)
    dynamics.run(steps)
    context.report(1.0, f"{steps} NVT steps at {temperature:.0f} K")


@register_task(
    "npt",
    defaults={
        "temperature": 300.0,
        "pressure_GPa": 0.0,
        "timestep_fs": 1.0,
        "steps": 200,
        "tdamp_fs": 100.0,
        "pdamp_fs": 1000.0,
        "isotropic": True,
        "write_interval": 10,
        "seed": 0,
    },
    description="Bulk: NPT molecular dynamics (MTK Nose-Hoover chains; needs stress, no constraints)",
    category="molecular dynamics",
    applies_to=("bulk",),
)
def npt(context: TaskContext) -> None:
    """Run Martyna-Tobias-Klein NPT MD, storing every ``write_interval``-th frame.

    Meant for bulk cells. ``isotropic`` scales the cell uniformly; otherwise
    the full cell fluctuates.
    """
    params = context.params
    steps = int(params["steps"])
    interval = max(1, int(params["write_interval"]))
    temperature = float(params["temperature"])
    interface = context.load_input()
    atoms = interface.interface
    if atoms.constraints:
        raise ValueError(
            "NPT does not support constraints; release fixed atoms or use NVT"
        )
    atoms.calc = context.calculator()
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=temperature, rng=np.random.default_rng(int(params["seed"]))
    )
    isotropic = bool(params["isotropic"])
    dynamics = (IsotropicMTKNPT if isotropic else MTKNPT)(
        atoms,
        timestep=float(params["timestep_fs"]) * units.fs,
        temperature_K=temperature,
        pressure_au=float(params["pressure_GPa"]) * units.GPa,
        tdamp=float(params["tdamp_fs"]) * units.fs,
        pdamp=float(params["pdamp_fs"]) * units.fs,
        logfile=None,
    )
    method = "NPT-MTK-iso" if isotropic else "NPT-MTK"

    def observe() -> None:
        step = dynamics.nsteps
        interface.substrate = atoms[: len(interface.substrate)]
        context.save(interface, stage="md", method=method, step=step)
        context.report(
            min(step / max(steps, 1), 0.99),
            f"step {step}: E = {atoms.get_potential_energy():.4f} eV, "
            f"T = {atoms.get_temperature():.0f} K, V = {atoms.get_volume():.3f} Å³",
        )

    dynamics.attach(observe, interval=interval)
    dynamics.run(steps)
    context.report(
        1.0,
        f"{steps} NPT steps at {temperature:.0f} K, V = {atoms.get_volume():.3f} Å³",
    )


class _StoringEvaluator(CalculatorEvaluator):
    """Evaluator that writes every evaluated structure to the store.

    Used by the search tasks, so the database grows with each relaxation the
    search performs (accepted or not) instead of only at the end.

    Args:
        context: Task context to save through.
        stage: Stage written on each row.
        method: Method written on each row.
        total: Expected number of evaluations (for progress only).
        score: Optional objective ``interface -> eV`` stored with each row.
        **kwargs: Passed to ``CalculatorEvaluator`` (``fmax``, ``relax_steps``).
    """

    def __init__(
        self,
        context: TaskContext,
        stage: str,
        method: str,
        total: int,
        score: Callable[[Interface], float | None] | None = None,
        **kwargs,
    ):
        super().__init__(
            context.spec.calculator, context.spec.calculator_params, **kwargs
        )
        self._context = context
        self._stage = stage
        self._method = method
        self._total = max(total, 1)
        self._score = score
        self._count = 0
        # Set by tasks that measure progress in their own units instead of evaluations.
        self.progress: float | None = None

    def evaluate(self, interface, relax=False, fmax=None, steps=None):
        interface = super().evaluate(interface, relax=relax, fmax=fmax, steps=steps)
        self._count += 1
        energy = interface.interface.get_potential_energy()
        score = self._score(interface) if self._score is not None else None
        self._context.save(
            interface,
            stage=self._stage,
            method=self._method,
            step=self._count,
            score=score,
        )
        message = f"evaluation {self._count}: E = {energy:.4f} eV"
        if score is not None:
            message += f", score = {score:.4f} eV"
        progress = (
            self.progress
            if self.progress is not None
            else min(self._count / self._total, 0.99)
        )
        self._context.report(progress, message)
        return interface


def count_species(interface: Interface, blocks: dict[str, Any]) -> Counter:
    """Count adsorbate groups per block name by matching their composition.

    Args:
        interface: The structure to analyse.
        blocks: Block name -> ASE ``Atoms`` template.
    """
    keys = {
        "".join(sorted(atoms.get_chemical_symbols())): name
        for name, atoms in blocks.items()
    }
    counts: Counter = Counter()
    symbols = interface.interface.get_chemical_symbols()
    for group in interface.adsList:
        name = keys.get("".join(sorted(symbols[i] for i in group)))
        if name is not None:
            counts[name] += 1
    return counts


class GrandPotential:
    """Grand potential relative to the first structure scored.

    ``score = (E - E_ref) - sum_i mu_i (N_i - N_i,ref)`` in eV, where the
    reference is the starting surface of the search. Lower is more stable at
    the given chemical potentials; the reference itself scores 0.

    Args:
        blocks: Block name -> ASE ``Atoms`` template.
        chemical_potentials: Block name -> chemical potential in eV.
    """

    def __init__(self, blocks: dict[str, Any], chemical_potentials: dict[str, float]):
        self.blocks = blocks
        self.chemical_potentials = chemical_potentials
        self._reference: tuple[float, Counter] | None = None

    def __call__(self, interface: Interface) -> float:
        energy = float(interface.interface.get_potential_energy())
        counts = count_species(interface, self.blocks)
        if self._reference is None:
            self._reference = (energy, counts)
        reference_energy, reference_counts = self._reference
        exchange = sum(
            mu * (counts[name] - reference_counts[name])
            for name, mu in self.chemical_potentials.items()
        )
        return energy - reference_energy - exchange


SearcherFactory = Callable[[CalculatorEvaluator, Interface, dict[str, Any], Path], Any]


def register_searcher(
    kind: str,
    factory: SearcherFactory,
    defaults: dict[str, Any],
    description: str = "",
    score: Callable[[dict[str, Any]], Callable[[Interface], float | None]]
    | None = None,
    applies_to: tuple[str, ...] = ("surface",),
) -> None:
    """Register a structure-search algorithm as a job kind.

    This is the plugin entry point for new search methods. The searcher only
    has to subclass ``BaseSearcher`` and evaluate candidates through the
    evaluator it is given; every evaluated structure is then written to the
    store with provenance, progress is reported, and the job shows up in the
    app with a parameter form built from ``defaults``.

    Args:
        kind: Job kind (also the stage and method written on each row).
        factory: ``(evaluator, start_structure, params, workdir) -> BaseSearcher``.
        defaults: Default parameters. ``n_steps`` (search steps), ``fmax``
            and ``relax_steps`` (relaxation settings of the evaluator) are
            read by the wrapper when present.
        description: One-line description shown in the app.
        score: Optional ``params -> (interface -> eV)`` building the objective
            stored with each row and used for ranking (lower is better).
        applies_to: Model types the search is offered for (surfaces by default).
    """
    defaults = {"n_steps": 20, "fmax": 0.1, "relax_steps": 100, **defaults}

    def run(context: TaskContext) -> None:
        params = context.params
        n_steps = int(params["n_steps"])
        evaluator = _StoringEvaluator(
            context,
            stage=kind,
            method=kind,
            total=n_steps,
            score=score(params) if score is not None else None,
            fmax=float(params["fmax"]),
            relax_steps=int(params["relax_steps"]),
        )
        searcher = factory(evaluator, context.load_input(), params, context.workdir)
        searcher.setup()
        for _ in range(n_steps):
            info = searcher.step()
            logger.info("%s step: %s", kind, info)
        context.report(
            1.0, f"{kind} finished: {context.n_structures} structures evaluated"
        )

    _check_task_labels("global search", applies_to)
    TASKS[kind] = Task(
        kind, run, defaults, description, "global search", tuple(applies_to)
    )


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
    category="global search",
    applies_to=("surface",),
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
        blocks=[context.block(name) for name in blocks],
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
    category="global search",
    applies_to=("surface",),
)
def grand_canonical_mc(context: TaskContext) -> None:
    """Run a GCMC search; every evaluated proposal is stored with its grand potential."""
    params = context.params
    blocks: list[str] = list(params["blocks"])
    n_steps = int(params["n_steps"])
    chemical_potentials = {
        k: float(v) for k, v in dict(params["chemical_potentials"]).items()
    }
    templates = {name: context.block(name) for name in blocks}
    evaluator = _StoringEvaluator(
        context,
        stage="gcmc",
        method="GCMC",
        total=n_steps + 1,
        score=GrandPotential(templates, chemical_potentials),
        fmax=float(params["fmax"]),
        relax_steps=int(params["relax_steps"]),
    )
    runner = GCMCRunner(
        evaluator=evaluator,
        substrate=context.load_input(),
        blocks=[templates[name] for name in blocks],
        block_names=blocks,
        chemical_potentials=chemical_potentials,
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


register_searcher(
    "minima_hopping",
    lambda evaluator, start, params, workdir: MinimaHoppingRunner(
        evaluator,
        start,
        workdir=str(workdir),
        temperature=float(params["temperature"]),
        ediff=float(params["ediff"]),
        mdmin=int(params["mdmin"]),
        timestep=float(params["timestep_fs"]),
        max_md_steps=int(params["max_md_steps"]),
        seed=int(params["seed"]),
    ),
    defaults={
        "n_steps": 20,
        "fmax": 0.05,
        "relax_steps": 200,
        "temperature": 1000.0,
        "ediff": 0.5,
        "mdmin": 2,
        "timestep_fs": 1.0,
        "max_md_steps": 500,
        "seed": 0,
    },
    description="Minima hopping (Goedecker 2004): MD escapes and relaxations with adaptive temperature",
    applies_to=("surface", "molecule"),
)


@register_task(
    "asop",
    defaults={
        "chemical_potentials": {"O": -5.0},
        "min_area": 1,
        "max_area": 4,
        "max_coverage": 1.0,
        "max_cycles": 3,
        "max_compositions": 15,
        "search_steps": 5,
        "sigma": 0.1,
        "min_distance": 1.5,
        "max_distance": 3.0,
        "temperature": 1000.0,
        "max_md_steps": 300,
        "n_steps": 1000,
        "fmax": 0.05,
        "relax_steps": 200,
        "seed": 0,
    },
    description="ASOP (Chen, Shang, Liu 2022): optimal surface phase over periodicity and composition "
    "at fixed chemical potentials; start from the clean primitive surface cell",
    category="global search",
    applies_to=("surface",),
)
def asop(context: TaskContext) -> None:
    """Run ASOP; every relaxation is stored with its surface free energy as the score."""
    params = context.params
    runner: ASOPRunner | None = None

    def _score(interface: Interface) -> float:
        return runner.surface_free_energy(interface) if runner is not None else 0.0

    evaluator = _StoringEvaluator(
        context,
        stage="asop",
        method="ASOP",
        total=1,
        score=_score,
        fmax=float(params["fmax"]),
        relax_steps=int(params["relax_steps"]),
    )
    runner = ASOPRunner(
        evaluator,
        context.load_input(),
        {name: float(mu) for name, mu in dict(params["chemical_potentials"]).items()},
        workdir=str(context.workdir),
        min_area=int(params["min_area"]),
        max_area=int(params["max_area"]),
        max_coverage=float(params["max_coverage"]),
        max_cycles=int(params["max_cycles"]),
        max_compositions=int(params["max_compositions"]),
        search_steps=int(params["search_steps"]),
        sigma=float(params["sigma"]),
        min_distance=float(params["min_distance"]),
        max_distance=float(params["max_distance"]),
        hopping={
            "temperature": float(params["temperature"]),
            "max_md_steps": int(params["max_md_steps"]),
        },
        seed=int(params["seed"]),
    )
    runner.setup()
    cycles = min(int(params["n_steps"]), runner.n_cycles)
    # Progress is counted in cycles, not evaluations: their number is not known in advance.
    for done in range(cycles):
        evaluator.progress = done / cycles
        info = runner.step()
        logger.info("ASOP cycle: %s", info)
        (Path(context.workdir) / "asop_phases.json").write_text(
            json.dumps(runner.phase_table(), indent=2)
        )
        context.report(
            min((done + 1) / cycles, 0.99),
            f"cycle {done + 1}/{cycles}, {info.get('grid', '')}: best gamma = {info.get('best_gamma', float('nan')):.4f} eV/cell",
        )
    best = runner.best_phases(1)
    summary = (
        f"most stable: {runner.formula(best[0].counts)} in {runner.grids[best[0].grid].supercell.label}, "
        f"gamma = {best[0].gamma:.4f} eV/cell"
        if best
        else "no phase found"
    )
    context.report(1.0, f"ASOP finished: {len(runner.phases)} phases, {summary}")
