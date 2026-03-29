"""GOFEE runner: surrogate-assisted global optimization.

Workflow:
1. Generate initial population (random or seeded).
2. Evaluate initial structures with the true calculator.
3. Loop:
   a. Fit/update surrogate model.
   b. Generate candidate pool via rattling/crossover from known structures.
   c. Score candidates with acquisition function.
   d. Select top-k candidates for true evaluation.
   e. Relax selected candidates; add to training data.
   f. Update best structure.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from ase import Atoms

from gs_gmlip.evaluate.base import BaseEvaluator
from gs_gmlip.search.base import BaseSearcher
from gs_gmlip.search.gofee.acquisition import (
    expected_improvement,
    lower_confidence_bound,
)
from gs_gmlip.search.gofee.surrogate import SurrogateModel
from gs_gmlip.structure.bonds import generate_blmin
from gs_gmlip.structure.composition import MolecularBlock
from gs_gmlip.structure.region import BoxRegion, Region

logger = logging.getLogger(__name__)


class GOFEERunner(BaseSearcher):
    """Global Optimization with First-principles Energy Expressions.

    Parameters
    ----------
    slab : Atoms
        Slab template.
    blocks : list of MolecularBlock
        Molecular building blocks.
    evaluator : BaseEvaluator
        True energy calculator.
    region : Region, optional
        Active region for structure generation.
    population_size : int
        Number of structures in the active population.
    n_candidates : int
        Number of candidate structures generated per step.
    n_select : int
        Number of candidates selected for true evaluation per step.
    kappa : float
        LCB kappa parameter.
    acquisition : str
        Acquisition function: 'lcb' or 'ei'.
    rattle_std : float
        Standard deviation of Gaussian rattling (Å).
    kernel : str
        GPR kernel type.
    seed : int, optional
        Random seed.
    work_dir : str
        Working directory.
    """

    def __init__(
        self,
        slab: Atoms,
        blocks: list[MolecularBlock],
        evaluator: BaseEvaluator,
        region: Region | None = None,
        population_size: int = 10,
        n_candidates: int = 50,
        n_select: int = 3,
        kappa: float = 2.0,
        acquisition: str = "lcb",
        rattle_std: float = 0.5,
        kernel: str = "rbf",
        seed: int | None = None,
        work_dir: str = "gofee_run",
    ):
        self.slab = slab.copy()
        self.blocks = blocks
        self.evaluator = evaluator
        self.population_size = population_size
        self.n_candidates = n_candidates
        self.n_select = n_select
        self.kappa = kappa
        self.acquisition_name = acquisition
        self.rattle_std = rattle_std
        self.work_dir = Path(work_dir)

        # Region
        if region is not None:
            self.region = region
        else:
            z_max = slab.positions[:, 2].max()
            self.region = BoxRegion(
                cell=slab.cell,
                z_min=z_max + 1.0,
                z_max=z_max + 8.0,
                pbc=slab.pbc,
            )

        # Surrogate
        self.surrogate = SurrogateModel(kernel=kernel)

        # RNG
        self.rng = np.random.default_rng(seed)

        # State
        self.population: list[Atoms] = []
        self.energies: list[float] = []
        self.all_structures: list[Atoms] = []
        self.all_energies: list[float] = []
        self.best_atoms: Atoms | None = None
        self.best_energy: float = np.inf
        self.step_count = 0
        self.blmin = generate_blmin(
            [s for b in blocks for s in b.symbols]
            + list(set(slab.get_chemical_symbols()))
        )

    def setup(self) -> None:
        """Generate and evaluate initial population."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "GOFEE setup: generating %d initial structures", self.population_size
        )

        from gs_gmlip.search.ga.startgenerator import StartGenerator

        gen = StartGenerator(
            slab=self.slab,
            blocks=self.blocks,
            region=self.region,
            blmin=self.blmin,
        )

        for i in range(self.population_size):
            candidate = gen.generate(self.rng)
            if candidate is None:
                logger.warning("Failed to generate initial structure %d", i)
                continue

            relaxed = self.evaluator.evaluate(candidate, relax=True)
            energy = relaxed.get_potential_energy()

            self.population.append(relaxed)
            self.energies.append(energy)
            self.all_structures.append(relaxed.copy())
            self.all_energies.append(energy)

            if energy < self.best_energy:
                self.best_energy = energy
                self.best_atoms = relaxed.copy()

        # Initial surrogate fit
        self.surrogate.update(self.all_structures, self.all_energies)
        logger.info(
            "GOFEE initialized with %d structures, best E=%.4f eV",
            len(self.population),
            self.best_energy,
        )

    def step(self) -> dict:
        """One GOFEE iteration: generate candidates, score, evaluate top-k."""
        self.step_count += 1

        # Generate candidate structures by rattling population members
        candidates = self._generate_candidates()

        if not candidates:
            logger.warning("No valid candidates generated at step %d", self.step_count)
            return {
                "step": self.step_count,
                "n_evaluated": 0,
                "best_energy": self.best_energy,
            }

        # Score with surrogate
        means, stds = self.surrogate.predict(candidates)

        if self.acquisition_name == "ei":
            scores = expected_improvement(means, stds, self.best_energy)
            # Higher EI = better, so we want descending sort
            selected_idx = np.argsort(scores)[-self.n_select :]
        else:
            # LCB: lower = better
            scores = lower_confidence_bound(means, stds, kappa=self.kappa)
            selected_idx = np.argsort(scores)[: self.n_select]

        # Evaluate selected candidates with true calculator
        n_evaluated = 0
        new_structures = []
        new_energies = []

        for idx in selected_idx:
            candidate = candidates[idx]
            try:
                relaxed = self.evaluator.evaluate(candidate, relax=True)
                energy = relaxed.get_potential_energy()
                new_structures.append(relaxed)
                new_energies.append(energy)
                n_evaluated += 1

                if energy < self.best_energy:
                    self.best_energy = energy
                    self.best_atoms = relaxed.copy()
                    logger.info(
                        "New best at step %d: E=%.4f eV", self.step_count, energy
                    )

            except Exception as e:
                logger.warning("Evaluation failed: %s", e)

        # Update surrogate and population
        if new_structures:
            self.surrogate.update(new_structures, new_energies)
            self.all_structures.extend(new_structures)
            self.all_energies.extend(new_energies)

            # Keep population as top-N by energy
            combined = list(
                zip(self.population + new_structures, self.energies + new_energies)
            )
            combined.sort(key=lambda x: x[1])
            self.population = [s for s, _ in combined[: self.population_size]]
            self.energies = [e for _, e in combined[: self.population_size]]

        return {
            "step": self.step_count,
            "n_evaluated": n_evaluated,
            "best_energy": self.best_energy,
            "pop_best": self.energies[0] if self.energies else None,
            "pop_worst": self.energies[-1] if self.energies else None,
            "total_evaluated": len(self.all_energies),
        }

    def _generate_candidates(self) -> list[Atoms]:
        """Generate candidate pool by rattling population members."""
        candidates = []
        if not self.population:
            return candidates

        for _ in range(self.n_candidates):
            # Pick random parent
            parent = self.population[self.rng.integers(len(self.population))].copy()

            # Rattle active atoms
            tags = parent.get_tags()
            active_idx = [i for i, t in enumerate(tags) if t > 0]

            if not active_idx:
                continue

            displacements = self.rng.normal(0, self.rattle_std, (len(active_idx), 3))
            for j, idx in enumerate(active_idx):
                parent.positions[idx] += displacements[j]

            candidates.append(parent)

        return candidates

    def get_results(self) -> list[Atoms]:
        """Return all evaluated structures sorted by energy."""
        order = np.argsort(self.all_energies)
        return [self.all_structures[i].copy() for i in order]

    def get_best(self, n: int = 1) -> list[Atoms]:
        order = np.argsort(self.all_energies)[:n]
        return [self.all_structures[i].copy() for i in order]

    @classmethod
    def from_config(cls, config: dict) -> "GOFEERunner":
        """Build from YAML config dict."""
        from gs_gmlip.structure.composition import (
            co2_block,
            co_block,
            formic_acid_block,
            hydrogen_block,
            oh_block,
            oxygen_block,
            so2_block,
            sulfur_block,
            water_block,
        )

        block_registry = {
            "H2O": water_block,
            "CO2": co2_block,
            "CO": co_block,
            "OH": oh_block,
            "H": hydrogen_block,
            "O": oxygen_block,
            "S": sulfur_block,
            "SO2": so2_block,
            "HCOOH": formic_acid_block,
        }

        blocks = []
        for bconf in config.get("blocks", []):
            name = bconf["name"]
            mu = bconf.get("chemical_potential", 0.0)
            if name in block_registry:
                blocks.append(block_registry[name](chemical_potential=mu))
            else:
                raise ValueError(f"Unknown block: {name}")

        from ase.io import read

        slab = read(config["slab_file"])

        from gs_gmlip.cli import build_evaluator

        evaluator = build_evaluator(config.get("evaluator", {}))

        return cls(
            slab=slab,
            blocks=blocks,
            evaluator=evaluator,
            population_size=config.get("population_size", 10),
            n_candidates=config.get("n_candidates", 50),
            n_select=config.get("n_select", 3),
            kappa=config.get("kappa", 2.0),
            acquisition=config.get("acquisition", "lcb"),
            rattle_std=config.get("rattle_std", 0.5),
            kernel=config.get("kernel", "rbf"),
            seed=config.get("seed"),
            work_dir=config.get("work_dir", "gofee_run"),
        )
