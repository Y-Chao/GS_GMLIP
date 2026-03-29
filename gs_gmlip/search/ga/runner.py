"""Genetic Algorithm runner / workflow orchestrator.

Orchestrates the full GA search workflow:
init population -> loop(select parents -> crossover/mutate -> evaluate
-> update population -> check convergence).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
from ase import Atoms

from gs_gmlip.data.database import CandidateDatabase, PrepareDatabase
from gs_gmlip.evaluate.base import BaseEvaluator
from gs_gmlip.search.base import BaseSearcher
from gs_gmlip.search.ga.comparators import InteratomicDistanceComparator
from gs_gmlip.search.ga.operators import (
    CutAndSplicePairing,
    MirrorMutation,
    OffspringCreator,
    OperationSelector,
    PermutationMutation,
    RattleMutation,
)
from gs_gmlip.search.ga.population import Population
from gs_gmlip.search.ga.startgenerator import StartGenerator
from gs_gmlip.structure.bonds import generate_blmin
from gs_gmlip.structure.composition import CompositionConstraint, MolecularBlock
from gs_gmlip.structure.region import BoxRegion

logger = logging.getLogger(__name__)


class GARunner(BaseSearcher):
    """Genetic Algorithm for global structure searching.

    Parameters
    ----------
    evaluator : BaseEvaluator
        Energy evaluator.
    slab : Atoms
        Fixed slab structure.
    blocks : list of MolecularBlock
        Building blocks for structure generation.
    counts : list of int
        Number of each block.
    workdir : str
        Working directory.
    population_size : int
        Population size.
    n_initial : int
        Number of initial random structures.
    region : BoxRegion, optional
        Region for atom placement.
    blmin : dict, optional
        Min interatomic distances. Auto-generated if None.
    comparator : object, optional
        Structure comparator for deduplication.
    operators : list of OffspringCreator, optional
        Custom genetic operators.
    operator_weights : list of float, optional
        Weights for operator selection.
    fmax : float
        Force convergence for relaxation.
    relax_steps : int
        Max relaxation steps.
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        evaluator: BaseEvaluator,
        slab: Atoms,
        blocks: list[MolecularBlock],
        counts: list[int],
        workdir: str = "ga_work",
        population_size: int = 20,
        n_initial: int = 20,
        region: BoxRegion | None = None,
        blmin: dict | None = None,
        comparator=None,
        operators: list[OffspringCreator] | None = None,
        operator_weights: list[float] | None = None,
        fmax: float = 0.05,
        relax_steps: int = 200,
        seed: int | None = None,
    ):
        super().__init__(evaluator, workdir, seed)
        self.slab = slab
        self.blocks = blocks
        self.counts = counts
        self.population_size = population_size
        self.n_initial = n_initial
        self.fmax = fmax
        self.relax_steps = relax_steps

        self.rng = np.random.default_rng(seed)
        self.db_file = os.path.join(workdir, "ga.db")

        # Setup region
        if region is not None:
            self.region = region
        else:
            # Auto region: box above slab
            z_max_slab = slab.positions[:, 2].max() if len(slab) > 0 else 0
            self.region = BoxRegion(
                cell=slab.get_cell(),
                z_min=z_max_slab + 1.5,
                z_max=z_max_slab + 8.0,
            )

        # Setup blmin
        if blmin is not None:
            self.blmin = blmin
        else:
            all_numbers = list(slab.get_atomic_numbers())
            for block in blocks:
                all_numbers.extend(block.atoms.get_atomic_numbers())
            self.blmin = generate_blmin(all_numbers, scale=0.7)

        # Setup comparator
        n_top = sum(c * b.n_atoms for b, c in zip(blocks, counts))
        self.n_top = n_top
        if comparator is not None:
            self.comparator = comparator
        else:
            self.comparator = InteratomicDistanceComparator(n_top=n_top)

        # Setup operators
        if operators is not None:
            self.operators = operators
            self.operator_weights = operator_weights or [1.0] * len(operators)
        else:
            self.operators = [
                CutAndSplicePairing(slab, n_top, self.blmin, rng=self.rng),
                RattleMutation(n_top, self.blmin, rng=self.rng),
                PermutationMutation(n_top, rng=self.rng),
                MirrorMutation(n_top, self.blmin, rng=self.rng),
            ]
            self.operator_weights = [4.0, 3.0, 2.0, 1.0]

        self.op_selector = OperationSelector(
            self.operator_weights, self.operators, rng=self.rng
        )

        self.population: Population | None = None
        self.db: CandidateDatabase | None = None

    def setup(self) -> None:
        """Initialize GA: create DB, generate initial population, evaluate."""
        os.makedirs(self.workdir, exist_ok=True)

        # Create database if it doesn't exist
        if not os.path.exists(self.db_file):
            prep = PrepareDatabase(
                self.db_file,
                slab=self.slab,
                population_size=self.population_size,
            )

            # Generate initial candidates
            gen = StartGenerator(
                self.slab,
                self.blocks,
                self.counts,
                self.blmin,
                self.region,
                rng=self.rng,
            )

            logger.info(f"Generating {self.n_initial} initial candidates...")
            for i in range(self.n_initial):
                candidate = gen.get_new_candidate()
                if candidate is not None:
                    # Evaluate
                    candidate = self.evaluator.evaluate(
                        candidate, relax=True, fmax=self.fmax, steps=self.relax_steps
                    )
                    energy = candidate.get_potential_energy()
                    candidate.info["key_value_pairs"] = {"raw_score": -energy}
                    candidate.info["data"] = {}
                    prep.add_relaxed_candidate(candidate)
                    self._results.append(candidate)
                    logger.info(f"Initial candidate {i + 1}: E = {energy:.4f} eV")

        # Connect to database
        self.db = CandidateDatabase(self.db_file)
        self.population = Population(
            self.db,
            self.population_size,
            comparator=self.comparator,
            rng=self.rng,
        )
        self._is_setup = True
        logger.info("GA setup complete.")

    def step(self) -> dict:
        """Perform one GA step: select, operate, evaluate, update."""
        op = self.op_selector.get_operator()

        if op.min_inputs >= 2:
            parents_tuple = self.population.get_two_candidates()
            if parents_tuple is None:
                return {"status": "no_parents"}
            parents = list(parents_tuple)
        else:
            parent = self.population.get_one_candidate()
            if parent is None:
                return {"status": "no_parents"}
            parents = [parent]

        # Generate offspring
        offspring = op.get_new_individual(parents)
        if offspring is None:
            return {"status": "operator_failed", "operator": op.descriptor}

        # Add to DB as unrelaxed
        self.db.add_unrelaxed_candidate(offspring, f"pairing:{op.descriptor}")

        # Evaluate and relax
        try:
            offspring = self.evaluator.evaluate(
                offspring, relax=True, fmax=self.fmax, steps=self.relax_steps
            )
            energy = offspring.get_potential_energy()
            offspring.info["key_value_pairs"]["raw_score"] = -energy

            # Add to DB as relaxed
            self.db.add_relaxed_step(offspring)
            self.population.update([offspring])
            self._results.append(offspring)

            return {
                "status": "success",
                "energy": energy,
                "operator": op.descriptor,
            }
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return {"status": "eval_failed", "error": str(e)}

    @classmethod
    def from_config(cls, config: dict, evaluator: BaseEvaluator, workdir: str):
        """Create GARunner from configuration dictionary.

        Config keys:
            slab_file: str - path to slab structure file
            blocks: list of dict - [{name: "H2O", count: 4}, ...]
            population_size: int
            n_initial: int
            fmax: float
            relax_steps: int
            seed: int
        """
        from ase.io import read

        slab = read(config["slab_file"])

        blocks = []
        counts = []
        for block_cfg in config.get("blocks", []):
            name = block_cfg["name"]
            count = block_cfg.get("count", 1)
            mu = block_cfg.get("chemical_potential", 0.0)

            if len(name) <= 2 and name[0].isupper():
                block = MolecularBlock.single_atom(name, chemical_potential=mu)
            else:
                block = MolecularBlock.from_ase(name, chemical_potential=mu)

            blocks.append(block)
            counts.append(count)

        return cls(
            evaluator=evaluator,
            slab=slab,
            blocks=blocks,
            counts=counts,
            workdir=workdir,
            population_size=config.get("population_size", 20),
            n_initial=config.get("n_initial", 20),
            fmax=config.get("fmax", 0.05),
            relax_steps=config.get("relax_steps", 200),
            seed=config.get("seed"),
        )
