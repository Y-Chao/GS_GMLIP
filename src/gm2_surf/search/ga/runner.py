"""Genetic Algorithm runner / workflow orchestrator.

Orchestrates the full GA search workflow:
init population → loop(select parents → crossover/mutate → evaluate
→ update population → check convergence).
"""

from __future__ import annotations

import logging
import os

import numpy as np
from ase import Atoms

from gm2_surf.evaluator.base import BaseEvaluator
from gm2_surf.interface.core import Interface
from gm2_surf.search.base import BaseSearcher
from gm2_surf.search.ga.operators import (
    CutAndSpliceCrossover,
    MirrorMutation,
    OffspringCreator,
    OperationSelector,
    PermutationMutation,
    RattleMutation,
    TwistMutation,
    _generate_blmin,
)
from gm2_surf.search.ga.population import Population
from gm2_surf.search.region import BoxRegion, Region

logger = logging.getLogger(__name__)


def _blocks_apart(
    interface: Interface, n_new: int, blmin: dict[tuple[int, int], float]
) -> bool:
    """Whether the last ``n_new`` atoms keep the ``blmin`` distances to the other appended atoms."""
    atoms = interface.interface
    numbers = atoms.get_atomic_numbers()
    new = range(len(atoms) - n_new, len(atoms))
    for i in new:
        for j in range(len(interface.substrate), len(atoms) - n_new):
            limit = blmin.get((int(numbers[i]), int(numbers[j])))
            if limit is not None and atoms.get_distance(i, j, mic=True) < limit:
                return False
    return True


class GARunner(BaseSearcher):
    """Genetic Algorithm for global structure searching on surfaces.

    Parameters
    ----------
    evaluator : BaseEvaluator
        Energy evaluator.
    substrate : Interface
        Template interface (bare slab).  The substrate part is preserved;
        appended atoms define the active search region.
    blocks : list[Atoms]
        Molecular / cluster building blocks available for the search.
    counts : list[int]
        Number of each block per individual (fixed composition).
    workdir : str
        Working directory.
    population_size : int
        Number of candidates in the population.
    n_initial : int
        Number of random initial structures to generate.
    region : Region, optional
        Region for random atom placement during initialization.
        Defaults to a box above the slab.
    blmin : dict, optional
        Minimum interatomic distances {(Z1, Z2): distance}.
        Auto-generated from covalent radii if not provided.
    operators : list[OffspringCreator], optional
        Custom genetic operators.  Reasonable defaults are used if omitted.
    operator_weights : list[float], optional
        Weights for operator selection (same length as *operators*).
    fmax : float
        Force convergence for relaxation.
    relax_steps : int
        Maximum relaxation steps.
    energy_tolerance : float
        Two candidates are duplicates only if their energies differ by less
        than this (eV) and their fingerprints match.
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        evaluator: BaseEvaluator,
        substrate: Interface,
        blocks: list[Atoms],
        counts: list[int],
        workdir: str = "ga_work",
        population_size: int = 20,
        n_initial: int = 20,
        region: Region | None = None,
        blmin: dict[tuple[int, int], float] | None = None,
        operators: list[OffspringCreator] | None = None,
        operator_weights: list[float] | None = None,
        fmax: float = 0.05,
        relax_steps: int = 200,
        energy_tolerance: float = 0.02,
        seed: int | None = None,
    ):
        super().__init__(evaluator, workdir, seed)
        self.energy_tolerance = float(energy_tolerance)
        self.substrate_template = substrate
        self.blocks = blocks
        self.counts = counts
        self.population_size = population_size
        self.n_initial = n_initial
        self.fmax = fmax
        self.relax_steps = relax_steps

        self.rng = np.random.default_rng(seed)

        # --- Region ---
        if region is not None:
            self.region = region
        else:
            slab = substrate.interface
            z_max = slab.positions[:, 2].max() if len(slab) > 0 else 0.0
            self.region = BoxRegion(
                cell=slab.get_cell(),
                z_min=z_max + 1.5,
                # Within reach of the surface: an atom placed beyond the range
                # of the potential feels no force and never lands.
                z_max=z_max + 4.0,
            )

        # --- blmin ---
        if blmin is not None:
            self.blmin = blmin
        else:
            all_numbers = list(substrate.interface.get_atomic_numbers())
            for block in blocks:
                all_numbers.extend(block.get_atomic_numbers())
            self.blmin = _generate_blmin(all_numbers, scale=0.7)

        # --- Operators ---
        if operators is not None:
            self.operators = operators
            self.operator_weights = operator_weights or [1.0] * len(operators)
        else:
            self.operators = [
                CutAndSpliceCrossover(blmin=self.blmin, rng=self.rng),
                RattleMutation(blmin=self.blmin, rng=self.rng),
                PermutationMutation(rng=self.rng),
                MirrorMutation(blmin=self.blmin, rng=self.rng),
                TwistMutation(blmin=self.blmin, rng=self.rng),
            ]
            self.operator_weights = [4.0, 3.0, 2.0, 1.0, 1.5]

        self.op_selector = OperationSelector(
            self.operator_weights, self.operators, rng=self.rng
        )

        # --- State ---
        self.population: Population | None = None
        self._n_appended = sum(c * len(b) for b, c in zip(blocks, counts))
        self._composition = sorted(
            int(number)
            for block, count in zip(blocks, counts)
            for number in list(block.get_atomic_numbers()) * count
        )

    # ------------------------------------------------------------------
    # BaseSearcher interface
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Initialize: generate random initial population, evaluate, populate."""
        os.makedirs(self.workdir, exist_ok=True)

        self.population = Population(
            population_size=self.population_size,
            comparator=self._looks_like,
            rng=self.rng,
        )

        logger.info("Generating %d initial candidates …", self.n_initial)
        for i in range(self.n_initial):
            iface = self._generate_random()
            if iface is None:
                logger.warning("Failed to generate candidate %d", i)
                continue
            try:
                iface = self.evaluator.evaluate(
                    iface, relax=True, fmax=self.fmax, steps=self.relax_steps
                )
                energy = iface.interface.get_potential_energy()
                self.population.add(iface)
                self._results.append(iface)
                logger.info("  init %d: E = %.4f eV", i + 1, energy)
            except Exception:
                logger.warning(
                    "Evaluation failed for initial candidate %d", i, exc_info=True
                )

        self._is_setup = True
        logger.info("GA setup complete. Population size = %d", self.population.size)

    def step(self) -> dict:
        """Perform one GA step: select operator → parents → offspring → evaluate → update."""
        if self.population is None or self.population.size == 0:
            return {"status": "empty_population"}

        op = self.op_selector.get_operator()

        # Get parents
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
        if not self._has_composition(offspring):
            return {"status": "wrong_composition", "operator": op.descriptor}

        # Evaluate
        try:
            offspring = self.evaluator.evaluate(
                offspring, relax=True, fmax=self.fmax, steps=self.relax_steps
            )
            energy = offspring.interface.get_potential_energy()
            accepted = self.population.add(offspring)
            if accepted:
                self._results.append(offspring)

            return {
                "status": "success",
                "energy": energy,
                "operator": op.descriptor,
                "accepted": accepted,
                "pop_best": self.population.best_energy(),
            }
        except Exception as e:
            logger.warning("Evaluation failed: %s", e)
            return {"status": "eval_failed", "error": str(e)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _generate_random(self, max_attempts: int = 100) -> Interface | None:
        """Generate one random individual by placing blocks in the region.

        A block is placed again if it comes closer to an already placed block
        than ``blmin`` allows; ``None`` is returned if no free spot is found.
        """
        iface = self.substrate_template.copy()
        for block, count in zip(self.blocks, self.counts):
            for _ in range(count):
                for _ in range(max_attempts):
                    pos = self.region.random_position(self.rng)
                    mol = block.copy()
                    # Random rotation for multi-atom blocks
                    if len(mol) > 1:
                        angle = self.rng.uniform(0, 360)
                        axis = self.rng.normal(0, 1, 3)
                        axis /= np.linalg.norm(axis)
                        mol.rotate(angle, axis)
                    mol.translate(pos - mol.get_center_of_mass())
                    trial = iface.copy()
                    trial.add_adsorbate(mol)
                    if _blocks_apart(trial, len(mol), self.blmin):
                        iface = trial
                        break
                else:
                    return None
        return iface

    def _has_composition(self, candidate: Interface) -> bool:
        """Whether the appended atoms of ``candidate`` are the requested blocks."""
        numbers = candidate.interface.get_atomic_numbers()[len(candidate.substrate) :]
        return sorted(int(number) for number in numbers) == self._composition

    def _looks_like(self, a: Interface, b: Interface) -> bool:
        """Deduplication: same energy within the tolerance and matching fingerprints.

        The fingerprint of a slab with a few adsorbates is dominated by the
        slab, so fingerprints alone make distinct candidates look identical.
        """
        energy_a = a.interface.get_potential_energy()
        energy_b = b.interface.get_potential_energy()
        if abs(energy_a - energy_b) > self.energy_tolerance:
            return False
        fp_a = a.fingerprint
        fp_b = b.fingerprint
        if len(fp_a) != len(fp_b):
            return False
        # Cosine similarity of sorted pair-product vectors
        if len(fp_a) == 0:
            return True  # both empty
        sim = np.dot(fp_a, fp_b) / (np.linalg.norm(fp_a) * np.linalg.norm(fp_b) + 1e-12)
        return bool(sim > 0.999)

    def get_population(self) -> list[Interface]:
        """Return the current population (sorted best-first)."""
        if self.population is None:
            return []
        return self.population.individuals
