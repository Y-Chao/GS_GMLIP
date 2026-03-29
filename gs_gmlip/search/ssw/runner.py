"""SSW runner: orchestrates the stochastic surface walking search.

Combines SSWWalker with MinimaHopping into a complete workflow
for exploring the potential energy surface.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from ase import Atoms

from gs_gmlip.evaluate.base import BaseEvaluator
from gs_gmlip.search.base import BaseSearcher
from gs_gmlip.search.ssw.minima_hopping import MinimaHopping
from gs_gmlip.search.ssw.walker import SSWWalker

logger = logging.getLogger(__name__)


class SSWRunner(BaseSearcher):
    """Stochastic Surface Walking structure search.

    Parameters
    ----------
    atoms : Atoms
        Initial structure (slab + adsorbates).
    evaluator : BaseEvaluator
        Energy/force calculator.
    temperature : float
        Initial temperature (K) for Metropolis acceptance.
    bias_height : float
        Gaussian bias height (eV).
    bias_width : float
        Gaussian bias width (Å).
    walk_length : float
        Walk step length (Å).
    n_soft_modes : int
        Number of soft modes to extract from Hessian.
    fmax : float
        Force convergence for relaxation (eV/Å).
    beta_decrease : float
        Temperature decrease factor on acceptance.
    beta_increase : float
        Temperature increase factor on rejection.
    seed : int, optional
        Random seed.
    work_dir : str
        Working directory.
    """

    def __init__(
        self,
        atoms: Atoms,
        evaluator: BaseEvaluator,
        temperature: float = 300.0,
        bias_height: float = 0.3,
        bias_width: float = 0.5,
        walk_length: float = 2.0,
        n_soft_modes: int = 3,
        fmax: float = 0.05,
        beta_decrease: float = 0.95,
        beta_increase: float = 1.05,
        seed: int | None = None,
        work_dir: str = "ssw_run",
    ):
        self.initial_atoms = atoms.copy()
        self.evaluator = evaluator
        self.seed = seed
        self.work_dir = Path(work_dir)

        # Walker
        self.walker = SSWWalker(
            evaluator=evaluator,
            temperature=temperature,
            bias_height=bias_height,
            bias_width=bias_width,
            walk_length=walk_length,
            n_soft_modes=n_soft_modes,
            fmax=fmax,
        )

        # Minima hopping
        self.hopper = MinimaHopping(
            temperature=temperature,
            beta_decrease=beta_decrease,
            beta_increase=beta_increase,
        )

        # RNG
        self.rng = np.random.default_rng(seed)

        # State
        self.current_atoms: Atoms | None = None
        self.current_energy: float = np.inf
        self.minima: list[tuple[Atoms, float]] = []
        self.step_count = 0
        self.trajectory: list[dict] = []

    def setup(self) -> None:
        """Relax the initial structure and set up the run."""
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Relax initial structure
        self.current_atoms = self.evaluator.evaluate(self.initial_atoms, relax=True)
        self.current_energy = self.current_atoms.get_potential_energy()

        self.minima.append((self.current_atoms.copy(), self.current_energy))
        self.hopper.visited_energies.append(self.current_energy)

        logger.info("SSW initialized: E=%.4f eV", self.current_energy)

    def step(self) -> dict:
        """Perform one SSW step: walk + relax + accept/reject."""
        self.step_count += 1

        # Identify active atoms
        tags = self.current_atoms.get_tags()
        active_indices = [i for i, t in enumerate(tags) if t > 0]
        if not active_indices:
            # If no tagged adsorbates, use top-layer atoms
            z_vals = self.current_atoms.positions[:, 2]
            z_thresh = z_vals.max() - 3.0
            active_indices = [i for i, z in enumerate(z_vals) if z > z_thresh]

        # Walk
        new_atoms, new_energy, walk_info = self.walker.walk(
            self.current_atoms, active_indices, self.rng
        )

        if new_atoms is None:
            return {
                "step": self.step_count,
                "accepted": False,
                "reason": walk_info.get("reason", "walk_failed"),
                "energy": self.current_energy,
                "best_energy": self.minima[0][1] if self.minima else np.inf,
                "temperature": self.hopper.temperature,
                "n_minima": len(self.minima),
            }

        # Accept / reject
        accepted = self.hopper.accept(self.current_energy, new_energy, self.rng)

        if accepted:
            self.current_atoms = new_atoms
            self.current_energy = new_energy

            # Track as new minimum if distinct
            if self.hopper.is_new_minimum(new_energy):
                self.minima.append((new_atoms.copy(), new_energy))
                # Keep sorted by energy
                self.minima.sort(key=lambda x: x[1])
                logger.info(
                    "New minimum #%d at step %d: E=%.4f eV",
                    len(self.minima),
                    self.step_count,
                    new_energy,
                )

        step_info = {
            "step": self.step_count,
            "accepted": accepted,
            "energy": self.current_energy,
            "new_energy": new_energy,
            "best_energy": self.minima[0][1] if self.minima else np.inf,
            "temperature": self.hopper.temperature,
            "n_minima": len(self.minima),
            "acceptance_rate": self.hopper.acceptance_rate,
            **walk_info,
        }
        self.trajectory.append(step_info)

        if self.step_count % 50 == 0:
            logger.info(
                "Step %d: E=%.4f, best=%.4f, T=%.1f K, minima=%d, acc=%.3f",
                self.step_count,
                self.current_energy,
                self.minima[0][1],
                self.hopper.temperature,
                len(self.minima),
                self.hopper.acceptance_rate,
            )

        return step_info

    def get_results(self) -> list[Atoms]:
        """Return all discovered minima sorted by energy."""
        return [atoms.copy() for atoms, _ in self.minima]

    def get_best(self, n: int = 1) -> list[Atoms]:
        """Return the n lowest-energy minima."""
        return [atoms.copy() for atoms, _ in self.minima[:n]]

    @classmethod
    def from_config(cls, config: dict) -> "SSWRunner":
        """Build SSWRunner from a config dict."""
        from ase.io import read

        from gs_gmlip.cli import build_evaluator

        atoms = read(config["structure_file"])
        evaluator = build_evaluator(config.get("evaluator", {}))

        return cls(
            atoms=atoms,
            evaluator=evaluator,
            temperature=config.get("temperature", 300.0),
            bias_height=config.get("bias_height", 0.3),
            bias_width=config.get("bias_width", 0.5),
            walk_length=config.get("walk_length", 2.0),
            n_soft_modes=config.get("n_soft_modes", 3),
            fmax=config.get("fmax", 0.05),
            beta_decrease=config.get("beta_decrease", 0.95),
            beta_increase=config.get("beta_increase", 1.05),
            seed=config.get("seed"),
            work_dir=config.get("work_dir", "ssw_run"),
        )
