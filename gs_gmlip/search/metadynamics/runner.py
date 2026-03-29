"""Metadynamics runner: orchestrates biased MD for structure search.

Combines PLUMED metadynamics with MLIP evaluation to explore
the free energy surface and identify stable/metastable structures.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from gs_gmlip.evaluate.base import BaseEvaluator
from gs_gmlip.search.base import BaseSearcher
from gs_gmlip.search.metadynamics.collective_vars import (
    CollectiveVariable,
    build_cvs_from_config,
)
from gs_gmlip.search.metadynamics.plumed_interface import (
    PlumedCalculator,
    extract_minima_from_fes,
    generate_plumed_input,
)

logger = logging.getLogger(__name__)


class MetadynamicsRunner(BaseSearcher):
    """Metadynamics-based structure search.

    Runs biased molecular dynamics using PLUMED to fill free energy
    basins and discover new minima along collective variables.

    Parameters
    ----------
    atoms : Atoms
        Initial structure.
    evaluator : BaseEvaluator
        MLIP or DFT calculator.
    cvs : list of CollectiveVariable
        Collective variables to bias.
    temperature : float
        MD temperature (K).
    height : float
        Gaussian hill height (eV).
    sigma : list of float, optional
        Gaussian widths per CV.
    pace : int
        Steps between hill depositions.
    biasfactor : float, optional
        For well-tempered metadynamics. None = standard metadynamics.
    timestep : float
        MD timestep (fs).
    friction : float
        Langevin friction coefficient (1/fs).
    seed : int, optional
        Random seed.
    work_dir : str
        Working directory.
    """

    def __init__(
        self,
        atoms: Atoms,
        evaluator: BaseEvaluator,
        cvs: list[CollectiveVariable],
        temperature: float = 300.0,
        height: float = 0.05,
        sigma: list[float] | None = None,
        pace: int = 500,
        biasfactor: float | None = 10.0,
        timestep: float = 1.0,
        friction: float = 0.01,
        seed: int | None = None,
        work_dir: str = "metad_run",
    ):
        self.initial_atoms = atoms.copy()
        self.evaluator = evaluator
        self.cvs = cvs
        self.temperature = temperature
        self.height = height
        self.sigma = sigma
        self.pace = pace
        self.biasfactor = biasfactor
        self.timestep = timestep
        self.friction = friction
        self.seed = seed
        self.work_dir = Path(work_dir)

        # State
        self.atoms: Atoms | None = None
        self.md = None
        self.step_count = 0
        self.trajectory: list[dict] = []
        self.snapshots: list[Atoms] = []
        self.rng = np.random.default_rng(seed)

    def setup(self) -> None:
        """Set up the metadynamics run."""
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.atoms = self.initial_atoms.copy()

        # Generate PLUMED input
        plumed_input = generate_plumed_input(
            cvs=self.cvs,
            height=self.height,
            sigma=self.sigma,
            pace=self.pace,
            temperature=self.temperature,
            biasfactor=self.biasfactor,
            file_prefix=str(self.work_dir / "metad"),
        )

        # Save PLUMED input for reference
        (self.work_dir / "plumed.dat").write_text(plumed_input)

        # Set up calculator
        base_calc = self.evaluator.get_calculator()
        plumed_calc = PlumedCalculator(
            base_calc=base_calc,
            plumed_input=plumed_input,
            timestep=self.timestep,
            work_dir=self.work_dir,
        )
        self.atoms.calc = plumed_calc.get_calculator(self.atoms)

        # Initialize velocities
        MaxwellBoltzmannDistribution(self.atoms, temperature_K=self.temperature)

        # Set up Langevin dynamics
        self.md = Langevin(
            self.atoms,
            timestep=self.timestep * units.fs,
            temperature_K=self.temperature,
            friction=self.friction / units.fs,
        )

        logger.info(
            "Metadynamics initialized: T=%.1f K, %d CVs, pace=%d",
            self.temperature,
            len(self.cvs),
            self.pace,
        )

    def step(self) -> dict:
        """Run one block of MD steps (= one pace interval).

        Each 'step' in the search context corresponds to one
        Gaussian deposition interval.
        """
        self.step_count += 1

        # Run MD for 'pace' steps
        self.md.run(self.pace)

        # Record snapshot
        snapshot = self.atoms.copy()
        self.snapshots.append(snapshot)

        energy = self.atoms.get_potential_energy()

        step_info = {
            "step": self.step_count,
            "md_steps": self.step_count * self.pace,
            "energy": energy,
            "temperature_inst": self.atoms.get_kinetic_energy()
            / (1.5 * units.kB * len(self.atoms)),
        }
        self.trajectory.append(step_info)

        if self.step_count % 10 == 0:
            logger.info(
                "Metad step %d (MD step %d): E=%.4f eV",
                self.step_count,
                self.step_count * self.pace,
                energy,
            )

        return step_info

    def get_results(self) -> list[Atoms]:
        """Return snapshots collected during the run."""
        return [s.copy() for s in self.snapshots]

    def get_best(self, n: int = 1) -> list[Atoms]:
        """Analyze HILLS to find minima and return corresponding snapshots.

        For a proper analysis, use the analysis module instead.
        """
        hills_file = self.work_dir / "metad_HILLS"
        minima = extract_minima_from_fes(hills_file, self.cvs, n_minima=n)

        if not minima:
            # Fallback: return lowest-energy snapshot
            if self.snapshots:
                energies = []
                for s in self.snapshots:
                    try:
                        e = s.get_potential_energy()
                    except Exception:
                        e = np.inf
                    energies.append(e)
                sorted_idx = np.argsort(energies)
                return [self.snapshots[i].copy() for i in sorted_idx[:n]]
            return []

        # Map FES minima back to nearest snapshots
        # (simplified: just return lowest energy snapshots)
        if self.snapshots:
            energies = []
            for s in self.snapshots:
                try:
                    e = s.get_potential_energy()
                except Exception:
                    e = np.inf
                energies.append(e)
            sorted_idx = np.argsort(energies)
            return [self.snapshots[i].copy() for i in sorted_idx[:n]]
        return []

    @classmethod
    def from_config(cls, config: dict) -> "MetadynamicsRunner":
        """Build MetadynamicsRunner from config dict."""
        from ase.io import read

        from gs_gmlip.cli import build_evaluator

        atoms = read(config["structure_file"])
        evaluator = build_evaluator(config.get("evaluator", {}))
        cvs = build_cvs_from_config(config.get("collective_variables", []))

        return cls(
            atoms=atoms,
            evaluator=evaluator,
            cvs=cvs,
            temperature=config.get("temperature", 300.0),
            height=config.get("height", 0.05),
            sigma=config.get("sigma"),
            pace=config.get("pace", 500),
            biasfactor=config.get("biasfactor", 10.0),
            timestep=config.get("timestep", 1.0),
            friction=config.get("friction", 0.01),
            seed=config.get("seed"),
            work_dir=config.get("work_dir", "metad_run"),
        )
