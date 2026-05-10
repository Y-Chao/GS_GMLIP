"""GCMC runner: orchestrates the grand canonical MC search.

Combines Moves, GCEnsemble, and BaseEvaluator into a complete
structure search workflow with checkpointing and trajectory logging.

v1.1: Hookean bond constraints for adsorbate molecules and
      ASE database recording of visited configurations.
"""

from __future__ import annotations

import logging
from pathlib import Path

import ase.db
import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms

from gs_gmlip.evaluate.base import BaseEvaluator
from gs_gmlip.search.base import BaseSearcher
from gs_gmlip.search.gcmc.ensemble import GCEnsemble
from gs_gmlip.search.gcmc.moves import (
    DeleteMove,
    DisplaceMove,
    InsertMove,
    MCMove,
    SwapMove,
)
from gs_gmlip.structure.composition import MolecularBlock
from gs_gmlip.structure.constraints import BlockBondInfo, get_adsorbate_bond_constraints
from gs_gmlip.structure.region import BoxRegion, Region

logger = logging.getLogger(__name__)


class GCMCRunner(BaseSearcher):
    """Grand Canonical Monte Carlo structure search.

    Parameters
    ----------
    slab : Atoms
        The slab / surface template.
    blocks : list of MolecularBlock
        Species available for insertion.
    evaluator : BaseEvaluator
        Calculator for energies.
    temperature : float
        Temperature in Kelvin.
    chemical_potentials : dict[str, float]
        Species name -> chemical potential (eV).
    region : Region, optional
        Region for insertion. Defaults to a box above the slab.
    move_weights : dict[str, float], optional
        Weights for each move type. Default: equal weights.
    max_displacement : float
        Maximum displacement in Å for displace moves.
    seed : int, optional
        Random seed.
    work_dir : str
        Working directory for outputs.
    """

    def __init__(
        self,
        slab: Atoms,
        blocks: list[MolecularBlock],
        evaluator: BaseEvaluator,
        temperature: float = 300.0,
        chemical_potentials: dict[str, float] | None = None,
        region: Region | None = None,
        move_weights: dict[str, float] | None = None,
        max_displacement: float = 1.0,
        seed: int | None = None,
        work_dir: str = "gcmc_run",
        db_file: str | None = None,
    ):
        super().__init__(evaluator=evaluator, workdir=work_dir, seed=seed)
        self.slab = slab.copy()
        self.blocks = blocks
        self.temperature = temperature
        self.work_dir = Path(work_dir)

        # Database recording
        self.db_file = db_file
        self.db = None

        # Build chemical potential dict
        if chemical_potentials is not None:
            self.chemical_potentials = chemical_potentials
        else:
            self.chemical_potentials = {b.name: b.chemical_potential for b in blocks}

        # Region
        if region is not None:
            self.region = region
        else:
            # Default: box above slab
            z_max = slab.positions[:, 2].max()
            self.region = BoxRegion(
                cell=slab.cell,
                z_min=z_max + 1.0,
                z_max=z_max + 8.0,
                pbc=slab.pbc,
            )

        # Build moves
        self.moves: list[MCMove] = []
        self.move_probs: list[float] = []
        self._build_moves(move_weights, max_displacement)

        # Ensemble
        self.ensemble = GCEnsemble(
            temperature=self.temperature,
            chemical_potentials=self.chemical_potentials,
        )

        # RNG
        self.rng = np.random.default_rng(seed)

        # State
        self.current_atoms: Atoms | None = None
        self.current_energy: float = np.inf
        self.best_atoms: Atoms | None = None
        self.best_energy: float = np.inf
        self.trajectory: list[dict] = []
        self.step_count = 0

        # Per-tag bond constraint metadata from inserted MolecularBlocks.
        # Populated when InsertMove inserts a block that has .bonds defined.
        self._block_bonds: dict[int, BlockBondInfo] = {}

    def _build_moves(
        self,
        move_weights: dict[str, float] | None,
        max_displacement: float,
    ) -> None:
        """Initialize MC moves and their selection probabilities."""
        defaults = {"insert": 0.3, "delete": 0.3, "displace": 0.3, "swap": 0.1}
        w = move_weights or defaults

        move_map: dict[str, MCMove] = {
            "insert": InsertMove(self.blocks, self.region),
            "delete": DeleteMove(self.blocks),
            "swap": SwapMove(self.blocks),
            "displace": DisplaceMove(max_displacement),
        }

        total = sum(w.get(k, 0.0) for k in move_map)
        for name, move in move_map.items():
            prob = w.get(name, 0.0) / total
            if prob > 0:
                self.moves.append(move)
                self.move_probs.append(prob)

    def setup(self) -> None:
        """Initialize the GCMC run."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._is_setup = True

        # Open database for recording
        if self.db_file is not None:
            db_path = self.work_dir / self.db_file
            self.db = ase.db.connect(str(db_path))
            logger.info("Database opened: %s", db_path)

        # Initial configuration = bare slab or slab with existing adsorbates
        self.current_atoms = self.slab.copy()
        result = self.evaluator.evaluate(self.current_atoms, relax=True)
        self.current_energy = result.get_potential_energy()
        self.current_atoms = result

        self.best_atoms = self.current_atoms.copy()
        self.best_energy = self.current_energy

        # Record initial structure
        self._record_to_db(
            self.current_atoms,
            step=0,
            move="init",
            accepted=True,
        )

        logger.info(
            "GCMC initialized: E=%.4f eV, T=%.1f K",
            self.current_energy,
            self.temperature,
        )

    def step(self) -> dict:
        """Perform one GCMC step."""
        self.step_count += 1

        # Select a move
        move_idx = self.rng.choice(len(self.moves), p=self.move_probs)
        move = self.moves[move_idx]

        # Propose
        new_atoms, log_ratio, move_info = move.propose(self.current_atoms, self.rng)

        if new_atoms is None:
            self.ensemble.n_invalid += 1
            return {
                "step": self.step_count,
                "accepted": False,
                "reason": move_info.get("reason", "invalid_proposal"),
                "energy": self.current_energy,
                "best_energy": self.best_energy,
                "n_adsorbates": self._count_adsorbates(self.current_atoms),
            }

        # Track per-block bond metadata for the newly inserted block
        new_block_bonds = dict(self._block_bonds)
        move_type = move_info.get("move", "")
        if move_type == "insert":
            block_name = move_info.get("species", "")
            block = self._find_block(block_name)
            if block is not None and block.bonds:
                new_tag = max(new_atoms.get_tags())
                new_block_bonds[new_tag] = BlockBondInfo(
                    bonds=block.bonds,
                    spring_constant=block.spring_constant,
                    threshold_scale=block.threshold_scale,
                )
        elif move_type == "delete":
            deleted_tag = move_info.get("deleted_tag")
            if deleted_tag is not None:
                new_block_bonds.pop(deleted_tag, None)

        # Apply bond constraints before evaluation
        if new_block_bonds:
            self._apply_bond_constraints(new_atoms, block_bonds=new_block_bonds)

        # Evaluate proposed configuration
        try:
            relaxed = self.evaluator.evaluate(new_atoms, relax=True)
            new_energy = relaxed.get_potential_energy()
        except Exception as e:
            logger.warning("Evaluation failed at step %d: %s", self.step_count, e)
            self.ensemble.n_invalid += 1
            return {
                "step": self.step_count,
                "accepted": False,
                "reason": "evaluation_failed",
                "energy": self.current_energy,
                "best_energy": self.best_energy,
                "n_adsorbates": self._count_adsorbates(self.current_atoms),
            }

        # Acceptance criterion
        accepted = self.ensemble.accept(
            self.current_energy, new_energy, move_info, log_ratio, self.rng
        )

        if accepted:
            self.current_atoms = relaxed
            self.current_energy = new_energy
            self._block_bonds = new_block_bonds

            # Update best
            if new_energy < self.best_energy:
                self.best_energy = new_energy
                self.best_atoms = relaxed.copy()
                logger.info(
                    "New best at step %d: E=%.4f eV (%s)",
                    self.step_count,
                    new_energy,
                    move_info.get("move"),
                )

        # Log trajectory
        step_info = {
            "step": self.step_count,
            "accepted": accepted,
            "move": move_info.get("move", "unknown"),
            "energy": self.current_energy,
            "new_energy": new_energy,
            "best_energy": self.best_energy,
            "n_adsorbates": self._count_adsorbates(self.current_atoms),
            "acceptance_rate": self.ensemble.acceptance_rate,
        }
        self.trajectory.append(step_info)

        # Record accepted configuration to database
        if accepted:
            self._record_to_db(
                self.current_atoms,
                step=self.step_count,
                move=move_info.get("move", "unknown"),
                accepted=True,
            )

        if self.step_count % 100 == 0:
            logger.info(
                "Step %d: E=%.4f, best=%.4f, acc_rate=%.3f, n_ads=%d",
                self.step_count,
                self.current_energy,
                self.best_energy,
                self.ensemble.acceptance_rate,
                step_info["n_adsorbates"],
            )

        return step_info

    def get_results(self) -> list[Atoms]:
        """Return trajectory of accepted configurations."""
        return [self.current_atoms.copy()] if self.current_atoms else []

    def get_best(self, n: int = 1) -> list[Atoms]:
        """Return the best configuration found."""
        if self.best_atoms is not None:
            return [self.best_atoms.copy()]
        return []

    def _count_adsorbates(self, atoms: Atoms) -> int:
        tags = atoms.get_tags()
        return len(set(t for t in tags if t > 0))

    def _find_block(self, name: str) -> "MolecularBlock | None":
        """Find a MolecularBlock by name."""
        for b in self.blocks:
            if b.name == name:
                return b
        return None

    def _apply_bond_constraints(
        self,
        atoms: Atoms,
        block_bonds: dict[int, BlockBondInfo] | None = None,
    ) -> None:
        """Apply FixAtoms + Hookean bond constraints to atoms in-place.

        Per-block spring_constant and threshold_scale are read from the
        ``BlockBondInfo`` stored when each molecule was inserted.
        """
        fix_atoms = [c for c in atoms.constraints if isinstance(c, FixAtoms)]
        hookean = get_adsorbate_bond_constraints(
            atoms,
            block_bonds=block_bonds,
        )
        atoms.set_constraint(fix_atoms + hookean)
        if hookean:
            logger.debug(
                "Applied %d Hookean bond constraints",
                len(hookean),
            )

    def _record_to_db(
        self,
        atoms: Atoms,
        step: int,
        move: str,
        accepted: bool,
    ) -> None:
        """Record a structure to the ASE database."""
        if self.db is None:
            return
        # Write a clean copy (no calculator, no constraints)
        clean = atoms.copy()
        clean.calc = None
        clean.constraints = []
        energy = atoms.get_potential_energy() if atoms.calc else 0.0
        n_ads = self._count_adsorbates(atoms)
        self.db.write(
            clean,
            data={
                "step": step,
                "total_energy": energy,
                "move": move,
                "accepted": accepted,
                "n_adsorbates": n_ads,
                "chem_formula": atoms.get_chemical_formula(),
            },
        )

    @classmethod
    def from_config(cls, config: dict) -> "GCMCRunner":
        """Build GCMCRunner from a config dict (parsed from YAML)."""
        from gs_gmlip.structure.composition import get_block

        # Build blocks
        blocks = []
        for bconf in config.get("blocks", []):
            name = bconf["name"]
            mu = bconf.get("chemical_potential", 0.0)
            block = get_block(name, mu=mu)
            blocks.append(block)

        # Load slab
        from ase.io import read

        slab = read(config["slab_file"])

        # Build evaluator
        from gs_gmlip.cli import build_evaluator

        evaluator = build_evaluator(config.get("evaluator", {}))

        return cls(
            slab=slab,
            blocks=blocks,
            evaluator=evaluator,
            temperature=config.get("temperature", 300.0),
            chemical_potentials=config.get("chemical_potentials"),
            move_weights=config.get("move_weights"),
            max_displacement=config.get("max_displacement", 1.0),
            seed=config.get("seed"),
            work_dir=config.get("work_dir", "gcmc_run"),
            db_file=config.get("db_file"),
        )
