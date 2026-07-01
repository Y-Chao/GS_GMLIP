"""GCMC runner: orchestrates the grand canonical MC search.

Combines Moves, GCEnsemble, and BaseEvaluator into a complete
structure search workflow with trajectory logging.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from gs_gmlip.evaluator.base import BaseEvaluator
from gs_gmlip.interface.core import Interface
from gs_gmlip.search.base import BaseSearcher
from gs_gmlip.search.gcmc.ensemble import GCEnsemble
from gs_gmlip.search.gcmc.moves import (
    DeleteMove,
    DisplaceMove,
    InsertMove,
    MCMove,
    SwapMove,
)
from gs_gmlip.search.region import BoxRegion, Region

logger = logging.getLogger(__name__)


class GCMCRunner(BaseSearcher):
    """Grand Canonical Monte Carlo structure search on surfaces.

    Parameters
    ----------
    evaluator : BaseEvaluator
        Energy evaluator.
    substrate : Interface
        Template interface (bare slab or slab + initial adsorbates).
    blocks : list[Atoms]
        Molecular / atomic blocks available for insertion.
    block_names : list[str]
        Names corresponding to *blocks*.
    chemical_potentials : dict[str, float]
        Species name → chemical potential (eV).
    temperature : float
        Temperature in Kelvin.
    region : Region, optional
        Region for insert moves.  Defaults to a box above the slab.
    move_weights : dict[str, float], optional
        Relative weights for each move type.
        Default: insert=0.3, delete=0.3, displace=0.3, swap=0.1.
    max_displacement : float
        Maximum displacement magnitude (Å) for displace moves.
    workdir : str
        Working directory.
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        evaluator: BaseEvaluator,
        substrate: Interface,
        blocks: list,
        block_names: list[str],
        chemical_potentials: dict[str, float],
        temperature: float = 300.0,
        region: Region | None = None,
        move_weights: dict[str, float] | None = None,
        max_displacement: float = 1.0,
        workdir: str = "gcmc_run",
        seed: int | None = None,
    ):
        super().__init__(evaluator, workdir, seed)
        self.substrate_template = substrate
        self.blocks = blocks
        self.block_names = block_names
        self.chemical_potentials = chemical_potentials
        self.temperature = temperature
        self.max_displacement = max_displacement

        self.rng = np.random.default_rng(seed)

        # --- Region ---
        if region is not None:
            self.region = region
        else:
            slab = substrate.interface
            z_max = slab.positions[:, 2].max() if len(slab) > 0 else 0.0
            self.region = BoxRegion(
                cell=slab.get_cell(),
                z_min=z_max + 1.0,
                z_max=z_max + 8.0,
            )

        # --- Moves ---
        self.moves: list[MCMove] = []
        self.move_probs: list[float] = []
        self._build_moves(move_weights or {})

        # --- Ensemble ---
        self.ensemble = GCEnsemble(
            temperature=self.temperature,
            chemical_potentials=self.chemical_potentials,
        )

        # --- State ---
        self.current: Interface | None = None
        self.current_energy: float = float("inf")
        self.best: Interface | None = None
        self.best_energy: float = float("inf")
        self.step_count: int = 0

    def _build_moves(self, move_weights: dict[str, float]) -> None:
        """Build the move set from weight specifications."""
        defaults = {"insert": 0.3, "delete": 0.3, "displace": 0.3, "swap": 0.1}
        w = {**defaults, **move_weights}

        # Determine allowed elements for swap (single-atom symbols from blocks)
        allowed_elements = list(dict.fromkeys(  # preserve order, deduplicate
            b.get_chemical_symbols()[0]
            for b in self.blocks
            if len(b) == 1
        ))

        move_map: dict[str, MCMove] = {
            "insert": InsertMove(
                self.blocks, self.block_names, self.chemical_potentials, self.region
            ),
            "delete": DeleteMove(self.chemical_potentials),
            "displace": DisplaceMove(self.max_displacement),
        }
        if len(allowed_elements) >= 2:
            move_map["swap"] = SwapMove(allowed_elements, self.chemical_potentials)

        total = sum(w.get(k, 0.0) for k in move_map)
        for name, move in move_map.items():
            prob = w.get(name, 0.0) / total if total > 0 else 0.0
            if prob > 0:
                self.moves.append(move)
                self.move_probs.append(prob)

    # ------------------------------------------------------------------
    # BaseSearcher interface
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Initialize: evaluate the starting configuration."""
        Path(self.workdir).mkdir(parents=True, exist_ok=True)

        self.current = self.substrate_template.copy()
        self.current = self.evaluator.evaluate(self.current, relax=True)
        self.current_energy = self.current.interface.get_potential_energy()
        self.best = self.current.copy()
        self.best_energy = self.current_energy
        self._results.append(self.current)
        self._is_setup = True

        logger.info(
            "GCMC initialized: E=%.4f eV, T=%.1f K",
            self.current_energy,
            self.temperature,
        )

    def step(self) -> dict:
        """Perform one GCMC step: select move → propose → evaluate → accept/reject."""
        if self.current is None:
            return {"status": "not_initialized"}

        self.step_count += 1

        # Select move
        move_idx = self.rng.choice(len(self.moves), p=self.move_probs)
        move = self.moves[move_idx]

        # Propose
        new_iface, log_ratio, move_info = move.propose(self.current, self.rng)

        if new_iface is None:
            self.ensemble.n_invalid += 1
            return {
                "step": self.step_count,
                "accepted": False,
                "reason": move_info.get("reason", "invalid_proposal"),
                "energy": self.current_energy,
                "best_energy": self.best_energy,
            }

        # Evaluate proposed configuration
        try:
            new_iface = self.evaluator.evaluate(new_iface, relax=True)
            new_energy = new_iface.interface.get_potential_energy()
        except Exception as e:
            logger.warning("Evaluation failed at step %d: %s", self.step_count, e)
            self.ensemble.n_invalid += 1
            return {
                "step": self.step_count,
                "accepted": False,
                "reason": "evaluation_failed",
                "energy": self.current_energy,
                "best_energy": self.best_energy,
            }

        # Acceptance
        accepted = self.ensemble.accept(
            self.current_energy, new_energy, move_info, log_ratio, self.rng
        )

        if accepted:
            self.current = new_iface
            self.current_energy = new_energy
            self._results.append(self.current)

            if new_energy < self.best_energy:
                self.best = new_iface.copy()
                self.best_energy = new_energy
                logger.info(
                    "New best at step %d: E=%.4f eV (%s)",
                    self.step_count,
                    new_energy,
                    move_info.get("move"),
                )

        if self.step_count % 100 == 0:
            logger.info(
                "Step %d: E=%.4f, best=%.4f, acc_rate=%.3f",
                self.step_count,
                self.current_energy,
                self.best_energy,
                self.ensemble.acceptance_rate,
            )

        return {
            "step": self.step_count,
            "accepted": accepted,
            "move": move_info.get("move", "unknown"),
            "energy": self.current_energy,
            "new_energy": new_energy,
            "best_energy": self.best_energy,
            "acceptance_rate": self.ensemble.acceptance_rate,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_results(self) -> list[Interface]:
        """Return all accepted configurations."""
        return list(self._results)

    def get_best(self, n: int = 1) -> list[Interface]:
        """Return the best configuration found."""
        if self.best is not None:
            return [self.best.copy()]
        return []

    def get_summary(self) -> dict:
        """Return ensemble statistics summary."""
        return self.ensemble.get_summary()

    @property
    def n_adsorbates(self) -> int:
        """Number of adsorbate groups on the current configuration."""
        if self.current is None:
            return 0
        return len(self.current.adsList)

    @property
    def current_formula(self) -> str:
        """Chemical formula of the current interface."""
        if self.current is None:
            return ""
        return self.current.interface.get_chemical_formula()
