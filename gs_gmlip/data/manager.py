"""Workflow data flow manager.

Coordinates data flow between search, evaluate, and analysis modules.
Handles checkpointing, logging, and result collection.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ase import Atoms

from gs_gmlip.data.io import read_structures, write_structures

logger = logging.getLogger(__name__)


class WorkflowManager:
    """Manages the data flow of a global search workflow.

    Provides a working directory with:
    - candidates/ : Generated candidate structures
    - relaxed/    : Relaxed structures
    - results/    : Final results and analysis output
    - checkpoints/: Checkpoint files for restart

    Parameters
    ----------
    workdir : str or Path
        Working directory for the workflow.
    """

    def __init__(self, workdir: str | Path):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)

        self.candidates_dir = self.workdir / "candidates"
        self.relaxed_dir = self.workdir / "relaxed"
        self.results_dir = self.workdir / "results"
        self.checkpoint_dir = self.workdir / "checkpoints"

        for d in [
            self.candidates_dir,
            self.relaxed_dir,
            self.results_dir,
            self.checkpoint_dir,
        ]:
            d.mkdir(exist_ok=True)

        self._step = 0
        self._history: list[dict] = []

    @property
    def step(self) -> int:
        return self._step

    def save_candidate(self, atoms: Atoms, label: str = "candidate") -> Path:
        """Save a candidate structure."""
        filename = self.candidates_dir / f"{label}_{self._step:06d}.xyz"
        write_structures(filename, atoms)
        return filename

    def save_relaxed(self, atoms: Atoms, label: str = "relaxed") -> Path:
        """Save a relaxed structure."""
        filename = self.relaxed_dir / f"{label}_{self._step:06d}.xyz"
        write_structures(filename, atoms)
        return filename

    def save_checkpoint(self, state: dict) -> Path:
        """Save a checkpoint for restart.

        Parameters
        ----------
        state : dict
            Serializable state dictionary.
        """
        filename = self.checkpoint_dir / f"checkpoint_{self._step:06d}.json"
        with open(filename, "w") as f:
            json.dump(state, f, indent=2, default=str)
        logger.info(f"Checkpoint saved: {filename}")
        return filename

    def load_checkpoint(self, step: int | None = None) -> dict | None:
        """Load a checkpoint.

        Parameters
        ----------
        step : int, optional
            Specific step to load. If None, loads the latest.
        """
        if step is not None:
            filename = self.checkpoint_dir / f"checkpoint_{step:06d}.json"
        else:
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.json"))
            if not checkpoints:
                return None
            filename = checkpoints[-1]

        if not filename.exists():
            return None

        with open(filename) as f:
            state = json.load(f)
        logger.info(f"Checkpoint loaded: {filename}")
        return state

    def log_step(self, info: dict):
        """Log information about a search step."""
        info["step"] = self._step
        self._history.append(info)
        self._step += 1

    def get_all_relaxed(self) -> list[Atoms]:
        """Load all relaxed structures from disk."""
        structures = []
        for f in sorted(self.relaxed_dir.glob("*.xyz")):
            structures.extend(read_structures(f))
        return structures

    def save_results_summary(self, results: dict) -> Path:
        """Save a results summary JSON."""
        filename = self.results_dir / "summary.json"
        with open(filename, "w") as f:
            json.dump(results, f, indent=2, default=str)
        return filename

    def save_history(self) -> Path:
        """Save the step history log."""
        filename = self.results_dir / "history.json"
        with open(filename, "w") as f:
            json.dump(self._history, f, indent=2, default=str)
        return filename
