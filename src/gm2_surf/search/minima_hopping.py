"""Minima hopping global optimization.

Implements the algorithm of Goedecker (``Goedecker_JCP_2004``: S. Goedecker,
J. Chem. Phys. 120, 9911 (2004)). From the current local minimum a short
molecular-dynamics run at temperature ``T`` tries to escape over a low
barrier; the escaped geometry is relaxed to a new minimum, which is accepted
if its energy rises by less than ``E_diff``. Both parameters adapt to the
history of the search:

- MD fell back into the current minimum: ``T *= beta1`` (heat up).
- The minimum reached was visited before: ``T *= beta2`` (heat up).
- The minimum reached is new: ``T *= beta3`` (cool down).
- The move is accepted: ``E_diff *= alpha1`` (be stricter).
- The move is rejected: ``E_diff *= alpha2`` (be more permissive).

Each MD escape stops once the potential energy has passed ``mdmin`` minima
along the trajectory, which favours crossings of low barriers
(Bell-Evans-Polanyi principle). Frozen atoms and other ASE constraints of
the structure are honoured by both the MD and the relaxation.
"""

from __future__ import annotations

import logging

import numpy as np
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

from gm2_surf.evaluator.base import BaseEvaluator
from gm2_surf.interface.core import Interface
from gm2_surf.search.base import BaseSearcher

logger = logging.getLogger(__name__)


class MinimaHoppingRunner(BaseSearcher):
    """Minima hopping search starting from one structure.

    Args:
        evaluator: Energy evaluator; its relaxation settings define a "minimum".
        start: Starting structure (composition is conserved).
        workdir: Working directory.
        temperature: Initial MD temperature ``T`` in K.
        ediff: Initial acceptance threshold ``E_diff`` in eV.
        beta1: Temperature factor when the MD returns to the current minimum (> 1).
        beta2: Temperature factor when a known minimum is reached (> 1).
        beta3: Temperature factor when a new minimum is reached (< 1).
        alpha1: ``E_diff`` factor after an accepted move (< 1).
        alpha2: ``E_diff`` factor after a rejected move (> 1).
        mdmin: Number of potential-energy minima to pass before an escape ends.
        timestep: MD time step in fs.
        max_md_steps: Upper bound on the length of one escape, in MD steps.
        energy_tolerance: Two minima closer than this in energy (eV), and with
            matching fingerprints, are the same minimum.
        seed: Random seed for the MD velocities.
    """

    def __init__(
        self,
        evaluator: BaseEvaluator,
        start: Interface,
        workdir: str = "minima_hopping",
        temperature: float = 1000.0,
        ediff: float = 0.5,
        beta1: float = 1.1,
        beta2: float = 1.1,
        beta3: float = 1.0 / 1.1,
        alpha1: float = 0.98,
        alpha2: float = 1.0 / 0.98,
        mdmin: int = 2,
        timestep: float = 1.0,
        max_md_steps: int = 500,
        energy_tolerance: float = 2e-3,
        seed: int | None = None,
    ):
        super().__init__(evaluator, workdir, seed)
        self.start = start
        self.temperature = float(temperature)
        self.ediff = float(ediff)
        self.beta1, self.beta2, self.beta3 = float(beta1), float(beta2), float(beta3)
        self.alpha1, self.alpha2 = float(alpha1), float(alpha2)
        self.mdmin = int(mdmin)
        self.timestep = float(timestep)
        self.max_md_steps = int(max_md_steps)
        self.energy_tolerance = float(energy_tolerance)
        self.rng = np.random.default_rng(seed)
        self.current: Interface | None = None
        self.current_energy = float("inf")
        self.best: Interface | None = None
        self.best_energy = float("inf")
        self.visited: list[tuple[float, np.ndarray]] = []

    # ------------------------------------------------------------------
    # BaseSearcher interface
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Relax the starting structure; it becomes the first minimum."""
        self.current = self.evaluator.evaluate(self.start.copy(), relax=True)
        self.current_energy = float(self.current.interface.get_potential_energy())
        self.best, self.best_energy = self.current, self.current_energy
        self.visited = [(self.current_energy, self.current.fingerprint)]
        self._results.append(self.current)
        self._is_setup = True

    def step(self) -> dict:
        """One hop: MD escape, relaxation, feedback on ``T`` and ``E_diff``."""
        if self.current is None:
            return {"status": "not_initialized"}
        candidate = self.evaluator.evaluate(self._escape(), relax=True)
        energy = float(candidate.interface.get_potential_energy())
        fingerprint = candidate.fingerprint
        info = {"energy": energy, "temperature": self.temperature, "ediff": self.ediff}

        if self._same(
            energy, fingerprint, self.current_energy, self.current.fingerprint
        ):
            self.temperature *= self.beta1
            return {**info, "status": "returned", "accepted": False}

        known = any(self._same(energy, fingerprint, e, f) for e, f in self.visited)
        self.temperature *= self.beta2 if known else self.beta3
        if not known:
            self.visited.append((energy, fingerprint))
            self._results.append(candidate)
            if energy < self.best_energy:
                self.best, self.best_energy = candidate, energy

        accepted = energy - self.current_energy < self.ediff
        if accepted:
            self.current, self.current_energy = candidate, energy
            self.ediff *= self.alpha1
        else:
            self.ediff *= self.alpha2
        return {**info, "status": "known" if known else "new", "accepted": accepted}

    def get_best(self, n: int = 1) -> list[Interface]:
        """The lowest-energy minima found."""
        return super().get_best(n)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _same(self, e1: float, f1: np.ndarray, e2: float, f2: np.ndarray) -> bool:
        if abs(e1 - e2) > self.energy_tolerance or len(f1) != len(f2):
            return False
        if len(f1) == 0:
            return True
        similarity = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-12)
        return bool(similarity > 0.999)

    def _escape(self) -> Interface:
        """Run MD from the current minimum until ``mdmin`` energy minima have been passed."""
        escaped = self.current.copy()
        atoms = escaped.interface
        atoms.calc = self.evaluator.get_calculator()
        MaxwellBoltzmannDistribution(
            atoms, temperature_K=self.temperature, rng=self.rng
        )
        dynamics = VelocityVerlet(
            atoms, timestep=self.timestep * units.fs, logfile=None
        )
        energies = [float(atoms.get_potential_energy())]
        passed = 0
        for _ in range(self.max_md_steps):
            dynamics.run(1)
            energies.append(float(atoms.get_potential_energy()))
            if (
                len(energies) >= 3
                and energies[-2] < energies[-3]
                and energies[-2] < energies[-1]
            ):
                passed += 1
                if passed >= self.mdmin:
                    break
        atoms.set_momenta(np.zeros((len(atoms), 3)))
        escaped.substrate = atoms[: len(escaped.substrate)]
        escaped._reset_cache()
        return escaped
