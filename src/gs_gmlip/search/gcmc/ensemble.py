"""Grand canonical ensemble and Metropolis acceptance criterion.

Implements the GC ensemble acceptance probability:

    P_acc = min(1, exp(-β·ΔE + β·μ_contrib + log_proposal_ratio))

where ΔE is the energy change, μ_contrib the chemical-potential
contribution (depending on move type), and log_proposal_ratio accounts
for asymmetric proposal distributions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Boltzmann constant in eV/K
_kB = 8.617333262145e-5


@dataclass
class GCEnsemble:
    """Grand canonical Monte Carlo ensemble.

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin.
    chemical_potentials : dict[str, float]
        Species name → chemical potential (eV).
    """

    temperature: float = 300.0
    chemical_potentials: dict[str, float] = field(default_factory=dict)

    # Statistics (accumulated during the run)
    n_accepted: int = field(default=0, init=False)
    n_rejected: int = field(default=0, init=False)
    n_invalid: int = field(default=0, init=False)
    move_stats: dict[str, dict[str, int]] = field(default_factory=dict, init=False)

    @property
    def beta(self) -> float:
        """Inverse temperature 1/(k_B T) in 1/eV."""
        return 1.0 / (_kB * self.temperature)

    @property
    def acceptance_rate(self) -> float:
        """Overall acceptance rate (excludes invalid proposals)."""
        total = self.n_accepted + self.n_rejected
        if total == 0:
            return 0.0
        return self.n_accepted / total

    # ------------------------------------------------------------------
    # Acceptance
    # ------------------------------------------------------------------

    def accept(
        self,
        energy_old: float,
        energy_new: float,
        move_info: dict,
        log_proposal_ratio: float,
        rng: np.random.Generator,
    ) -> bool:
        """Metropolis-Hastings acceptance for the GC ensemble.

        Parameters
        ----------
        energy_old : float
            Energy of the current configuration (eV).
        energy_new : float
            Energy of the proposed configuration (eV).
        move_info : dict
            Metadata from the move proposal (must contain "move" and
            "chemical_potential" keys).
        log_proposal_ratio : float
            Log of the forward/reverse proposal probability ratio.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        accepted : bool
        """
        move_type = move_info.get("move", "unknown")
        dE = energy_new - energy_old

        # Chemical potential contribution
        mu_contrib = 0.0
        mu = move_info.get("chemical_potential", 0.0)
        if move_type == "insert":
            mu_contrib = mu
        elif move_type == "delete":
            mu_contrib = -mu
        elif move_type == "swap":
            mu_contrib = mu  # already computed as mu_new - mu_old in the move

        log_p_accept = -self.beta * dE + self.beta * mu_contrib + log_proposal_ratio

        accepted = bool(log_p_accept >= 0 or rng.random() < np.exp(min(log_p_accept, 0)))

        # Update stats
        self._update_stats(move_type, accepted)

        if accepted:
            self.n_accepted += 1
        else:
            self.n_rejected += 1

        return accepted

    def _update_stats(self, move_type: str, accepted: bool) -> None:
        """Increment per-move-type counters."""
        if move_type not in self.move_stats:
            self.move_stats[move_type] = {"accepted": 0, "rejected": 0}
        if accepted:
            self.move_stats[move_type]["accepted"] += 1
        else:
            self.move_stats[move_type]["rejected"] += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_summary(self) -> dict:
        """Return a summary dict of ensemble statistics."""
        return {
            "temperature": self.temperature,
            "chemical_potentials": dict(self.chemical_potentials),
            "total_accepted": self.n_accepted,
            "total_rejected": self.n_rejected,
            "total_invalid": self.n_invalid,
            "acceptance_rate": self.acceptance_rate,
            "move_stats": {k: dict(v) for k, v in self.move_stats.items()},
        }

    def reset_stats(self) -> None:
        """Reset all accumulated statistics."""
        self.n_accepted = 0
        self.n_rejected = 0
        self.n_invalid = 0
        self.move_stats.clear()
