"""Grand canonical ensemble and Metropolis acceptance.

Implements the GC ensemble acceptance criterion:
    P_acc = min(1, exp(-beta * dE + beta * mu * dn + log_proposal_ratio))

where dE is the energy change, mu the chemical potential, dn the
change in particle number, and log_proposal_ratio accounts for asymmetric
proposal distributions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GCEnsemble:
    """Grand canonical Monte Carlo ensemble.

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin.
    chemical_potentials : dict[str, float]
        Species name -> chemical potential (eV).
    kB : float
        Boltzmann constant in eV/K (default: ASE value).
    """

    temperature: float
    chemical_potentials: dict[str, float]
    kB: float = 8.617333262e-5  # eV/K

    # Statistics
    n_accepted: int = field(default=0, init=False)
    n_rejected: int = field(default=0, init=False)
    n_invalid: int = field(default=0, init=False)
    move_stats: dict = field(default_factory=dict, init=False)

    @property
    def beta(self) -> float:
        return 1.0 / (self.kB * self.temperature)

    @property
    def acceptance_rate(self) -> float:
        total = self.n_accepted + self.n_rejected
        if total == 0:
            return 0.0
        return self.n_accepted / total

    def accept(
        self,
        energy_old: float,
        energy_new: float,
        move_info: dict,
        log_proposal_ratio: float,
        rng: np.random.Generator,
    ) -> bool:
        """Metropolis-Hastings acceptance criterion for GC ensemble.

        Parameters
        ----------
        energy_old : float
            Energy of current configuration (eV).
        energy_new : float
            Energy of proposed configuration (eV).
        move_info : dict
            Information from the move proposal (contains move type, species, mu).
        log_proposal_ratio : float
            Log of the proposal probability ratio.
        rng : Generator
            Random number generator.

        Returns
        -------
        accepted : bool
        """
        move_type = move_info.get("move", "unknown")

        # Energy difference
        dE = energy_new - energy_old

        # Chemical potential contribution
        mu_contribution = 0.0
        mu = move_info.get("chemical_potential", 0.0)
        if move_type == "insert":
            mu_contribution = mu  # Adding a particle: +mu
        elif move_type == "delete":
            mu_contribution = -mu  # Removing a particle: -mu
        elif move_type == "swap":
            # For swap, need difference of chemical potentials
            old_species = move_info.get("old_species", "")
            new_species = move_info.get("new_species", "")
            mu_old = self.chemical_potentials.get(old_species, 0.0)
            mu_new = self.chemical_potentials.get(new_species, 0.0)
            mu_contribution = mu_new - mu_old

        # Acceptance probability
        log_p_accept = (
            -self.beta * dE + self.beta * mu_contribution + log_proposal_ratio
        )

        accepted = bool(log_p_accept >= 0 or rng.random() < np.exp(log_p_accept))

        # Update statistics
        self._update_stats(move_type, accepted)

        if accepted:
            self.n_accepted += 1
            logger.debug(
                "ACCEPTED %s: dE=%.4f, mu_contrib=%.4f, log_p=%.4f",
                move_type,
                dE,
                mu_contribution,
                log_p_accept,
            )
        else:
            self.n_rejected += 1
            logger.debug(
                "REJECTED %s: dE=%.4f, mu_contrib=%.4f, log_p=%.4f",
                move_type,
                dE,
                mu_contribution,
                log_p_accept,
            )

        return accepted

    def _update_stats(self, move_type: str, accepted: bool) -> None:
        if move_type not in self.move_stats:
            self.move_stats[move_type] = {"accepted": 0, "rejected": 0}
        if accepted:
            self.move_stats[move_type]["accepted"] += 1
        else:
            self.move_stats[move_type]["rejected"] += 1

    def get_summary(self) -> dict:
        """Get summary of ensemble statistics."""
        return {
            "temperature": self.temperature,
            "chemical_potentials": self.chemical_potentials,
            "total_accepted": self.n_accepted,
            "total_rejected": self.n_rejected,
            "total_invalid": self.n_invalid,
            "acceptance_rate": self.acceptance_rate,
            "move_stats": dict(self.move_stats),
        }

    def reset_stats(self) -> None:
        self.n_accepted = 0
        self.n_rejected = 0
        self.n_invalid = 0
        self.move_stats.clear()
