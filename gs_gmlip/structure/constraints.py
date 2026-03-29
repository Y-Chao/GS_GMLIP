"""Adsorbate bond constraints for structure searching.

Provides Hookean spring constraints that preserve intramolecular bonds
within adsorbate molecules during relaxation. Without these constraints,
multi-atom adsorbates (CO2, COOH, H2O, etc.) can dissociate when the
optimizer moves atoms that are free to drift apart.
"""

from __future__ import annotations

import logging

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms, Hookean

from gs_gmlip.structure.bonds import BondData

logger = logging.getLogger(__name__)


def get_adsorbate_bond_constraints(
    atoms: Atoms,
    spring_constant: float = 15.0,
    threshold_scale: float = 1.3,
    bond_data: BondData | None = None,
) -> list[Hookean]:
    """Build Hookean constraints for all intramolecular bonds in adsorbates.

    For each adsorbate molecule (atoms sharing the same tag > 0),
    detect internal bonds and create Hookean spring constraints
    to prevent bond breaking during relaxation.

    Parameters
    ----------
    atoms : Atoms
        Structure with tagged adsorbates (tag > 0).
    spring_constant : float
        Spring constant k in eV/Å². Applied when bond length exceeds
        the threshold. Typical values: 5–20 eV/Å².
    threshold_scale : float
        Multiply current bond length by this factor to set the threshold
        distance ``rt``. Values < 1.5 keep bonds tight; 1.3 is reasonable.
    bond_data : BondData, optional
        Bond detection parameters. Defaults to standard covalent radii × 1.2.

    Returns
    -------
    constraints : list of Hookean
        One Hookean constraint per detected intramolecular bond.
    """
    if bond_data is None:
        bond_data = BondData()

    tags = atoms.get_tags()
    numbers = atoms.get_atomic_numbers()
    symbols = atoms.get_chemical_symbols()

    # Group atoms by adsorbate tag
    groups: dict[int, list[int]] = {}
    for i, t in enumerate(tags):
        if t > 0:
            groups.setdefault(t, []).append(i)

    constraints = []
    for tag, indices in groups.items():
        if len(indices) < 2:
            continue  # Single atom, no internal bonds

        # Check all pairs within this molecule
        for ii, idx_a in enumerate(indices):
            for idx_b in indices[ii + 1 :]:
                dist = atoms.get_distance(idx_a, idx_b, mic=True)
                max_bl = bond_data.get_max_bond_length(
                    numbers[idx_a],
                    numbers[idx_b],
                    symbols[idx_a],
                    symbols[idx_b],
                )
                if dist < max_bl:
                    # Bonded pair — add spring constraint
                    rt = dist * threshold_scale
                    constraints.append(
                        Hookean(a1=idx_a, a2=idx_b, k=spring_constant, rt=rt)
                    )
                    logger.debug(
                        "Hookean: tag=%d %s(%d)-%s(%d) d=%.3f rt=%.3f k=%.1f",
                        tag,
                        symbols[idx_a],
                        idx_a,
                        symbols[idx_b],
                        idx_b,
                        dist,
                        rt,
                        spring_constant,
                    )

    if constraints:
        logger.debug(
            "Built %d Hookean constraints for %d multi-atom adsorbates",
            len(constraints),
            sum(1 for idx in groups.values() if len(idx) >= 2),
        )

    return constraints


def apply_constraints(
    atoms: Atoms,
    spring_constant: float = 15.0,
    threshold_scale: float = 1.3,
    bond_data: BondData | None = None,
) -> None:
    """Apply FixAtoms + Hookean bond constraints to an atoms object in-place.

    Preserves existing FixAtoms constraints and adds Hookean constraints
    for all intramolecular adsorbate bonds.

    Parameters
    ----------
    atoms : Atoms
        Structure to constrain. Modified in-place.
    spring_constant : float
        Hookean spring constant (eV/Å²).
    threshold_scale : float
        Bond threshold scale factor.
    bond_data : BondData, optional
        Bond detection parameters.
    """
    # Preserve FixAtoms
    fix_atoms = [c for c in atoms.constraints if isinstance(c, FixAtoms)]

    # Build Hookean constraints for adsorbate bonds
    hookean = get_adsorbate_bond_constraints(
        atoms,
        spring_constant=spring_constant,
        threshold_scale=threshold_scale,
        bond_data=bond_data,
    )

    atoms.set_constraint(fix_atoms + hookean)
