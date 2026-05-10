"""Adsorbate bond constraints for structure searching.

Provides Hookean spring constraints that preserve intramolecular bonds
within adsorbate molecules during relaxation. Without these constraints,
multi-atom adsorbates (CO2, COOH, H2O, etc.) can dissociate when the
optimizer moves atoms that are free to drift apart.

Two modes for bond detection:
  1. **From MolecularBlock** (preferred): Use the explicit bond list,
     spring_constant, and threshold_scale stored on each MolecularBlock.
     Avoids heuristic distance-based detection entirely.
  2. **Auto-detect**: Fall back to covalent-radii distance check when no
     explicit bonds are available (legacy behaviour).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms, Hookean

from gs_gmlip.structure.bonds import BondData

logger = logging.getLogger(__name__)


@dataclass
class BlockBondInfo:
    """Per-tag bond constraint metadata carried from a MolecularBlock."""

    bonds: list[tuple[int, int]]
    spring_constant: float = 15.0
    threshold_scale: float = 1.3


def get_adsorbate_bond_constraints(
    atoms: Atoms,
    spring_constant: float = 15.0,
    threshold_scale: float = 1.3,
    bond_data: BondData | None = None,
    block_bonds: dict[int, BlockBondInfo | list[tuple[int, int]]] | None = None,
) -> list[Hookean]:
    """Build Hookean constraints for all intramolecular bonds in adsorbates.

    Parameters
    ----------
    atoms : Atoms
        Structure with tagged adsorbates (tag > 0 for molecules).
    spring_constant : float
        Default spring constant k in eV/Å² (used for auto-detect mode).
    threshold_scale : float
        Default bond-length multiplier (used for auto-detect mode).
    bond_data : BondData, optional
        Bond detection parameters (used only in auto-detect mode).
    block_bonds : dict, optional
        Mapping ``{tag: BlockBondInfo}`` or ``{tag: [(i, j), ...]}`` for
        backwards compatibility.  When a ``BlockBondInfo`` is provided the
        per-block ``spring_constant`` and ``threshold_scale`` are used;
        otherwise the function-level defaults apply.

    Returns
    -------
    constraints : list of Hookean
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
            continue

        # --- Mode 1: explicit bonds from MolecularBlock ---
        if block_bonds and tag in block_bonds:
            info = block_bonds[tag]
            if isinstance(info, BlockBondInfo):
                bond_pairs = info.bonds
                k = info.spring_constant
                ts = info.threshold_scale
            else:
                # Legacy: plain list of tuples
                bond_pairs = info
                k = spring_constant
                ts = threshold_scale

            for local_a, local_b in bond_pairs:
                if local_a >= len(indices) or local_b >= len(indices):
                    continue
                idx_a = indices[local_a]
                idx_b = indices[local_b]
                dist = atoms.get_distance(idx_a, idx_b, mic=True)
                rt = dist * ts
                constraints.append(Hookean(a1=idx_a, a2=idx_b, k=k, rt=rt))
                logger.debug(
                    "Hookean (explicit): tag=%d %s(%d)-%s(%d) d=%.3f rt=%.3f k=%.1f",
                    tag,
                    symbols[idx_a],
                    idx_a,
                    symbols[idx_b],
                    idx_b,
                    dist,
                    rt,
                    k,
                )
            continue

        # --- Mode 2: auto-detect from covalent radii ---
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
                    rt = dist * threshold_scale
                    constraints.append(
                        Hookean(a1=idx_a, a2=idx_b, k=spring_constant, rt=rt)
                    )
                    logger.debug(
                        "Hookean (auto): tag=%d %s(%d)-%s(%d) d=%.3f rt=%.3f k=%.1f",
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
    block_bonds: dict[int, BlockBondInfo | list[tuple[int, int]]] | None = None,
) -> None:
    """Apply FixAtoms + Hookean bond constraints to an atoms object in-place.

    Preserves existing FixAtoms constraints and adds Hookean constraints
    for all intramolecular adsorbate bonds.

    Parameters
    ----------
    atoms : Atoms
        Structure to constrain. Modified in-place.
    spring_constant : float
        Default Hookean spring constant (eV/Å²).
    threshold_scale : float
        Default bond threshold scale factor.
    bond_data : BondData, optional
        Bond detection parameters.
    block_bonds : dict, optional
        Mapping ``{tag: BlockBondInfo}`` or ``{tag: [(i,j), ...]}`` for
        backwards compatibility.
    """
    # Preserve FixAtoms
    fix_atoms = [c for c in atoms.constraints if isinstance(c, FixAtoms)]

    # Build Hookean constraints for adsorbate bonds
    hookean = get_adsorbate_bond_constraints(
        atoms,
        spring_constant=spring_constant,
        threshold_scale=threshold_scale,
        bond_data=bond_data,
        block_bonds=block_bonds,
    )

    atoms.set_constraint(fix_atoms + hookean)
