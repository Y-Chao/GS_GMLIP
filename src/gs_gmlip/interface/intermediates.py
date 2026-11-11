"""Seed set of reaction intermediates for electrocatalysis.

Covers the usual intermediates of hydrogen evolution (HER), oxygen evolution
and reduction (OER, ORR), and CO2/CO reduction (CO2RR, CORR) to C1 and C2
products.

Provenance, stated plainly: these are **initial-guess gas-phase geometries**,
not adsorbed structures taken from the literature. Each one is an ASE G2
(MP2-optimized) molecule, or is derived from a G2 parent by removing hydrogen
atoms, then oriented with its surface-binding atom lowest. Relax every
intermediate on the surface before using its energy. Curated, cited geometries
(e.g. from Catalysis-Hub) can be added to the global library on top of this
set; a user entry with the same name takes precedence.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from ase import Atoms
from ase.build import molecule as ase_molecule
from pymatgen.core import Molecule

from gs_gmlip.interface.operations import orient_adsorbate

REACTIONS: tuple[str, ...] = ("HER", "OER", "ORR", "CO2RR", "CORR")

# An atom of the parent molecule, addressed as (element symbol, ordinal among that element).
AtomRef = tuple[str, int]
# Longest X-H distance (Å) counted as a bond when removing hydrogens.
_XH_BOND = 1.25


@dataclass(frozen=True)
class Intermediate:
    """Definition of one seed intermediate.

    Attributes:
        name: Library name (adsorbed species are written without the ``*``).
        reactions: Reactions the species takes part in, from ``REACTIONS``.
        parent: ASE G2 molecule the geometry starts from.
        bind: Surface-binding atom of the parent.
        remove: Hydrogens to strip, as ``(heavy atom, number of H bonded to it)``.
        description: Short chemical description.
    """

    name: str
    reactions: tuple[str, ...]
    parent: str
    bind: AtomRef
    remove: tuple[tuple[AtomRef, int], ...] = ()
    description: str = ""

    @property
    def note(self) -> str:
        """Provenance of the geometry."""
        if not self.remove:
            return (
                f"ASE G2 geometry of {self.parent}; initial guess, relax on the surface"
            )
        stripped = sum(count for _, count in self.remove)
        return (
            f"derived from ASE G2 {self.parent} by removing {stripped} H; "
            "initial guess, relax on the surface"
        )


_C1 = ("CO2RR", "CORR")
_O = ("OER", "ORR")

INTERMEDIATES: tuple[Intermediate, ...] = (
    # Hydrogen and oxygen electrochemistry
    Intermediate(
        "H", ("HER", "CO2RR", "CORR"), "H", ("H", 0), description="adsorbed hydrogen"
    ),
    Intermediate("H2", ("HER",), "H2", ("H", 0), description="hydrogen (product)"),
    Intermediate("O", _O, "O", ("O", 0), description="adsorbed oxygen"),
    Intermediate("OH", _O, "OH", ("O", 0), description="hydroxyl"),
    Intermediate(
        "OOH",
        _O,
        "H2O2",
        ("O", 1),
        ((("O", 1), 1),),
        "hydroperoxyl, bound through the bare O",
    ),
    Intermediate("O2", _O, "O2", ("O", 0), description="dioxygen, end-on"),
    Intermediate(
        "H2O2",
        ("ORR",),
        "H2O2",
        ("O", 0),
        description="hydrogen peroxide (2e- ORR product)",
    ),
    Intermediate("H2O", ("HER", "OER", "ORR"), "H2O", ("O", 0), description="water"),
    # CO2 / CO reduction, C1
    Intermediate(
        "CO2",
        ("CO2RR",),
        "CO2",
        ("C", 0),
        description="carbon dioxide, physisorbed flat",
    ),
    Intermediate(
        "COOH", ("CO2RR",), "HCOOH", ("C", 0), ((("C", 0), 1),), "carboxyl, C-bound"
    ),
    Intermediate(
        "OCHO", ("CO2RR",), "HCOOH", ("O", 0), ((("O", 0), 1),), "formate, O-bound"
    ),
    Intermediate(
        "HCOOH", ("CO2RR",), "HCOOH", ("O", 1), description="formic acid (product)"
    ),
    Intermediate("CO", _C1, "CO", ("C", 0), description="carbon monoxide, C-bound"),
    Intermediate("CHO", _C1, "HCO", ("C", 0), description="formyl, C-bound"),
    Intermediate(
        "COH", _C1, "CH3OH", ("C", 0), ((("C", 0), 3),), "hydroxymethylidyne, C-bound"
    ),
    Intermediate(
        "CHOH", _C1, "CH3OH", ("C", 0), ((("C", 0), 2),), "hydroxymethylene, C-bound"
    ),
    Intermediate("CH2O", _C1, "H2CO", ("C", 0), description="formaldehyde, C-bound"),
    Intermediate("CH2OH", _C1, "H2COH", ("C", 0), description="hydroxymethyl, C-bound"),
    Intermediate("CH3O", _C1, "CH3O", ("O", 0), description="methoxy, O-bound"),
    Intermediate("CH3OH", _C1, "CH3OH", ("O", 0), description="methanol (product)"),
    Intermediate("C", _C1, "C", ("C", 0), description="adsorbed carbon"),
    Intermediate("CH", _C1, "CH", ("C", 0), description="methylidyne"),
    Intermediate("CH2", _C1, "CH2_s3B1d", ("C", 0), description="methylene"),
    Intermediate("CH3", _C1, "CH3", ("C", 0), description="methyl"),
    Intermediate("CH4", _C1, "CH4", ("C", 0), description="methane (product)"),
    # CO2 / CO reduction, C2
    Intermediate(
        "OCCO", _C1, "OCHCHO", ("C", 0), ((("C", 0), 1), (("C", 1), 1)), "CO dimer"
    ),
    Intermediate(
        "OCCHO", _C1, "OCHCHO", ("C", 0), ((("C", 0), 1),), "glyoxyl, C-bound"
    ),
    Intermediate(
        "CCO",
        _C1,
        "H2CCO",
        ("C", 0),
        ((("C", 0), 2),),
        "ketenylidene, bound through the terminal C",
    ),
    Intermediate("CHCO", _C1, "H2CCO", ("C", 0), ((("C", 0), 1),), "ketenyl"),
    Intermediate("CH2CO", _C1, "H2CCO", ("C", 0), description="ketene"),
    Intermediate(
        "CH3CO",
        _C1,
        "CH3CO",
        ("C", 1),
        description="acetyl, bound through the carbonyl C",
    ),
    Intermediate(
        "CH3CHO", _C1, "CH3CHO", ("O", 0), description="acetaldehyde (product)"
    ),
    Intermediate("CH3CH2O", _C1, "CH3CH2O", ("O", 0), description="ethoxy, O-bound"),
    Intermediate(
        "CH3CH2OH", _C1, "CH3CH2OH", ("O", 0), description="ethanol (product)"
    ),
    Intermediate("CCH", _C1, "CCH", ("C", 1), description="ethynyl"),
    Intermediate("C2H2", _C1, "C2H2", ("C", 0), description="acetylene"),
    Intermediate("C2H3", _C1, "C2H3", ("C", 0), description="vinyl"),
    Intermediate("C2H4", _C1, "C2H4", ("C", 0), description="ethylene (product)"),
    Intermediate(
        "C2H5", _C1, "C2H5", ("C", 1), description="ethyl, bound through the CH2 carbon"
    ),
)

INTERMEDIATE_NAMES: tuple[str, ...] = tuple(item.name for item in INTERMEDIATES)


def _resolve(atoms: Atoms, ref: AtomRef) -> int:
    symbol, ordinal = ref
    matches = [atom.index for atom in atoms if atom.symbol == symbol]
    if ordinal >= len(matches):
        raise ValueError(
            f"{atoms.get_chemical_formula()} has no {symbol} atom number {ordinal}"
        )
    return matches[ordinal]


def build_intermediate(item: Intermediate) -> Molecule:
    """Build the oriented geometry of one seed intermediate.

    Raises:
        ValueError: If the parent does not have the hydrogens the definition removes.
    """
    atoms = ase_molecule(item.parent)
    bind = _resolve(atoms, item.bind)
    drop: list[int] = []
    for ref, count in item.remove:
        heavy = _resolve(atoms, ref)
        distances = atoms.get_distances(heavy, range(len(atoms)))
        bonded = [
            i
            for i in np.argsort(distances)
            if atoms[i].symbol == "H" and distances[i] < _XH_BOND and i not in drop
        ]
        if len(bonded) < count:
            raise ValueError(
                f"{item.name}: {item.parent} has only {len(bonded)} H on {ref}, cannot remove {count}"
            )
        drop.extend(int(i) for i in bonded[:count])
    keep = [i for i in range(len(atoms)) if i not in drop]
    return orient_adsorbate(atoms[keep], keep.index(bind))


@lru_cache(maxsize=1)
def _built() -> dict[str, Molecule]:
    return {item.name: build_intermediate(item) for item in INTERMEDIATES}


def get_intermediate(name: str) -> Molecule:
    """Oriented geometry of a seed intermediate (binding atom at the origin).

    Raises:
        KeyError: If the name is not in the seed set.
    """
    built = _built()
    if name not in built:
        raise KeyError(f"unknown intermediate {name!r}")
    return built[name].copy()
