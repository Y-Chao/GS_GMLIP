"""Structure module: representation and manipulation of atomic structures."""

from gs_gmlip.structure.atoms import SlabAtoms
from gs_gmlip.structure.bonds import BondData, get_bond_matrix
from gs_gmlip.structure.composition import (
    CompositionConstraint,
    MolecularBlock,
)
from gs_gmlip.structure.constraints import (
    apply_constraints,
    get_adsorbate_bond_constraints,
)
from gs_gmlip.structure.region import BoxRegion, Region, SphereRegion

__all__ = [
    "SlabAtoms",
    "Region",
    "BoxRegion",
    "SphereRegion",
    "BondData",
    "get_bond_matrix",
    "CompositionConstraint",
    "MolecularBlock",
    "apply_constraints",
    "get_adsorbate_bond_constraints",
]
