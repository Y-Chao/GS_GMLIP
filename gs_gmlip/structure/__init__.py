"""Structure module: representation and manipulation of atomic structures."""

from gs_gmlip.structure.atoms import SlabAtoms, find_molecules
from gs_gmlip.structure.bonds import BondData, get_bond_matrix
from gs_gmlip.structure.composition import (
    CompositionConstraint,
    MolecularBlock,
    cooh_block,
    get_block,
    get_block_registry,
    hcoo_block,
    register_block,
)
from gs_gmlip.structure.constraints import (
    BlockBondInfo,
    apply_constraints,
    get_adsorbate_bond_constraints,
)
from gs_gmlip.structure.region import BoxRegion, Region, SphereRegion

__all__ = [
    "SlabAtoms",
    "find_molecules",
    "Region",
    "BoxRegion",
    "SphereRegion",
    "BondData",
    "get_bond_matrix",
    "CompositionConstraint",
    "MolecularBlock",
    "register_block",
    "get_block",
    "get_block_registry",
    "apply_constraints",
    "get_adsorbate_bond_constraints",
    "BlockBondInfo",
]
