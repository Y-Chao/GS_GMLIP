"""Shared test fixtures for gs_gmlip tests.

Provides lightweight slab, evaluator, and composition fixtures
using EMT for fast testing without MACE dependencies.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111, molecule
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms

from gs_gmlip.evaluate.mlip.ase_calc import ASECalculatorEvaluator
from gs_gmlip.structure.composition import MolecularBlock
from gs_gmlip.structure.region import BoxRegion


@pytest.fixture
def cu111_slab():
    """Small Cu(111) slab for testing."""
    slab = fcc111("Cu", size=(2, 2, 3), vacuum=10.0, periodic=True)
    z = slab.positions[:, 2]
    z_sorted = sorted(set(z.round(2)))
    fix_mask = z <= z_sorted[0] + 0.1
    slab.set_constraint(FixAtoms(mask=fix_mask))
    tags = [0 if fix_mask[i] else 1 for i in range(len(slab))]
    slab.set_tags(tags)
    return slab


@pytest.fixture
def emt_evaluator():
    """ASE EMT evaluator for fast testing."""
    return ASECalculatorEvaluator(calculator=EMT())


@pytest.fixture
def h_block():
    """Hydrogen atom block."""
    h = Atoms("H", positions=[[0, 0, 0]])
    return MolecularBlock(name="H", atoms=h, chemical_potential=-3.0)


@pytest.fixture
def o_block():
    """Oxygen atom block."""
    o = Atoms("O", positions=[[0, 0, 0]])
    return MolecularBlock(name="O", atoms=o, chemical_potential=-4.5)


@pytest.fixture
def simple_blocks(h_block, o_block):
    """List of simple single-atom blocks."""
    return [h_block, o_block]


@pytest.fixture
def box_region(cu111_slab):
    """Box region above the Cu slab."""
    z_top = cu111_slab.positions[:, 2].max()
    return BoxRegion(
        cell=cu111_slab.cell,
        z_min=z_top + 1.0,
        z_max=z_top + 5.0,
        pbc=cu111_slab.pbc,
    )


@pytest.fixture
def rng():
    """Deterministic random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def slab_with_adsorbate(cu111_slab, rng):
    """Cu slab with a few H atoms adsorbed."""
    atoms = cu111_slab.copy()
    z_top = atoms.positions[:, 2].max()
    for i in range(3):
        pos = atoms.positions[i].copy()
        pos[2] = z_top + 1.8 + rng.uniform(-0.2, 0.2)
        h = Atoms("H", positions=[pos])
        h.set_tags([2 + i])
        atoms.extend(h)
    return atoms
