"""Tests for gs_gmlip.evaluate module."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111
from ase.calculators.emt import EMT

from gs_gmlip.evaluate.base import BaseEvaluator
from gs_gmlip.evaluate.mlip.ase_calc import ASECalculatorEvaluator


class TestASECalculatorEvaluator:
    def test_evaluate(self, cu111_slab, emt_evaluator):
        result = emt_evaluator.evaluate(cu111_slab.copy(), relax=False)
        e = result.get_potential_energy()
        assert isinstance(e, float)
        assert np.isfinite(e)

    def test_relax(self, cu111_slab, emt_evaluator):
        atoms = cu111_slab.copy()
        # Perturb positions slightly
        atoms.positions[4] += [0.1, 0.05, 0.0]
        relaxed = emt_evaluator.evaluate(atoms, relax=True)
        e = relaxed.get_potential_energy()
        assert np.isfinite(e)

    def test_get_calculator(self, emt_evaluator):
        calc = emt_evaluator.get_calculator()
        assert calc is not None

    def test_batch_evaluate(self, cu111_slab, emt_evaluator):
        structures = [cu111_slab.copy() for _ in range(3)]
        results = emt_evaluator.batch_evaluate(structures, relax=False)
        assert len(results) == 3
        for r in results:
            assert np.isfinite(r.get_potential_energy())
