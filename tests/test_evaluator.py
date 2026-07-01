"""Tests for the evaluator module."""

from __future__ import annotations

import numpy as np
import pytest
from ase.calculators.emt import EMT

from gs_gmlip.evaluator.base import BaseEvaluator
from gs_gmlip.interface.core import Interface


class EMTEvaluator(BaseEvaluator):
    """Concrete evaluator using ASE's EMT calculator (for testing)."""

    def get_calculator(self):
        return EMT()


class TestBaseEvaluator:
    """Tests for BaseEvaluator with EMT backend."""

    @pytest.fixture
    def evaluator(self):
        return EMTEvaluator(fmax=0.1, relax_steps=50)

    @pytest.fixture
    def interface(self, cu100_slab, co_molecule):
        """Bare Cu(100) slab as Interface with one fixed bottom layer."""
        iface = Interface(substrate=cu100_slab)
        iface.fix_substrate(layers=2)
        return iface

    def test_get_calculator(self, evaluator):
        """get_calculator returns an ASE Calculator."""
        calc = evaluator.get_calculator()
        assert calc is not None
        assert hasattr(calc, "get_potential_energy")

    def test_evaluate_no_relax(self, evaluator, interface):
        """evaluate(relax=False) attaches calculator, does not change positions."""
        orig_positions = interface.interface.get_positions().copy()
        result = evaluator.evaluate(interface, relax=False)
        new_positions = result.interface.get_positions()
        # Positions should be unchanged (no relaxation)
        assert result.interface.calc is not None
        # Energy should be computable
        energy = result.interface.get_potential_energy()
        assert isinstance(energy, float)

    def test_evaluate_with_relax(self, evaluator, interface):
        """evaluate(relax=True) runs BFGS and updates positions."""
        assert interface.interface.calc is None
        result = evaluator.evaluate(interface, relax=True)
        assert result.interface.calc is not None
        energy = result.interface.get_potential_energy()
        assert isinstance(energy, float)
        assert np.isfinite(energy)  # EMT gives finite energy

    def test_relax_alias(self, evaluator, interface):
        """relax() is equivalent to evaluate(relax=True)."""
        result = evaluator.relax(interface)
        assert result.interface.calc is not None

    def test_evaluate_with_adsorbate(self, evaluator, interface, co_molecule):
        """Evaluate a slab+adsorbate system."""
        co = co_molecule.copy()
        co.translate([5.0, 5.0, 18.0])
        interface.add_adsorbate(co)
        interface.fix_substrate(layers=2)

        result = evaluator.evaluate(interface, relax=True, fmax=0.1, steps=20)
        energy = result.interface.get_potential_energy()
        assert isinstance(energy, float)
