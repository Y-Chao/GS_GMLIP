"""Tests for the calculator registry and registry-backed evaluator."""

from __future__ import annotations

import pytest
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator

from gs_gmlip.evaluator import (
    CalculatorEvaluator,
    calculator_entries,
    get_calculator,
    register_calculator,
)
from gs_gmlip.interface import operations as ops


def test_builtin_entries():
    entries = calculator_entries()
    assert {"emt", "matgl", "mace", "vasp"} <= set(entries)
    assert entries["emt"].available
    assert entries["matgl"].defaults["model"].startswith("TensorNet-MatPES-r2SCAN")


def test_get_calculator_and_errors():
    assert isinstance(get_calculator("emt"), EMT)
    with pytest.raises(KeyError, match="unknown calculator"):
        get_calculator("does-not-exist")


def test_register_custom_and_missing_dependency():
    seen = {}

    def factory(**params):
        seen.update(params)
        return EMT()

    register_calculator("custom-emt", factory, defaults={"alpha": 1, "beta": 2})
    get_calculator("custom-emt", beta=3)
    assert seen == {"alpha": 1, "beta": 3}

    register_calculator("ghost", factory, requires="definitely_not_a_module_xyz")
    assert not calculator_entries()["ghost"].available
    with pytest.raises(ImportError):
        get_calculator("ghost")


def test_evaluator_reuses_calculator_and_freezes_results():
    slab = ops.build_slab(
        ops.make_bulk("Cu", "fcc", 3.61), (1, 1, 1), layers=3, fix_layers=1
    )
    evaluator = CalculatorEvaluator("emt", fmax=0.5, relax_steps=3)
    assert evaluator.get_calculator() is evaluator.get_calculator()
    first = evaluator.evaluate(slab.copy(), relax=True)
    second = evaluator.evaluate(slab.copy())
    assert isinstance(first.interface.calc, SinglePointCalculator)
    assert isinstance(second.interface.calc, SinglePointCalculator)
    # Evaluating a second structure must not disturb the first one's results.
    assert (
        first.interface.get_potential_energy()
        <= second.interface.get_potential_energy()
    )
