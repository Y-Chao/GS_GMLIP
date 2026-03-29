"""Tests for gs_gmlip.search.ga module."""

import numpy as np
import pytest
from ase import Atoms

from gs_gmlip.search.ga.comparators import (
    EnergyComparator,
    InteratomicDistanceComparator,
    SequentialComparator,
)
from gs_gmlip.search.ga.operators import (
    CutAndSplicePairing,
    OperationSelector,
    RattleMutation,
)
from gs_gmlip.search.ga.startgenerator import StartGenerator
from gs_gmlip.structure.bonds import generate_blmin


class TestStartGenerator:
    def test_generate_candidate(self, cu111_slab, simple_blocks, box_region):
        blmin = generate_blmin(["Cu", "H", "O"])
        gen = StartGenerator(
            slab=cu111_slab,
            blocks=simple_blocks,
            counts=[1, 1],
            region=box_region,
            blmin=blmin,
        )
        candidate = gen.get_new_candidate()
        assert candidate is not None
        assert len(candidate) > len(cu111_slab)

    def test_generate_multiple(self, cu111_slab, simple_blocks, box_region):
        blmin = generate_blmin(["Cu", "H", "O"])
        gen = StartGenerator(
            slab=cu111_slab,
            blocks=simple_blocks,
            counts=[1, 1],
            region=box_region,
            blmin=blmin,
        )
        candidates = [gen.get_new_candidate() for _ in range(5)]
        # At least some should succeed
        valid = [c for c in candidates if c is not None]
        assert len(valid) >= 1


class TestComparators:
    def test_energy_comparator(self, cu111_slab):
        comp = EnergyComparator(dE=0.02)
        a1 = cu111_slab.copy()
        a2 = cu111_slab.copy()
        a1.info["key_value_pairs"] = {"raw_score": -10.0}
        a2.info["key_value_pairs"] = {"raw_score": -10.005}
        assert comp.looks_like(a1, a2)

        a2.info["key_value_pairs"] = {"raw_score": -11.0}
        assert not comp.looks_like(a1, a2)

    def test_sequential_comparator(self, cu111_slab):
        c1 = EnergyComparator(dE=0.02)
        seq = SequentialComparator([c1], logic="AND")
        a1 = cu111_slab.copy()
        a2 = cu111_slab.copy()
        a1.info["key_value_pairs"] = {"raw_score": -10.0}
        a2.info["key_value_pairs"] = {"raw_score": -10.0}
        assert seq.looks_like(a1, a2)


class TestOperators:
    def test_operation_selector(self):
        class DummyOp:
            def __init__(self, name):
                self.descriptor = name

            def get_new_individual(self, parents):
                return parents[0] if parents else None

        ops = [DummyOp("op1"), DummyOp("op2")]
        selector = OperationSelector([0.5, 0.5], ops)
        op = selector.get_operator()
        assert op.descriptor in ("op1", "op2")
