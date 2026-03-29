"""Tests for gs_gmlip.analysis module."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111

from gs_gmlip.analysis.structure import (
    StructureAnalyzer,
    fingerprint_distance,
    pair_distribution,
)
from gs_gmlip.analysis.trajectory import TrajectoryAnalyzer


class TestPairDistribution:
    def test_shape(self, cu111_slab):
        r, g = pair_distribution(cu111_slab, active_only=False)
        assert r.shape == g.shape
        assert len(r) == 100


class TestFingerprintDistance:
    def test_self_distance(self, cu111_slab):
        d = fingerprint_distance(cu111_slab, cu111_slab)
        assert d < 1e-10

    def test_different_structures(self, cu111_slab):
        atoms2 = cu111_slab.copy()
        atoms2.positions += np.random.default_rng(0).normal(
            0, 0.5, atoms2.positions.shape
        )
        d = fingerprint_distance(cu111_slab, atoms2)
        assert d > 0


class TestStructureAnalyzer:
    def test_energy_summary(self, cu111_slab):
        structures = [cu111_slab.copy() for _ in range(5)]
        energies = [-10.0, -9.5, -11.0, -10.2, -9.8]
        analyzer = StructureAnalyzer(structures, energies)
        summary = analyzer.energy_landscape_summary()
        assert summary["energy_min"] == -11.0
        assert summary["n_structures"] == 5

    def test_deduplicate(self, cu111_slab):
        structures = [cu111_slab.copy() for _ in range(5)]
        energies = [0.0] * 5
        analyzer = StructureAnalyzer(structures, energies)
        unique = analyzer.deduplicate(threshold=0.01)
        assert len(unique) == 1  # All identical


class TestTrajectoryAnalyzer:
    def test_summary(self):
        trajectory = [
            {
                "step": i,
                "energy": -10.0 + np.sin(i / 10),
                "best_energy": -10.0 - i * 0.01,
                "accepted": i % 2 == 0,
            }
            for i in range(100)
        ]
        analyzer = TrajectoryAnalyzer(trajectory)
        summary = analyzer.summary()
        assert summary["n_steps"] == 100
        assert "acceptance_rate" in summary

    def test_convergence(self):
        trajectory = [{"step": i, "best_energy": -10.0 - 0.001 * i} for i in range(100)]
        analyzer = TrajectoryAnalyzer(trajectory)
        result = analyzer.convergence_check(window=50, threshold=0.1)
        assert isinstance(result["converged"], bool)

    def test_energy_distribution(self):
        trajectory = [
            {"energy": -10.0 + np.random.default_rng(i).normal()} for i in range(50)
        ]
        analyzer = TrajectoryAnalyzer(trajectory)
        centers, counts = analyzer.energy_distribution(n_bins=20)
        assert len(centers) == 20
        assert counts.sum() == 50
