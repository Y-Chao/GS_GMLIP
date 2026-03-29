"""Tests for gs_gmlip.search.gofee module."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111

from gs_gmlip.search.gofee.acquisition import (
    expected_improvement,
    lower_confidence_bound,
    probability_of_improvement,
)
from gs_gmlip.search.gofee.surrogate import SurrogateModel, get_fingerprint


class TestFingerprint:
    def test_fingerprint_shape(self, cu111_slab):
        fp = get_fingerprint(cu111_slab)
        assert fp.shape == (64,)

    def test_fingerprint_nonzero(self, cu111_slab):
        fp = get_fingerprint(cu111_slab)
        assert fp.sum() > 0

    def test_empty_active(self):
        fp = get_fingerprint(Atoms(), active_indices=[])
        assert fp.shape == (64,)
        assert fp.sum() == 0.0


class TestSurrogateModel:
    def test_update_and_predict(self, cu111_slab):
        model = SurrogateModel(kernel="rbf")
        # Create some training data
        structures = []
        energies = []
        rng = np.random.default_rng(42)
        for i in range(10):
            atoms = cu111_slab.copy()
            atoms.positions += rng.normal(0, 0.01, atoms.positions.shape)
            structures.append(atoms)
            energies.append(-10.0 + rng.normal(0, 0.1))

        model.update(structures, energies)

        # Predict
        mean, std = model.predict(structures[:3])
        assert mean.shape == (3,)
        assert std.shape == (3,)
        assert all(np.isfinite(mean))
        assert all(std >= 0)

    def test_predict_single(self, cu111_slab):
        model = SurrogateModel()
        structures = [cu111_slab.copy() for _ in range(5)]
        energies = list(range(5))
        model.update(structures, energies)
        mean, std = model.predict_single(cu111_slab)
        assert np.isfinite(mean)
        assert std >= 0


class TestAcquisition:
    def test_lcb(self):
        mean = np.array([-10.0, -8.0, -12.0])
        std = np.array([1.0, 2.0, 0.5])
        lcb = lower_confidence_bound(mean, std, kappa=1.0)
        assert lcb.shape == (3,)
        assert lcb[2] < lcb[0]  # Lower mean, lower std → lower LCB

    def test_ei(self):
        mean = np.array([-10.0, -8.0, -12.0])
        std = np.array([1.0, 2.0, 0.5])
        ei = expected_improvement(mean, std, best_energy=-11.0)
        assert ei.shape == (3,)
        assert all(ei >= 0)

    def test_pi(self):
        mean = np.array([-10.0, -8.0, -12.0])
        std = np.array([1.0, 2.0, 0.5])
        pi = probability_of_improvement(mean, std, best_energy=-11.0)
        assert pi.shape == (3,)
        assert all(pi >= 0)
        assert all(pi <= 1)
