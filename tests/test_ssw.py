"""Tests for gs_gmlip.search.ssw module."""

import numpy as np
import pytest
from ase import Atoms

from gs_gmlip.search.ssw.minima_hopping import MinimaHopping
from gs_gmlip.search.ssw.walker import get_soft_modes


class TestMinimaHopping:
    def test_accept_lower(self):
        mh = MinimaHopping(temperature=300.0)
        rng = np.random.default_rng(42)
        assert mh.accept(-10.0, -11.0, rng)
        assert mh.n_accepted == 1

    def test_adaptive_temperature(self):
        mh = MinimaHopping(temperature=500.0, beta_decrease=0.9, beta_increase=1.1)
        rng = np.random.default_rng(42)
        # Accept → temperature decreases
        mh.accept(-10.0, -11.0, rng)
        assert mh.temperature < 500.0

    def test_is_new_minimum(self):
        mh = MinimaHopping()
        mh.visited_energies = [-10.0, -8.0, -12.0]
        assert mh.is_new_minimum(-9.0)
        assert not mh.is_new_minimum(-10.005, tolerance=0.01)

    def test_acceptance_rate(self):
        mh = MinimaHopping(temperature=1000.0)
        rng = np.random.default_rng(0)
        for _ in range(10):
            mh.accept(0.0, -0.01, rng)
        assert mh.acceptance_rate > 0


class TestSoftModes:
    def test_get_soft_modes(self):
        # Random symmetric positive semidefinite matrix
        rng = np.random.default_rng(42)
        A = rng.normal(0, 1, (9, 9))
        H = A @ A.T  # Positive semidefinite

        eigenvalues, eigenvectors = get_soft_modes(H, n_modes=3)
        assert eigenvalues.shape == (3,)
        assert eigenvectors.shape == (3, 9)
        # Eigenvalues should be sorted
        assert eigenvalues[0] <= eigenvalues[1] <= eigenvalues[2]
