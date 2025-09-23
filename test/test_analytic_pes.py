#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import annotations

__author__ = "Chao Yang"
__version__ = "1.0"


"""
Test for the analytic PES functions and their derivatives.
"""

import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, hessian

    jax_available = True
except ImportError:
    import numpy as np

    jax_available = False

from analytic_pes import AnalyticPES


class TestAnalyticPES:
    @pytest.mark.parametrize(
        "x, y, expected_z",
        [(-0.568, 1.432, 0.005898), (-0.85, 1.05, 6.512918), (0.0, 0.0, 14.7448)],
    )
    def test_muller_brown(self, x, y, expected_z):
        pes = AnalyticPES("muller_brown")
        v = pes.get_potential(x, y)
        assert np.isclose(v, expected_z, atol=1e-5)

        if jax_available:
            grad = pes.get_gradient(x, y)
            assert grad.shape == (2,)

            hess = pes.get_hessian(x, y)
            assert hess.shape == (2, 2)

        else:
            grad = pes.get_gradient(x, y)
            assert isinstance(grad, np.ndarray)
            assert grad.shape == (2,)

            hess = pes.get_hessian(x, y)
            assert isinstance(hess, np.ndarray)
            assert hess.shape == (2, 2)
