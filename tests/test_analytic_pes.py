"""Tests for analytic_pes module."""
from __future__ import annotations

import numpy as np
import pytest

from gs_gmlip.evaluator.analytic_pes import AnalyticPES

PES_NAMES = [
    "muller_brown",
    "muller_brown_three_stats",
    "leps",
    "leps_harmonic",
    "wolfe_quapp",
    "wolfe_quapp_local_soft",
]


class TestAnalyticPESNumPy:
    """Tests using the NumPy backend (no JAX required)."""

    @pytest.mark.parametrize("name", PES_NAMES)
    def test_construction(self, name):
        """Each registered PES name constructs without error."""
        pes = AnalyticPES(name, use_jax=False)
        assert pes.function_name == name
        assert pes.xp is np

    @pytest.mark.parametrize("name", PES_NAMES)
    def test_energy_scalar(self, name):
        """Energy for a scalar (x, y) returns a finite float or array element."""
        pes = AnalyticPES(name, use_jax=False)
        val = pes.get_potential_energy(0.0, 0.0)
        assert np.isfinite(float(val))

    @pytest.mark.parametrize("name", PES_NAMES)
    def test_energy_array(self, name):
        """Energy for array inputs returns same-shape output."""
        pes = AnalyticPES(name, use_jax=False)
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        z = pes.get_potential_energy(x, y)
        assert z.shape == (10,)

    @pytest.mark.parametrize("name", PES_NAMES)
    def test_forces_shape(self, name):
        """Forces for scalar input return a (2,) array."""
        pes = AnalyticPES(name, use_jax=False)
        f = pes.get_forces(0.0, 0.0)
        assert f.shape == (2,)
        assert np.all(np.isfinite(f))

    @pytest.mark.parametrize("name", PES_NAMES)
    def test_hessian_shape(self, name):
        """Hessian for scalar input returns a (2, 2) array."""
        pes = AnalyticPES(name, use_jax=False)
        h = pes.get_hessian(0.0, 0.0)
        assert h.shape == (2, 2)
        assert np.all(np.isfinite(h))

    @pytest.mark.parametrize("name", PES_NAMES)
    def test_pes_matrix_shape(self, name):
        """pes_matrix returns X, Y, Z with consistent shapes."""
        pes = AnalyticPES(name, use_jax=False)
        X, Y, Z = pes.pes_matrix(num_points=20)
        assert X.shape == (20, 20)
        assert Y.shape == (20, 20)
        assert Z.shape == (20, 20)
        assert np.all(np.isfinite(Z))

    @pytest.mark.parametrize("name", PES_NAMES)
    def test_default_ranges(self, name):
        """Each PES has default range and z_max attributes."""
        pes = AnalyticPES(name, use_jax=False)
        assert hasattr(pes, "range")
        assert hasattr(pes, "z_max")
        assert len(pes.range) == 2
        assert len(pes.range[0]) == 2
        assert len(pes.range[1]) == 2

    def test_muller_brown_known_minima(self):
        """Muller-Brown has lower energy at the global minimum than at a local minimum."""
        pes = AnalyticPES("muller_brown", use_jax=False)
        # Near (-0.56, 1.44) — the global minimum
        e_global = float(pes.get_potential_energy(-0.56, 1.44))
        # Near (-0.05, 0.47) — a local minimum
        e_local = float(pes.get_potential_energy(-0.05, 0.47))
        # Known: global min < local min for Muller-Brown
        assert e_global < e_local
        # Both should be finite
        assert np.isfinite(e_global)
        assert np.isfinite(e_local)

    def test_unknown_name_raises(self):
        """Constructing with an unregistered name raises ValueError."""
        with pytest.raises(ValueError):
            AnalyticPES("nonexistent_pes", use_jax=False)


class TestAnalyticPESJAX:
    """Tests requiring JAX. Skipped if JAX is not installed."""

    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    @pytest.mark.parametrize("name", PES_NAMES)
    def test_construction_jax(self, name):
        """JAX backend sets xp to jnp."""
        pes = AnalyticPES(name, use_jax=True)
        import jax.numpy as jnp
        assert pes.xp is jnp

    @pytest.mark.parametrize("name", PES_NAMES)
    def test_jax_numpy_consistency_energy(self, name):
        """JAX and NumPy backends give the same energy at a test point."""
        pes_np = AnalyticPES(name, use_jax=False)
        pes_jax = AnalyticPES(name, use_jax=True)
        x, y = 0.3, 0.7
        e_np = float(pes_np.get_potential_energy(x, y))
        e_jax = float(pes_jax.get_potential_energy(x, y))
        assert np.isclose(e_np, e_jax, rtol=1e-5)

    @pytest.mark.parametrize("name", PES_NAMES)
    def test_jax_numpy_consistency_forces(self, name):
        """JAX and NumPy gradients agree at a test point."""
        pes_np = AnalyticPES(name, use_jax=False)
        pes_jax = AnalyticPES(name, use_jax=True)
        x, y = 0.3, 0.7
        f_np = np.asarray(pes_np.get_forces(x, y), dtype=float)
        f_jax = np.asarray(pes_jax.get_forces(x, y), dtype=float)
        assert np.allclose(f_np, f_jax, rtol=1e-3, atol=1e-3)
