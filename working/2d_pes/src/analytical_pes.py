#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import annotations

__author__ = "Chao Yang"
__version__ = "1.0"

# Backend check
try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

import numpy as np


def muller_brown_potential(data, xp):
    """
    Calculate the Muller-Brown potential at coordinates (x, y).
    This is a 2D potential function commonly used in optimization problems.

    ::math::
        V(x, y) = \\sum_{i=1}^{4} A_i \\exp \\left[ a_i (x - x_{0i})^2 + b_i (x - x_{0i})(y - y_{0i}) + c_i (y - y_{0i})^2 \\right] + offset
    """
    x = data[0]
    y = data[1]

    prefactor = 0.15
    A = xp.array([-200, -100, -170, 15])
    a = xp.array([-1, -1, -6.5, 0.7])
    b = xp.array([0, 0, 11, 0.6])
    c = xp.array([-10, -10, -6.5, 0.7])
    x0 = xp.array([1, 0, -0.5, -1])
    y0 = xp.array([0, 0.5, 1.5, 1])
    # offset = -146.7

    # v = -prefactor * offset
    v = 0.0
    for i in range(4):
        v += (
            prefactor
            * A[i]
            * xp.exp(
                a[i] * (x - x0[i]) ** 2
                + b[i] * (x - x0[i]) * (y - y0[i])
                + c[i] * (y - y0[i]) ** 2
            )
        )
    return v


def grad_muller_brown_potential_analytic(data):
    """
    data: shape (2)
    """

    x = data[0]
    y = data[1]
    prefactor = 0.15
    A = np.array([-200, -100, -170, 15])
    a = np.array([-1, -1, -6.5, 0.7])
    b = np.array([0, 0, 11, 0.6])
    c = np.array([-10, -10, -6.5, 0.7])
    x0 = np.array([1, 0, -0.5, -1])
    y0 = np.array([0, 0.5, 1.5, 1])
    # offset = -146.7

    # dvdx = -prefactor * offset
    # dvdy = -prefactor * offset
    dvdx = 0.0
    dvdy = 0.0
    for i in range(4):
        dvdx += (
            prefactor
            * A[i]
            * (2 * a[i] * (x - x0[i]) + b[i] * (y - y0[i]))
            * np.exp(
                a[i] * (x - x0[i]) ** 2
                + b[i] * (x - x0[i]) * (y - y0[i])
                + c[i] * (y - y0[i]) ** 2
            )
        )

        dvdy += (
            prefactor
            * A[i]
            * (b[i] * (x - x0[i]) + 2 * c[i] * (y - y0[i]))
            * np.exp(
                a[i] * (x - x0[i]) ** 2
                + b[i] * (x - x0[i]) * (y - y0[i])
                + c[i] * (y - y0[i]) ** 2
            )
        )
    return np.vstack((dvdx, dvdy)).T


def grad_muller_brown_potetnial(data, backend="auto"):
    """
    data: shape (2)
    backend: "auto", "jax", "numpy"
    """
    if backend == "jax" or (backend == "auto" and _HAS_JAX):
        energy = lambda data: muller_brown_potential(data, jnp)
        grad_energy = jax.grad(energy)
        return grad_energy(jnp.asarray(data))

    return grad_muller_brown_potential_analytic(np.asarray(data))


def muller_brown_potential_three_stats(data, xp):
    """
    Calculate the Muller-Brown potential at coordinates (x, y).
    This is a 2D potential function commonly used in optimization problems.
    """
    x = data[0]
    y = data[1]

    prefactor = 0.15
    A = xp.array([-280, -170, -170, 15])
    a = xp.array([-15, -1, -6.5, 0.7])
    b = xp.array([0, 0, 11, 0.6])
    c = xp.array([-10, -10, -6.5, 0.7])
    x0 = xp.array([1, 0.2, -0.5, -1])
    y0 = xp.array([0, 0.5, 1.5, 1])
    # offset = -146.7

    # v = -prefactor * offset

    v = 0.0
    for i in range(4):
        v += (
            prefactor
            * A[i]
            * xp.exp(
                a[i] * (x - x0[i]) ** 2
                + b[i] * (1 - x0[i]) * (y - y0[i])
                + c[i] * (y - y0[i]) ** 2
            )
        )
    return v


def grad_muller_brown_potential_three_stats_analytical(data, backend="auto"):
    """
    data: shape (2)
    """
    x = data[0]
    y = data[1]

    prefactor = 0.15
    A = np.array([-280, -170, -170, 15])
    a = np.array([-15, -1, -6.5, 0.7])
    b = np.array([0, 0, 11, 0.6])
    c = np.array([-10, -10, -6.5, 0.7])
    x0 = np.array([1, 0.2, -0.5, -1])
    y0 = np.array([0, 0.5, 1.5, 1])
    # offset = -146.7

    # v = -prefactor * offset
    dvdx = 0.0
    dvdy = 0.0
    for i in range(4):
        dvdx += (
            prefactor
            * A[i]
            * (2 * a[i] * (x - x0[i]) + b[i] * (y - y0[i]))
            * np.exp(
                a[i] * (x - x0[i]) ** 2
                + b[i] * (1 - x0[i]) * (y - y0[i])
                + c[i] * (y - y0[i]) ** 2
            )
        )
        dvdy += (
            prefactor
            * A[i]
            * (b[i] * (x - x0[i]) + 2 * c[i] * (y - y0[i]))
            * np.exp(
                a[i] * (x - x0[i]) ** 2
                + b[i] * (1 - x0[i]) * (y - y0[i])
                + c[i] * (y - y0[i]) ** 2
            )
        )

    return np.vstack((dvdx, dvdy)).T


def grad_muller_brown_potetnial_three_stats(data, backend="auto"):
    """
    data: shape (2)
    backend: "auto", "jax", "numpy"
    """
    if backend == "jax" or (backend == "auto" and _HAS_JAX):
        energy = lambda data: muller_brown_potential_three_stats(data, jnp)
        grad_energy = jax.grad(energy)
        return grad_energy(jnp.asarray(data))

    return grad_muller_brown_potential_three_stats_analytical(np.asarray(data))


def leps_potential(data, xp):
    """
    Calculate the LEPS potential at coordinates (x, y).
    This is a 2D potential function commonly used in optimization problems.
    """
    x = data[0]
    y = data[1]

    a = 0.05
    b = 0.30
    c = 0.05
    dab = 4.746
    dbc = 4.746
    dac = 3.445
    r0 = 0.742
    alpha = 1.942

    rab = x
    rbc = y
    rac = x + y

    qab = (
        0.5
        * dab
        * (1.5 * xp.exp(-2 * alpha * (rab - r0)) - xp.exp(-alpha * (rab - r0)))
    )
    qbc = (
        0.5
        * dbc
        * (1.5 * xp.exp(-2 * alpha * (rbc - r0)) - xp.exp(-alpha * (rbc - r0)))
    )
    qac = (
        0.5
        * dac
        * (1.5 * xp.exp(-2 * alpha * (rac - r0)) - xp.exp(-alpha * (rac - r0)))
    )
    jab = (
        0.25 * dab * (xp.exp(-2 * alpha * (rab - r0)) - 6 * xp.exp(-alpha * (rab - r0)))
    )
    jbc = (
        0.25 * dbc * (xp.exp(-2 * alpha * (rbc - r0)) - 6 * xp.exp(-alpha * (rbc - r0)))
    )
    jac = (
        0.25 * dac * (xp.exp(-2 * alpha * (rac - r0)) - 6 * xp.exp(-alpha * (rac - r0)))
    )

    v = (
        qab / (1 + a)
        + qbc / (1 + b)
        + qac / (1 + c)
        - xp.sqrt(
            jab**2 / (1 + a) ** 2
            + jbc**2 / (1 + b) ** 2
            + jac**2 / (1 + c) ** 2
            - jab * jbc / (1 + a) / (1 + b)
            - jbc * jac / (1 + b) / (1 + c)
            - jab * jac / (1 + a) / (1 + c)
        )
    )
    return v


def grad_leps_potential_analytic(data):
    """
    data: shape (2)
    """
    x = data[0]
    y = data[1]

    a = 0.05
    b = 0.30
    c = 0.05
    dab = 4.746
    dbc = 4.746
    dac = 3.445
    r0 = 0.742
    alpha = 1.942

    rab = x
    rbc = y
    rac = x + y

    qab = (
        0.5
        * dab
        * (1.5 * np.exp(-2 * alpha * (rab - r0)) - np.exp(-alpha * (rab - r0)))
    )

    dqab_dx = (
        0.5
        * dab
        * (
            1.5 * (-2) * alpha * np.exp(-2 * alpha * (rab - r0))
            - (-alpha) * np.exp(-alpha * (rab - r0))
        )
    )

    dqab_dy = 0.0

    qbc = (
        0.5
        * dbc
        * (1.5 * np.exp(-2 * alpha * (rbc - r0)) - np.exp(-alpha * (rbc - r0)))
    )

    dqbc_dx = 0.0
    dqbc_dy = (
        0.5
        * dbc
        * (
            1.5 * (-2 * alpha) * np.exp(-2 * alpha * (rbc - r0))
            - (-alpha) * np.exp(-alpha * (rbc - r0))
        )
    )

    qac = (
        0.5
        * dac
        * (1.5 * np.exp(-2 * alpha * (rac - r0)) - np.exp(-alpha * (rac - r0)))
    )

    dqac_dx = (
        0.5
        * dac
        * (
            1.5 * (-2 * alpha) * np.exp(-2 * alpha * (rac - r0))
            - (-alpha) * np.exp(-alpha * (rac - r0))
        )
    )
    dqac_dy = dqac_dx

    jab = (
        0.25 * dab * (np.exp(-2 * alpha * (rab - r0)) - 6 * np.exp(-alpha * (rab - r0)))
    )

    djab_dx = (
        0.25
        * dab
        * (
            -2 * alpha * np.exp(-2 * alpha * (rab - r0))
            - -6 * (-alpha) * np.exp(-alpha * (rab - r0))
        )
    )
    djab_dy = 0.0

    jbc = (
        0.25 * dbc * (np.exp(-2 * alpha * (rbc - r0)) - 6 * np.exp(-alpha * (rbc - r0)))
    )

    djbc_dx = 0.0
    djbc_dy = (
        0.25
        * dbc
        * (
            -2 * alpha * np.exp(-2 * alpha * (rbc - r0))
            - -6 * (-alpha) * np.exp(-alpha * (rbc - r0))
        )
    )

    jac = (
        0.25 * dac * (np.exp(-2 * alpha * (rac - r0)) - 6 * np.exp(-alpha * (rac - r0)))
    )

    djac_dx = (
        0.25
        * dac
        * (
            -2 * alpha * np.exp(-2 * alpha * (rac - r0))
            - -6 * (-alpha) * np.exp(-alpha * (rac - r0))
        )
    )
    djac_dy = djac_dx

    v = (
        qab / (1 + a)
        + qbc / (1 + b)
        + qac / (1 + c)
        - np.sqrt(
            jab**2 / (1 + a) ** 2
            + jbc**2 / (1 + b) ** 2
            + jac**2 / (1 + c) ** 2
            - jab * jbc / (1 + a) / (1 + b)
            - jbc * jac / (1 + b) / (1 + c)
            - jab * jac / (1 + a) / (1 + c)
        )
    )

    dvdx = (
        dqab_dx / (1 + a)
        + dqab_dy / (1 + b)
        + dqac_dx / (1 + c)
        - 1
        / 2
        * 1
        / np.sqrt(
            jab**2 / (1 + a) ** 2
            + jbc**2 / (1 + b) ** 2
            + jac**2 / (1 + c) ** 2
            - jab * jbc / (1 + a) / (1 + b)
            - jbc * jac / (1 + b) / (1 + c)
            - jab * jac / (1 + a) / (1 + c)
        )
        * (
            2 * jab * djab_dx / (1 + a) ** 2
            + 2 * jbc * djbc_dx / (1 + b) ** 2
            + 2 * jac * djac_dx / (1 + c) ** 2
            - djab_dx * jbc / (1 + a) / (1 + b)
            - jab * djbc_dx / (1 + a) / (1 + b)
            - djbc_dx * jac / (1 + b) / (1 + c)
            - jbc * djac_dx / (1 + b) / (1 + c)
            - djab_dx * jac / (1 + a) / (1 + c)
            - jab * djac_dx / (1 + a) / (1 + c)
        )
    )
    dvdy = (
        dqab_dy / (1 + a)
        + dqbc_dy / (1 + b)
        + dqac_dy / (1 + c)
        - 1
        / 2
        * 1
        / np.sqrt(
            jab**2 / (1 + a) ** 2
            + jbc**2 / (1 + b) ** 2
            + jac**2 / (1 + c) ** 2
            - jab * jbc / (1 + a) / (1 + b)
            - jbc * jac / (1 + b) / (1 + c)
            - jab * jac / (1 + a) / (1 + c)
        )
        * (
            2 * jab * djab_dy / (1 + a) ** 2
            + 2 * jbc * djbc_dy / (1 + b) ** 2
            + 2 * jac * djac_dy / (1 + c) ** 2
            - djab_dy * jbc / (1 + a) / (1 + b)
            - jab * djbc_dy / (1 + a) / (1 + b)
            - djbc_dy * jac / (1 + b) / (1 + c)
            - jbc * djac_dy / (1 + b) / (1 + c)
            - djab_dy * jac / (1 + a) / (1 + c)
            - jab * djac_dy / (1 + a) / (1 + c)
        )
    )
    return np.vstack((dvdx, dvdy)).T


def grad_leps_potential(data, backend="auto"):
    """
    data: shape (2)
    backend: "auto", "jax", "numpy"
    """
    if backend == "jax" or (backend == "auto" and _HAS_JAX):
        energy = lambda data: leps_potential(data, jnp)
        grad_energy = jax.grad(energy)
        return grad_energy(jnp.asarray(data))

    return grad_leps_potential_analytic(np.asarray(data))


def leps_potential_harmonic(data, xp):
    """
    Calculate the LEPS potential with a harmonic term at coordinates (x, y).
    This is a modified version of the LEPS potential.
    """
    x = data[0]
    y = data[1]

    a = 0.05
    b = 0.30
    c = 0.05
    dab = 4.746
    dbc = 4.746
    dac = 3.445
    r0 = 0.742
    alpha = 1.942

    rab = x
    rbc = y
    rac = x + y

    qab = (
        0.5
        * dab
        * (1.5 * xp.exp(-2 * alpha * (rab - r0)) - xp.exp(-alpha * (rab - r0)))
    )
    qbc = (
        0.5
        * dbc
        * (1.5 * xp.exp(-2 * alpha * (rbc - r0)) - xp.exp(-alpha * (rbc - r0)))
    )
    qac = (
        0.5
        * dac
        * (1.5 * xp.exp(-2 * alpha * (rac - r0)) - xp.exp(-alpha * (rac - r0)))
    )
    jab = (
        0.25 * dab * (xp.exp(-2 * alpha * (rab - r0)) - 6 * xp.exp(-alpha * (rab - r0)))
    )
    jbc = (
        0.25 * dbc * (xp.exp(-2 * alpha * (rbc - r0)) - 6 * xp.exp(-alpha * (rbc - r0)))
    )
    jac = (
        0.25 * dac * (xp.exp(-2 * alpha * (rac - r0)) - 6 * xp.exp(-alpha * (rac - r0)))
    )

    v = (
        qab / (1 + a)
        + qbc / (1 + b)
        + qac / (1 + c)
        - xp.sqrt(
            jab**2 / (1 + a) ** 2
            + jbc**2 / (1 + b) ** 2
            + jac**2 / (1 + c) ** 2
            - jab * jbc / (1 + a) / (1 + b)
            - jbc * jac / (1 + b) / (1 + c)
            - jab * jac / (1 + a) / (1 + c)
        )
        + x**2
        + y**2
    )
    return v


def wolfe_quapp_potential(data, xp):
    x = data[0]
    y = data[1]

    a = 1
    b = 1
    v = (
        a * (x**4 + y**4)
        - b * (2 * x**2 + 4 * y**2 - x * y)
        + x * y
        + 0.3 * x
        + 0.1 * y
    )
    return v


def wolfe_quapp_potential_local_soft(data, xp):
    """
    Calculate the Wolfe-Quapp potential with a local softening term at coordinates (x, y).
    This is a modified version of the Wolfe-Quapp potential.
    """
    x = data[0]
    y = data[1]

    a = 1
    b = 1
    A = 5

    v = (
        a * (x**4 + y**4)
        - b * (2 * x**2 + 4 * y**2 - x * y)
        + x * y
        + 0.3 * x
        + 0.1 * y
    )
    v += A * xp.exp(-(x + 1.17) / (0.2 * 2))
    return v
