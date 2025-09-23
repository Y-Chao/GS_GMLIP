#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import annotations

__author__ = "Chao Yang"
__version__ = "1.0"

"""
Implementation of the expressions of analytic potential energy surface (PES) and its derivatives.

Functions include:
- muller_brown_potential: Calculate the Muller-Brown potential at given coordinates.
- muller_brown_potential_three_stats: A variant of the Muller-Brown potential with three stationary points.
- leps_potential: Calculate the LEPS potential at given coordinates.
- leps_potential_harmonic: Calculate the LEPS potential with a harmonic term at given coordinates.
- wolfe_quapp_potential: Calculate the Wolfe-Quapp potential at given coordinates.
- wolfe_quapp_potential_local_soft: A variant of the Wolfe-Quapp potential with a local softening term.
"""

import numpy as np


def muller_brown_potential(x, y, xp):
    """
    Calculate the Muller-Brown potential at coordinates (x, y).
    This is a 2D potential function commonly used in optimization problems.

    Parameters:
    x (float or np.ndarray): x-coordinate(s)
    y (float or np.ndarray): y-coordinate(s)

    Returns:
    v (float or np.ndarray): The potential energy at the given coordinates.
    """
    prefactor = 0.15
    A = xp.array([-200, -100, -170, 15])
    a = xp.array([-1, -1, -6.5, 0.7])
    b = xp.array([0, 0, 11, 0.6])
    c = xp.array([-10, -10, -6.5, 0.7])
    x0 = xp.array([1, 0, -0.5, -1])
    y0 = xp.array([0, 0.5, 1.5, 1])
    offset = -146.7

    v = -prefactor * offset
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


def muller_brown_potential_three_stats(x, y, xp):
    """
    Calculate the Muller-Brown potential at coordinates (x, y).
    This is a 2D potential function commonly used in optimization problems.

    Parameters:
    x (float or np.ndarray): x-coordinate(s)
    y (float or np.ndarray): y-coordinate(s)

    Returns:
    v (float or np.ndarray): The potential energy at the given coordinates.
    """
    prefactor = 0.15
    A = xp.array([-280, -170, -170, 15])
    a = xp.array([-15, -1, -6.5, 0.7])
    b = xp.array([0, 0, 11, 0.6])
    c = xp.array([-10, -10, -6.5, 0.7])
    x0 = xp.array([1, 0.2, -0.5, -1])
    y0 = xp.array([0, 0.5, 1.5, 1])
    offset = -146.7

    v = -prefactor * offset
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


def leps_potential(x, y, xp):
    """
    Calculate the LEPS potential at coordinates (x, y).
    This is a 2D potential function commonly used in optimization problems.

    Parameters:
    x (float or np.ndarray): x-coordinate(s)
    y (float or np.ndarray): y-coordinate(s)

    Returns:
    v (float or np.ndarray): The potential energy at the given coordinates.
    """
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


def leps_potential_harmonic(x, y, xp):
    """
    Calculate the LEPS potential with a harmonic term at coordinates (x, y).
    This is a modified version of the LEPS potential.

    Parameters:
    x (float or np.ndarray): x-coordinate(s)
    y (float or np.ndarray): y-coordinate(s)

    Returns:
    v (float or np.ndarray): The potential energy at the given coordinates.
    """
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


def wolfe_quapp_potential(x, y, xp):
    """
    Calculate the Wolfe-Quapp potential at coordinates (x, y).

    Parameters:
    x (float or np.ndarray): x-coordinate(s)
    y (float or np.ndarray): y-coordinate(s)

    Returns:
    v (float or np.ndarray): The potential energy at the given coordinates.
    """
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


def wolfe_quapp_potential_local_soft(x, y, xp):
    """
    Calculate the Wolfe-Quapp potential with a local softening term at coordinates (x, y).
    This is a modified version of the Wolfe-Quapp potential.

    Parameters:
    x (float or np.ndarray): x-coordinate(s)
    y (float or np.ndarray): y-coordinate(s)

    Returns:
    v (float or np.ndarray): The potential energy at the given coordinates.
    """
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


class AnalyticPES:
    def __init__(self, function_name, use_jax=False):
        self.function_name = function_name
        self.use_jax = use_jax
        self._initialize_function()

    def _initialize_function(self):
        if self.function_name == "muller_brown":
            self.func = muller_brown_potential
        elif self.function_name == "muller_brown_three_stats":
            self.func = muller_brown_potential_three_stats
        elif self.function_name == "leps":
            self.func = leps_potential
        elif self.function_name == "leps_harmonic":
            self.func = leps_potential_harmonic
        elif self.function_name == "wolfe_quapp":
            self.func = wolfe_quapp_potential
        elif self.function_name == "wolfe_quapp_local_soft":
            self.func = wolfe_quapp_potential_local_soft
        else:
            raise ValueError(f"Function '{self.function_name}' is not recognized.")

        if self.use_jax:
            import jax
            import jax.numpy as jnp

            self.xp = jnp
            self.grad_func = jax.jit(jax.grad(self.func, argnums=(0, 1)))
            self.hess_func = jax.jit(jax.hessian(self.func, argnums=(0, 1)))
        else:
            self.xp = np
            self.grad_func = self.numberic_gradient()[0]
            self.hess_func = self.numberic_gradient()[1]

    def numberic_gradient(self):
        eps = 1e-5

        def grad_func(x, y):
            dfdx = (self.func(x + eps, y, self.xp) - self.func(x - eps, y, self.xp)) / (
                2 * eps
            )
            dfdy = (self.func(x, y + eps, self.xp) - self.func(x, y - eps, self.xp)) / (
                2 * eps
            )
            return self.xp.array([dfdx, dfdy])

        def hess_func(x, y):
            f = self.func
            f_xx = (
                f(x + eps, y, self.xp) + f(x - eps, y, self.xp) - 2 * f(x, y, self.xp)
            ) / (eps**2)
            f_yy = (
                f(x, y + eps, self.xp) + f(x, y - eps, self.xp) - 2 * f(x, y, self.xp)
            ) / (eps**2)
            f_xy = (
                f(x + eps, y + eps, self.xp)
                - f(x + eps, y - eps, self.xp)
                - f(x - eps, y + eps, self.xp)
                + f(x - eps, y - eps, self.xp)
            ) / (4 * eps**2)

            return self.xp.array([[f_xx, f_xy], [f_xy, f_yy]])

        return grad_func, hess_func

    def get_potential(self, x, y):
        return self.func(x, y, self.xp)

    def get_gradient(self, x, y):
        return -self.grad_func(x, y)

    def get_hessian(self, x, y):
        return self.hess_func(x, y)
