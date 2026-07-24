"""It defines a series of analytic potential energy surfaces (PES)."""
from __future__ import annotations

__author__ = "Chao Yang"
__version__ = "1.0"

from abc import ABC, abstractmethod
import numpy as np
from matplotlib import pyplot as plt

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None

from gs_gmlip import utils

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


def muller_brown_potential(x, y, xp=jnp):
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


def muller_brown_potential_three_stats(x, y, xp=jnp):
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


def leps_potential(x, y, xp=jnp):
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


def leps_potential_harmonic(x, y, xp=jnp):
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


def wolfe_quapp_potential(x, y, xp=jnp):
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


def wolfe_quapp_potential_local_soft(x, y, xp=jnp):
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

class AnalyticPES_base(ABC):
    def __init__(self, pes_name: str):
        self.pes_name = pes_name

    def __call__(self, x: np.ndarray|list):
        return x, self.get_pes_energy(x), self.get_forces(x)

    def get_energy(self, x: np.ndarray|list):
        raise NotImplementedError

    def get_forces(self, x:np.ndarray|list):
        raise NotImplementedError

class AnalyticPES:
    def __init__(self, function_name, use_jax=False):
        self.function_name = function_name
        self.use_jax = use_jax
        self.xp = None
        self._initialize_function()

    def _initialize_function(self):
        if self.function_name == "muller_brown":
            self.func = muller_brown_potential
            self.range = [[-1.75, 1.25], [-0.5, 2.5]]
            self.z_max = 24
        elif self.function_name == "muller_brown_three_stats":
            self.func = muller_brown_potential_three_stats
            self.range = [[-1.75, 1.25], [-0.5, 2.5]]
            self.z_max = 24
        elif self.function_name == "leps":
            self.func = leps_potential
            self.range = [[0, 4], [0, 4]]
            self.z_max = 2
        elif self.function_name == "leps_harmonic":
            self.func = leps_potential_harmonic
            self.range = [[0, 4], [0, 4]]
            self.z_max = 2
        elif self.function_name == "wolfe_quapp":
            self.func = wolfe_quapp_potential
            self.range = [[-2, 2], [-2, 2]]
            self.z_max = 2
        elif self.function_name == "wolfe_quapp_local_soft":
            self.func = wolfe_quapp_potential_local_soft
            self.range = [[-2, 2], [-2, 2]]
            self.z_max = 2
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

    def get_potential_energy(self, x, y):
        if self.use_jax:
            if isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray):
                return self.xp.array(
                    [self.func(x[i], y[i], self.xp) for i in range(x.shape[0])]
                )
            elif isinstance(x, list):
                return [self.func(x[i], y[i], self.xp) for i in range(len(x))]
            else:
                return self.func(x, y, self.xp)
        else:
            return self.func(x, y, self.xp)

    def get_forces(self, x, y):
        if self.use_jax:
            if isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray):
                grads = self.xp.array(
                    [self.grad_func(x[i], y[i]) for i in range(x.shape[0])]
                )
                return -1 * grads
            elif isinstance(x, list):
                grads = [self.grad_func(x[i], y[i]) for i in range(len(x))]
                return -1 * self.xp.array(grads)
            else:
                return -1 * self.xp.array(self.grad_func(x, y))
        else:
            return -1 * self.grad_func(x, y)

    def get_hessian(self, x, y):
        if self.use_jax:
            if isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray):
                hessians = self.xp.array(
                    [self.hess_func(x[i], y[i]) for i in range(x.shape[0])]
                )
                return hessians
            elif isinstance(x, list):
                hessians = [self.hess_func(x[i], y[i]) for i in range(len(x))]
                return self.xp.array(hessians)
            else:
                return self.hess_func(x, y)
        else:
            return self.hess_func(x, y)

    def pes_matrix(self, x_range=None, y_range=None, num_points=100):
        """
        Get the matrix of potential energy surface.
        """
        if x_range is None:
            x_min, x_max = self.range[0]
        else:
            x_min, x_max = x_range
        if y_range is None:
            y_min, y_max = self.range[1]
        else:
            y_min, y_max = y_range

        x = self.xp.linspace(x_min, x_max, num_points)
        y = self.xp.linspace(y_min, y_max, num_points)
        X, Y = self.xp.meshgrid(x, y)
        Z = self.get_potential_energy(X, Y)
        return X, Y, Z

    def plot_pes(
        self, x_range=None, y_range=None, z_max=None, num_points=100, levels=50, ax=None
    ):

        X, Y, Z = self.pes_matrix(x_range, y_range, num_points)

        if z_max is None:
            z_max = self.z_max
        Z_mask = np.ma.masked_greater(Z, z_max)

        if ax is None:
            fig, ax = plt.subplots()

        ax.contourf(X, Y, Z_mask, levels=levels, cmap=utils.cm_fessa)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"PES: {self.function_name}")
        plt.colorbar(
            ax.contourf(X, Y, Z_mask, levels=levels, cmap=utils.cm_fessa),
            ax=ax,
            label="Potential Energy",
        )
        return ax

