"""Analytic 2D potential energy surfaces for benchmarking global-search algorithms."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib import pyplot as plt

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jax = None
    jnp = None

# ---------------------------------------------------------------------------
# PES registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PESConfig:
    """Configuration for one named analytic PES.

    Attributes:
        name: Human-readable label (key in REGISTRY).
        family: Mathematical family -- ``"muller_brown"``, ``"leps"``, or ``"wolfe_quapp"``.
        params: Family-specific coefficients as a flat dict.
        x_range: Default x-axis range ``(x_min, x_max)``.
        y_range: Default y-axis range ``(y_min, y_max)``.
        z_max: Clipping value for contour plots.
    """

    name: str
    family: str
    params: dict[str, Any]
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_max: float


REGISTRY: dict[str, PESConfig] = {
    "muller_brown": PESConfig(
        name="muller_brown",
        family="muller_brown",
        params={
            "prefactor": 0.15,
            "A": [-200, -100, -170, 15],
            "a": [-1, -1, -6.5, 0.7],
            "b": [0, 0, 11, 0.6],
            "c": [-10, -10, -6.5, 0.7],
            "x0": [1, 0, -0.5, -1],
            "y0": [0, 0.5, 1.5, 1],
            "offset": -146.7,
        },
        x_range=(-1.75, 1.25),
        y_range=(-0.5, 2.5),
        z_max=24,
    ),
    "muller_brown_three_stats": PESConfig(
        name="muller_brown_three_stats",
        family="muller_brown",
        params={
            "prefactor": 0.15,
            "A": [-280, -170, -170, 15],
            "a": [-15, -1, -6.5, 0.7],
            "b": [0, 0, 11, 0.6],
            "c": [-10, -10, -6.5, 0.7],
            "x0": [1, 0.2, -0.5, -1],
            "y0": [0, 0.5, 1.5, 1],
            "offset": -146.7,
        },
        x_range=(-1.75, 1.25),
        y_range=(-0.5, 2.5),
        z_max=24,
    ),
    "leps": PESConfig(
        name="leps",
        family="leps",
        params={
            "a": 0.05,
            "b": 0.30,
            "c": 0.05,
            "d_ab": 4.746,
            "d_bc": 4.746,
            "d_ac": 3.445,
            "r0": 0.742,
            "alpha": 1.942,
            "harmonic": False,
        },
        x_range=(0, 4),
        y_range=(0, 4),
        z_max=2,
    ),
    "leps_harmonic": PESConfig(
        name="leps_harmonic",
        family="leps",
        params={
            "a": 0.05,
            "b": 0.30,
            "c": 0.05,
            "d_ab": 4.746,
            "d_bc": 4.746,
            "d_ac": 3.445,
            "r0": 0.742,
            "alpha": 1.942,
            "harmonic": True,
        },
        x_range=(0, 4),
        y_range=(0, 4),
        z_max=2,
    ),
    "wolfe_quapp": PESConfig(
        name="wolfe_quapp",
        family="wolfe_quapp",
        params={
            "a": 1,
            "b": 1,
            "local_soft": False,
        },
        x_range=(-2, 2),
        y_range=(-2, 2),
        z_max=2,
    ),
    "wolfe_quapp_local_soft": PESConfig(
        name="wolfe_quapp_local_soft",
        family="wolfe_quapp",
        params={
            "a": 1,
            "b": 1,
            "local_soft": True,
            "A_local": 5,
            "x0_local": -1.17,
            "sigma_local": 0.2,
        },
        x_range=(-2, 2),
        y_range=(-2, 2),
        z_max=2,
    ),
}

# ---------------------------------------------------------------------------
# Family implementations -- one function per mathematical form
# ---------------------------------------------------------------------------


def _muller_brown(x, y, params: dict, xp) -> Any:
    """Sum of four Gaussian terms (Muller-Brown family)."""
    p = params
    pfac = p["prefactor"]
    v = -pfac * p["offset"]
    for i in range(4):
        dx = x - p["x0"][i]
        dy = y - p["y0"][i]
        v += pfac * p["A"][i] * xp.exp(
            p["a"][i] * dx**2 + p["b"][i] * dx * dy + p["c"][i] * dy**2
        )
    return v


def _leps(x, y, params: dict, xp) -> Any:
    """LEPS potential with optional harmonic tail."""
    p = params
    a, b, c = p["a"], p["b"], p["c"]
    d_ab, d_bc, d_ac = p["d_ab"], p["d_bc"], p["d_ac"]
    r0 = p["r0"]
    alpha = p["alpha"]

    rab, rbc, rac = x, y, x + y

    def _q(d, r):
        return 0.5 * d * (1.5 * xp.exp(-2 * alpha * (r - r0)) - xp.exp(-alpha * (r - r0)))

    def _j(d, r):
        return 0.25 * d * (xp.exp(-2 * alpha * (r - r0)) - 6 * xp.exp(-alpha * (r - r0)))

    q_ab, q_bc, q_ac = _q(d_ab, rab), _q(d_bc, rbc), _q(d_ac, rac)
    j_ab, j_bc, j_ac = _j(d_ab, rab), _j(d_bc, rbc), _j(d_ac, rac)

    v = (
        q_ab / (1 + a)
        + q_bc / (1 + b)
        + q_ac / (1 + c)
        - xp.sqrt(
            j_ab**2 / (1 + a) ** 2
            + j_bc**2 / (1 + b) ** 2
            + j_ac**2 / (1 + c) ** 2
            - j_ab * j_bc / ((1 + a) * (1 + b))
            - j_bc * j_ac / ((1 + b) * (1 + c))
            - j_ab * j_ac / ((1 + a) * (1 + c))
        )
    )

    if p.get("harmonic", False):
        v += x**2 + y**2
    return v


def _wolfe_quapp(x, y, params: dict, xp) -> Any:
    """Wolfe-Quapp quartic potential with optional local softening."""
    p = params
    a, b = p["a"], p["b"]
    v = (
        a * (x**4 + y**4)
        - b * (2 * x**2 + 4 * y**2 - x * y)
        + x * y
        + 0.3 * x
        + 0.1 * y
    )
    if p.get("local_soft", False):
        v += p["A_local"] * xp.exp(-(x - p["x0_local"]) / (2 * p["sigma_local"]))
    return v


# ---------------------------------------------------------------------------
# Numeric gradient / Hessian factory (NumPy backend only)
# ---------------------------------------------------------------------------


def _make_numeric_gradient(impl_fn):
    """Return (grad_fn, hess_fn) using central finite differences.

    Each returned function has the signature ``fn(x, y, params, xp)``
    so callers can pass them through identically whether using autodiff or
    finite differences.
    """
    eps = 1e-5

    def grad_fn(x, y, params, xp):
        dfdx = (impl_fn(x + eps, y, params, xp) - impl_fn(x - eps, y, params, xp)) / (2 * eps)
        dfdy = (impl_fn(x, y + eps, params, xp) - impl_fn(x, y - eps, params, xp)) / (2 * eps)
        return xp.array([dfdx, dfdy])

    def hess_fn(x, y, params, xp):
        f = impl_fn
        f_xx = (f(x + eps, y, params, xp) + f(x - eps, y, params, xp) - 2 * f(x, y, params, xp)) / (eps**2)
        f_yy = (f(x, y + eps, params, xp) + f(x, y - eps, params, xp) - 2 * f(x, y, params, xp)) / (eps**2)
        f_xy = (
            f(x + eps, y + eps, params, xp)
            - f(x + eps, y - eps, params, xp)
            - f(x - eps, y + eps, params, xp)
            + f(x - eps, y - eps, params, xp)
        ) / (4 * eps**2)
        return xp.array([[f_xx, f_xy], [f_xy, f_yy]])

    return grad_fn, hess_fn


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
