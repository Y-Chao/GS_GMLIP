#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import annotations

__author__ = "Chao Yang"
__version__ = "1.0"


from abc import ABC, abstractmethod

try:
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False
import numpy as np
from analytical_pes import (
    grad_leps_potential,
    grad_muller_brown_potetnial,
    grad_muller_brown_potetnial_three_stats,
    leps_potential,
    muller_brown_potential,
    muller_brown_potential_three_stats,
)


class PES2D(ABC):
    resolution: int = 200

    def __init__(self, backend="auto"):
        self.backend = backend
        if backend == "auto" and _HAS_JAX:
            self.xp = jnp
        else:
            self.xp = np

    @abstractmethod
    def energy(self, data):
        pass

    @abstractmethod
    def gradient(self, data):
        pass

    @abstractmethod
    def hessian(self, data):
        pass


class MB_PES2D(PES2D):
    """
    Muller-Brown 2D Potential Energy Surface.
    """

    x_range = (-2.5, 1.5)
    y_range = (-0.6, 2.5)
    v_max = 5.0

    def energy(self, data):
        return muller_brown_potential(data, self.xp)

    def gradient(self, data):
        return grad_muller_brown_potetnial(data, self.backend)

    def hessian(self, data):
        pass


class MB_3S_PES2D(PES2D):
    """
    3-State Muller-Brown 2D Potential Energy Surface.
    """

    x_range = (-2.0, 2.0)
    y_range = (-1.0, 3.0)
    v_max = 5.0

    def energy(self, data):
        return muller_brown_potential_three_stats(data, self.xp)

    def gradient(self, data):
        return grad_muller_brown_potetnial_three_stats(data, self.backend)

    def hessian(self, data):
        pass


class LEPS_PES2D(PES2D):
    """
    LEPS 2D Potential Energy Surface.
    """

    x_range = (0, 4)
    y_range = (0, 4)
    v_max = -2

    def energy(self, data):
        return leps_potential(data, self.xp)

    def gradient(self, data):
        return grad_leps_potential(data, self.backend)

    def hessian(self, data):
        pass


def plot_isoline_2D(
    function,
    component=None,
    limits=((-1.8, 1.2), (-0.4, 2.1)),
    num_points=(100, 100),
    mode="contourf",
    levels: int | jnp.ndarray = 12,
    cmap=None,
    colorbar=None,
    max_value=None,
    ax=None,
    allow_grad=False,
    **kwargs,
):
    """
    Plot isolines of a 2D function.

    Parameters:
        function: Callable, the function to plot.
        component: int or None, the component to plot (if applicable).
        limits: tuple of tuples, the limits for x and y axes.
        num_points: tuple, number of points in x and y directions.
        mode: str, plotting mode ('contourf' or 'contour').
        levels: int, number of contour levels.
        cmap: Colormap, colormap to use.
        colorbar: bool, whether to show colorbar.
        ax: Axes, matplotlib axes to plot on.
        allow_grad: bool, whether to allow gradient computation.
        **kwargs: Additional keyword arguments for plotting.
    """
    if type(num_points) is int:
        num_points = (num_points, num_points)
    xx = jnp.linspace(limits[0][0], limits[0][1], num_points[0])
    yy = jnp.linspace(limits[1][0], limits[1][1], num_points[1])
    xv, yv = jnp.meshgrid(xx, yy)

    z = function(xv, yv)

    if max_value is not None:
        z = jnp.where(z > max_value, max_value, z)

    # Setup plot
    return_axs = False
    if ax is None:
        return_axs = True
        _, ax = plt.subplots(figsize=(6.0, 4.0), dpi=300)

    # Color scheme
    if cmap is None:
        if mode == "contourf":
            cmap = "fessa"
        elif mode == "contour":
            if "colors" not in kwargs:
                cmap = "Greys_r"

    # Colorbar
    if colorbar is None:
        if mode == "contourf":
            colorbar = True
        elif mode == "contour":
            colorbar = False

    # Plot
    if mode == "contourf":
        pp = ax.contourf(xv, yv, z, levels=levels, cmap=cmap, **kwargs)
        if colorbar:
            plt.colorbar(pp, ax=ax)
    else:
        pp = ax.contour(xv, yv, z, levels=levels, cmap=cmap, **kwargs)

    if return_axs:
        return ax
    else:
        return None
