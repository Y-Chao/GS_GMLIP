#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import annotations

__author__ = "Chao Yang"
__version__ = "1.0"


"""
Utility functions for the GS_GMLIPS package.

Functions include:
- paletteFessa: A color palette for plotting, inspired by the Fessa color scheme.
- coolwarmLight: A light version of the coolwarm color map for plotting.
- wrapper functions to include the gradient and hessian calculations for analytic PES functions.

"""

import matplotlib as mpl
from matplotlib.colors import ColorConverter, LinearSegmentedColormap

"""
Load PaletteFessa colors, forked from https://github.com/luigibonati/fessa-color-palette/blob/master/fessa.py
"""

paletteFessa = [
    "#1F3B73",  # dark-blue
    "#2F9294",  # green-blue
    "#50B28D",  # green
    "#A7D655",  # pisello
    "#FFE03E",  # yellow
    "#FFA955",  # orange
    "#D6573B",  # red
]

# Create a colormap object
cm_fessa = LinearSegmentedColormap.from_list("fessa", paletteFessa)
mpl.colormaps.register(cmap=cm_fessa)
mpl.colormaps.register(cmap=cm_fessa.reversed())

# Register colors in paletteFessa
for i in range(len(paletteFessa)):
    ColorConverter.colors[f"fessa_{i}"] = paletteFessa[i]


coolwarmLight = ["#45A9E6", "#FFFFFF", "#FF5757"]  # light-blue  # white  # light-red

cm_cwl = LinearSegmentedColormap.from_list("coolwarmLight", coolwarmLight)
mpl.colormaps.register(cmap=cm_cwl)
mpl.colormaps.register(cmap=cm_cwl.reversed())

# Register colors in coolwarmLight
for i in range(len(coolwarmLight)):
    ColorConverter.colors[f"cwl_{i}"] = coolwarmLight[i]


def jax_backend(func):
    """
    A decorator to switch between numpy and jax.numpy based on JAX availability.
    """
    try:
        import jax
        import jax.numpy as jnp

        JAX_AVAILABLE = True
    except ImportError:
        import numpy as np

        JAX_AVAILABLE = False

    def wrapper(*args, **kwargs):
        if JAX_AVAILABLE:
            return func(*args, xp=jnp, **kwargs)
        else:
            return func(*args, xp=np, **kwargs)

    return wrapper
