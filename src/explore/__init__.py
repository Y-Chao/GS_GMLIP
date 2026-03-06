#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import annotations

__author__ = "Chao Yang"
__version__ = "1.0"

"""Explore module is used to explore the potential energy of surface.
Currently, we implement three methods to explore the PES, which are constraint molecular dynamics(cmd),
Grand Canonical Monte Carlo (GCMC) and Grand Canonical Genetic Algorithm (GCGA). The stochastic surface
warker method (SSW) may be implemented in the future.
"""


class BaseExplore:
    """Base class for explore module"""

    def __init__(self):
        pass
