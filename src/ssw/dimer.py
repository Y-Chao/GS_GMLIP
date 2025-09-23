#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import annotations

__author__ = "Chao Yang"
__version__ = "1.0"

"""
The core of the SSW algorithm is the BP-CBD (Biased potential driven constrained Broyden dimer) method.
It is similar to the original dimer method proposed by Henkelman et al., but with two main differences:
    1. The rotation of the dimer is biased by a pre-defined direction, which is a combination of a soft global move and a stiff local move.
    2. The translation of the dimer is driven by a biased potential, which is the real potential plus a biasing term that depends on the displacement along the pre-defined direction.

The pseudo code of BP-CBD is as follows:
"""
