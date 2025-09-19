#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import annotations

__author__ = 'Chao Yang'
__version__=	'1.0'

"""
This module implements the stochastic surface walk (SSW) algorithm based on ZhiPan Liu et al.

Pseudo SSW algorithm:
1. Climbing phase:
    a. Generate a random direction $N^0_i$ at the current minimum $R^m_i$
    .. math::
        N^0_i = (N^g_i + \lambda N^l_i) / ||N^g_i + \lambda N^l_i||
        
    - Soft global move: $N^g_i$ is a randomly generated normalized vector, the distribution satisfies the Maxwll-Boltzmann velocity distribution at 300K;
    - Stiff local move: $N^l_i$ is the bond formation mode between two non-neighboring atoms.
    b. A biased dimer rotation method is applied to refine the mode.
    .. math::
        \begin{align}
        \mathbf{R}_1 &= \mathbf{R}_0 + \mathbf{N}_t\cdot \Delta R \\
        C &= \frac{(mathbf{F}_0-\mathbf{F}_1)\cdot\mathbf{N}_t}{\Delta R} \\
        V_{R1} &= V_{real} + V_N \\
        V_N *= -\frac{a}{2} \cdot [(\mathbf{R}_1 - \mathbf{R}_0)\cdot\mathbf{N}_i^0]^2 = -\frac{a}{2}\cdot(\Delta R\cdot \mathbf{N}_t\mathbf{N}_i^0)^2
        \end{align}