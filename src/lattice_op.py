#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import annotations

__author__ = "Chao Yang"
__version__ = "1.0"


from functools import cached_property

import numpy as np

"""
Lattice operations
===================
This module provides basic algebraic operations on lattices arrays.
"""


class LatticeOp:
    def __init__(self, array: np.ndarray):
        array = np.asarray(array, dtype=float)
        if array.shape != (3, 3):
            array = self.complete_lattice(array)
        self._matrix = array

    @cached_property
    def lengths(self) -> np.ndarray:
        """Return the lengths of the lattice vectors."""
        return np.linalg.norm(self._matrix, axis=1)

    @cached_property
    def angles(self) -> np.ndarray:
        """Return the angles (in degrees) between the lattice vectors."""
        angles = np.zeros(3)
        for i in range(3):
            j = i - 1
            k = i - 2
            # For numerical stability, clip the cosine values to the range [-1, 1]
            angles[i] = np.clip(
                np.dot(self._matrix[j], self._matrix[k])
                / (self.lengths[j] * self.lengths[k]),
                -1.0,
                1.0,
            )
        angles = np.arccos(angles) * (180.0 / np.pi)
        return angles

    @property
    def parameters(self) -> tuple[float, float, float, float, float, float]:
        """Return the cell parameters: [a, b, c, alpha, beta, gamma]."""
        return (*self.lengths, *self.angles)

    @property
    def a(self) -> float:
        return self.lengths[0]

    @property
    def b(self) -> float:
        return self.lengths[1]

    @property
    def c(self) -> float:
        return self.lengths[2]

    @property
    def alpha(self) -> float:
        return self.angles[0]

    @property
    def beta(self) -> float:
        return self.angles[1]

    @property
    def gamma(self) -> float:
        return self.angles[2]

    @property
    def reciprocal_lattice(self) -> np.ndarray:
        r"""Return the reciprocal lattice vectors.
        Generally, the basis vectors are stored as a column vector. So it should be transposed
        when calculating the reciprocal lattice.
        .. math::

            \mathbf{G} = 2 \pi (\mathbf{A}^{-1})^T

        Returns:
            np.ndarray: The 3x3 array of reciprocal lattice vectors.
        """
        return np.linalg.inv(self._matrix).T * (2.0 * np.pi)

    @property
    def reciprocal_lattice_crystallographic(self) -> np.ndarray:
        r"""Return the crystallographic reciprocal lattice vectors.
        Generally, the basis vectors are stored as a column vector. So it should be transposed
        when calculating the reciprocal lattice.
        .. math::

            \mathbf{G} = (\mathbf{A}^{-1})^T

        Returns:
            np.ndarray: The 3x3 array of reciprocal lattice vectors.
        """
        return np.linalg.inv(self._matrix).T

    @property
    def metric_tensor(self) -> np.ndarray:
        """Return the *metric tensor* of the lattice.
        *metric tensor* is a function tells how to compute the distances between any two points in a given space.

        For a lattice with basis vectors :math:`\\mathbf{a}`, :math:`\\mathbf{b}`, :math:`\\mathbf{c}`, the metric tensor g is defined as:
        .. math::

            g = 
            \begin{pmatrix}
            \\mathbf{a} \\cdot \\mathbf{a} & \\mathbf{a} \\cdot \\mathbf{b} & \\mathbf{a} \\cdot \\mathbf{c} \\
            \\mathbf{b} \\cdot \\mathbf{a} & \\mathbf{b} \\cdot \\mathbf{b} & \\mathbf{b} \\cdot \\mathbf{c} \\
            \\mathbf{c} \\cdot \\mathbf{a} & \\mathbf{c} \\cdot \\mathbf{b} & \\mathbf{c} \\cdot \\mathbf{c}
            \\end{pmatrix}
            =
            \begin{pmatrix}
            a^2 & ab \\cos \\gamma & ac \\cos \beta \\
            ab \\cos \\gamma & b^2 & bc \\cos \alpha \\
            ac \\cos \beta & bc \\cos \alpha & c^2
            \\end{pmatrix}

        Returns:
            np.ndarray: The 3x3 metric tensor.
        """
        return np.dot(self._matrix, self._matrix.T)

    @property
    def volume(self) -> float:
        """Return the volume of the unit cell."""
        a, b, c = self._matrix
        return abs(float(np.dot(a, np.cross(b, c))))

    @classmethod
    def from_parameters(
        cls, a: float, b: float, c: float, alpha: float, beta: float, gamma: float
    ) -> LatticeOp:
        """Create a LatticeOp instance from cell parameters."""
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
        gamma_rad = np.radians(gamma)

        v_x = a
        v_y = b * np.cos(gamma_rad)
        v_z = c * np.cos(beta_rad)

        w_y = b * np.sin(gamma_rad)
        w_z = (
            c
            * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad))
            / np.sin(gamma_rad)
        )

        u_z = c * np.sqrt(
            1
            - np.cos(beta_rad) ** 2
            - (
                (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad))
                / np.sin(gamma_rad)
            )
            ** 2
        )

        lattice_matrix = np.array([[v_x, 0.0, 0.0], [v_y, w_y, 0.0], [v_z, w_z, u_z]])
        return cls(lattice_matrix)

    def complete_lattice(self, array: np.ndarray) -> np.ndarray:
        """Complete the lattice array to a 3x3 matrix if it is given in a reduced form.

        Parameters:
            array (np.ndarray): The input lattice array, which can be of shape (1, 3), (2, 3), or (3, 3).
        """
        ...
