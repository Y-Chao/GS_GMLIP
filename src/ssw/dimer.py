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

import numpy as np

from ssw.path import Image


class BasicDimer:
    def __init__(self, atoms, calculator, norm, delta_r=0.01):
        """
        Initialize the dimer with two images separated by delta_r along the norm direction.
        """
        self.norm = np.reshape(norm / np.linalg.norm(norm), (1, -1))
        self.delta_r = delta_r
        self.calculator = calculator
        self.image = Image(atoms.copy())
        self.image1 = Image(atoms.copy())
        self.image2 = Image(atoms.copy())
        self.image1.update(self.image.data + delta_r * self.norm)
        self.image2.update(self.image.data - delta_r * self.norm)

    def rotate(self):
        """
        Rotate the dimer to align with the minimum curvature direction.
        """
        ...

    def translate(self):
        """
        Translate the dimer along the specified direction.
        """
        ...

    def rotate_modified_newton(self, trial_angles=1, F_prime=None):
        """
        Rotate the dimer using a modified Newton-Raphson method.
        Netwon method: theta_{n+1} = theta_n - f(theta_n) / f'(theta_n)
        Modified Newton method: theta_{n+1} = theta_n - alpha * f(theta_n) / f'(theta_n)
            where alpha is a damping factor to control the step size, usually between 0 and 1.
            In addition, using finite difference to approximate f'(theta_n).
        """

        # Finite difference estimation of F'
        force = self.get_sum_forces()
        rotation_force = self.dimer_rotation_force()
        rotation_norm = self.rotation_norm(rotation_force)

        if F_prime is None:
            F_prime = self.get_gradient_force(
                trial_angles,
                force,
                rotation_force,
            )

        angles = (np.dot(force, rotation_norm.T)) / (-2 * F_prime)

        self.image1.update(
            self.image.data
            + self.delta_r
            * (self.norm * np.cos(angles) + rotation_norm * np.sin(angles))
        )
        self.image2.update(
            self.image.data
            - self.delta_r
            * (self.norm * np.cos(angles) + rotation_norm * np.sin(angles))
        )
        self.update_norm()
        return float(angles[0]), F_prime

    def get_gradient_force(
        self,
        trial_angles,
        force,
        rotation_force,
    ):
        """
        Calculate the gradient force for the dimer rotation.
        """
        angles = trial_angles / 180 * np.pi  # convert to radians

        force = self.get_sum_forces()
        rotation_norm = self.rotation_norm(rotation_force)

        # update images by the trial angles
        self.image1.update(
            self.image.data
            + self.delta_r
            * (self.norm * np.cos(angles) + rotation_norm * np.sin(angles))
        )
        self.image2.update(
            self.image.data
            - self.delta_r
            * (self.norm * np.cos(angles) + rotation_norm * np.sin(angles))
        )

        force_new = self.get_sum_forces()
        rotation_force_new = self.dimer_rotation_force()
        rotation_norm_new = self.rotation_norm(rotation_force_new)

        F_prime = (
            np.linalg.norm(
                np.dot(force_new, rotation_norm_new.T) - np.dot(force, rotation_norm.T)
            )
            / trial_angles
        )
        return F_prime

    def rotate_modified_gradient_analytic(self, trial_angles=1, F_prime=None):
        """
        Rotate the dimer using a modified gradient method with analytic derivative.
        F0 = A sin(2(theta - theta_min))
        F0' = 2A cos(2(theta - theta_min))
        theta = -1/2 * arctan(2F0/F0')
        """

        force = self.get_sum_forces()
        rotation_force = self.dimer_rotation_force()
        rotation_norm = self.rotation_norm(rotation_force)

        if F_prime is None:
            F_prime = self.get_gradient_force(
                trial_angles,
                force,
                rotation_force,
            )

        F = np.linalg.norm(force) / 2
        angles = np.arctan(2 * F / F_prime) / -2
        self.image1.update(
            self.image.data
            + self.delta_r
            * (self.norm * np.cos(angles) + rotation_norm * np.sin(angles))
        )
        self.image2.update(
            self.image.data
            - self.delta_r
            * (self.norm * np.cos(angles) + rotation_norm * np.sin(angles))
        )
        self.update_norm()
        return float(angles * 180 / np.pi), F_prime

    def update_norm(self):
        """
        Update the dimer norm based on the current images.
        """
        dimer_vector = self.image1.data - self.image2.data
        self.norm = dimer_vector / np.linalg.norm(dimer_vector)

    def rotation_norm(self, rotation_vector):
        """
        Calculate the rotation norm.
        This is the unit vector perpendicular to the dimer direction.
        """
        return rotation_vector / np.linalg.norm(rotation_vector)

    def parallel_force(self, force, norm):
        """
        Calculate the parallel force.
        """
        return np.dot(force, norm.T) * norm

    def rotation_force(self, force, norm):
        """
        Calculate the rotation force.
        """
        f_parallel = self.parallel_force(force, norm)
        f_perpendicular = force - f_parallel
        return f_perpendicular

    def dimer_rotation_force(self):
        """
        Calculate the dimer rotation force.
        """
        rotation_force_image1 = self.rotation_force(
            self.calculator.get_forces(self.image1.data[:, 0], self.image1.data[:, 1]),
            self.norm,
        )
        rotation_force_image2 = self.rotation_force(
            self.calculator.get_forces(self.image2.data[:, 0], self.image2.data[:, 1]),
            self.norm,
        )
        return rotation_force_image1 - rotation_force_image2

    def scalar_rotation_force(self):
        """
        Calculate the scalar rotation force.
        """
        vertical_force = self.dimer_rotation_force()
        rotation_norm = self.rotation_norm(vertical_force)
        return np.dot(vertical_force, rotation_norm.T) / self.delta_r

    def get_potential_energy(self):
        """
        Get the potential energy of the dimer (average of the two images).
        """
        e1 = self.calculator.get_potential_energy(
            self.image1.data[:, 0], self.image1.data[:, 1]
        )
        e2 = self.calculator.get_potential_energy(
            self.image2.data[:, 0], self.image2.data[:, 1]
        )
        e0 = self.calculator.get_potential_energy(
            self.image.data[:, 0], self.image.data[:, 1]
        )
        return e1, e2, e0

    def get_forces(self):
        """
        Get the forces on the two images.
        """
        f1 = self.calculator.get_forces(self.image1.data[:, 0], self.image1.data[:, 1])
        f2 = self.calculator.get_forces(self.image2.data[:, 0], self.image2.data[:, 1])
        return f1, f2

    def get_sum_forces(self):
        """
        Get the sum of forces on the two images.
        """
        f1, f2 = self.get_forces()
        return f1 + f2

    def get_sum_energy(self):
        """
        Get the sum of potential energies of the two images.
        """
        e1, e2, _ = self.get_potential_energy()
        return e1 + e2

    def get_curvature(self, dimer_energy, mid_energy):
        """
        Calculate the curvature of the dimer.
        """
        return (dimer_energy - 2 * mid_energy) / (self.delta_r**2)

    def run_modified_newton(
        self, max_steps=100, trial_angles=10, tol=1e-3, verbose=True
    ):
        """
        Run the dimer method for a maximum number of steps or until convergence.
        """
        prev_curvature = 1000.0
        F_prime = None
        record_images = []
        print("Starting dimer run...")
        print(f"Initial norm: {self.norm}")
        print(f"{'Step':<5} {'Curvature':<15} {'Rotation (deg)':<15}")
        for step in range(max_steps):
            dimer_energy = self.get_sum_energy()
            mid_energy = self.get_potential_energy()[2]
            curvature = self.get_curvature(dimer_energy, mid_energy)
            print(f"{step:<5} {curvature[0]:.4f} {trial_angles:.4f}")
            if verbose:
                record_images.append([self.image1.data.copy(), self.image2.data.copy()])

            trial_angles, F_prime = self.rotate_modified_newton(trial_angles, F_prime)

            if curvature[0] < prev_curvature:
                prev_curvature = curvature[0]
                record_image_1 = self.image1.data.copy()
                record_image_2 = self.image2.data.copy()
            else:
                print(f"Converged at step {step}")
                self.image1.update(record_image_1)
                self.image2.update(record_image_2)
                break
        return record_images

    def run_modified_newton_analytical(
        self, max_steps=100, trial_angles=10, tol=1e-3, verbose=True
    ):
        """
        Run the dimer method for a maximum number of steps or until convergence.
        """
        prev_curvature = 1000.0
        F_prime = None
        record_images = []
        print("Starting dimer run...")
        print(f"Initial norm: {self.norm}")
        print(f"{'Step':<5} {'Curvature':<15} {'Rotation (deg)':<15}")
        for step in range(max_steps):
            dimer_energy = self.get_sum_energy()
            mid_energy = self.get_potential_energy()[2]
            curvature = self.get_curvature(dimer_energy, mid_energy)
            print(f"{step:<5} {curvature[0]:.4f} {trial_angles:.4f}")
            if verbose:
                record_images.append([self.image1.data.copy(), self.image2.data.copy()])

            trial_angles, F_prime = self.rotate_modified_gradient_analytic(
                trial_angles, F_prime
            )

            if curvature[0] < prev_curvature:
                prev_curvature = curvature[0]
                record_image_1 = self.image1.data.copy()
                record_image_2 = self.image2.data.copy()
            else:
                print(f"Converged at step {step}")
                self.image1.update(record_image_1)
                self.image2.update(record_image_2)
                break
        return record_images
        # Translate the dimer (not implemented here)
        # self.translate()
