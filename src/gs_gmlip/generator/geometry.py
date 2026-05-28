"""This module provides a series of geometry operations for the generator.
These operations include:
- `blda`: A method to generate the bond length based on the distribution of bond lengths in the training data.
"""

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii


def box_point_picking():
    """To pick a random point in a box, we can simply pick x, y, z from uniform distributions in the range of the box."""
    return np.random.rand(3)


def sphere_point_picking():
    """To pick a random point on the surface of a unit sphere, it is incorrect to select spherical coordinates theta
    and phi from uniform distributions theta in [0,2pi) and phi in [0,pi].

    Reference:
        Marsaglia (1972) derived this elegant method that consists of picking x1 and x2 from independent uniform
        distributions on (-1, 1) and rejecting points for which x1^2 + x2^2 >= 1. The remaining points are uniformly
        distributed on the unit shpere.
    """
    x1, x2 = 1, 1
    while x1**2 + x2**2 >= 1:
        x1, x2 = 2 * (np.random.rand(2) - 0.5)

    return np.array(
        [
            2 * x1 * np.sqrt(1 - x1**2 - x2**2),
            2 * x2 * np.sqrt(1 - x1**2 - x2**2),
            1 - 2 * (x1**2 + x2**2),
        ]
    )


def sphere_point_picking_constraint_z():
    """To pick a random point on the surface of a unit sphere with z > 0 based on the method of Marsaglia (1972)."""
    vector = sphere_point_picking()
    vector[2] = abs(vector[2])
    return vector


def find_point_on_line_within_distance(point: np.ndarray, ref_point: np.ndarray, distance: float):
    """To find a point on the line defined by the point and the reference point that is within a certain distance from the
    reference point."""
    ...


def BLDA(elem1: str | int, elem2: str | int, sigma: float = 0.1, scale: float = 1.0):
    """Bond Length Distribution Algorithm (BLDA) is a method to generate the bond length based on the distribution of
    bond lengths in the previous data.

    ref:
        J. Chem. Theory Comput. 2016, 12, 12, 6213–6226

    args:
        elem1 (str|int): The symbol|atomic number of the first element.
        elem2 (str|int): The symbol|atomic number of the second element.
        sigma (float, optional): The standard deviation of the Gaussian distribution. Defaults to 0.1.
        scale (float, optional): The scale factor for the bond length. Defaults to 1.0.
    returns:
        float: The generated bond length.
    """

    if type(elem1) is str:
        elem1 = atomic_numbers[elem1]
    if type(elem2) is str:
        elem2 = atomic_numbers[elem2]

    r1 = covalent_radii[elem1]
    r2 = covalent_radii[elem2]

    return np.random.normal(loc=(r1 + r2), scale=sigma) * scale


def F_BLDA(template: Atoms, elem2: str | int, sigma: float = 0.1, scale: float = 1.0):
    """The F-BLDA method is a variant of the BLDA method that generates the bond length based on the distribution of
    bond lengths in the previous data, but also takes into account the geometry of the template structure.

    args:
        template (Atoms): The template structure.
        elem2 (str|int): The symbol|atomic number of the second element.
        sigma (float, optional): The standard deviation of the Gaussian distribution. Defaults to 0.1.
        scale (float, optional): The scale factor for the bond length. Defaults to 1.0.
    returns:
        new_structure (Atoms): The generated structure with the new bond length.
    """
    if type(template) is not Atoms:
        raise ValueError('The template structure must be an instance of the Atoms class.')

    if len(template) == 0 and elem2 is None:
        raise ValueError('The template structure is empty and the second element is not specified.')

    if len(template) == 0:
        if not np.all(template.cell):
            default_cell = np.eye(3) * 10
            template.set_cell(default_cell)
            template.set_pbc([False, False, False])

        box = template.cell.array
        pos = np.dot(box_point_picking().T, box)
        template.append(elem2, positions=pos)
        return template
    
    com = template.get_center_of_mass()
    random_vector = sphere_point_picking()

    for i, elem1 in enumerate(template.get_chemical_symbols()):
        
