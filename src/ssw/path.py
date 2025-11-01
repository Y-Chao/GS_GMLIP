#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import annotations

__author__ = "Chao Yang"
__version__ = "1.0"


from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms

if TYPE_CHECKING:
    from typing import List, Optional, Union

"""
The basic path for optimization and transition state search algorithms.
"""


class Node:
    """
    Node is a double linked list, used to store the images along the path.
    """

    def __init__(self, data=None):
        self.data = data
        self.next = None
        self.prev = None

    def link_next(self, n):
        self.next = n
        n.prev = self

    def link_prev(self, p):
        self.prev = p
        p.next = self


class Image(Node):
    """
    Image is a node that stores the information of an image along the path.
    """

    def __init__(self, image: Union[Atoms, List, "Image"]):
        if isinstance(image, Atoms):
            self._structures = image.copy()
            self.type = str(image.__class__)
            data = self._structures.positions
        elif isinstance(image, Image):
            if hasattr(image, "_structures") and hasattr(image._structures, "copy"):
                self._structures = image._structures.copy()
            else:
                self._structures = None
            self.type = image.type
            data = image.data.copy()
        else:
            self._structures = None
            self.type = "Simple"
            data = np.atleast_2d(image).astype(np.float64)
        super().__init__(data)

    def __str__(self):
        return f"Image type: {self.type}, and data: {self.data}"

    def __repr__(self):
        return self.__str__()

    def move(self, displacement: np.ndarray):
        """
        Move the image by the given displacement.
        """
        self.data += displacement
        if self._structures is not None:
            self._structures.set_positions(self.data)

    def update(self, new_data: np.ndarray):
        """
        Update the image data.
        """
        self.data = new_data
        if self._structures is not None:
            self._structures.set_positions(new_data)

    @property
    def structures(self) -> "Optional[Atoms]":
        """
        Return the ASE Atoms object of the image, or None if not set.
        """
        if isinstance(self._structures, Atoms):
            return self._structures
        else:
            pass

    @structures.setter
    def structures(self, atoms: Atoms):
        """
        Set the ASE Atoms object of the image.
        """
        self._structures = atoms.copy()
        self.data = atoms.positions

    def copy(self) -> "Image":
        """
        Return a copy of the image.
        """
        return self.__class__(self)
