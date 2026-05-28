"""The core module for defining the Interface class, which defines the interface for the global searching.
The `Interface` class is designed to be flexible and extensible, allowing for easy integration with `ASE` and `pymatgen`,
the former is easy to interface with the calculators, while the latter is more proficient in structure manipulation.
"""

from typing import Optional

from ase import Atoms


class Interface:
    def __init__(self, substrate: Optional[Atoms] = None, interface: Optional[Atoms] = None, **kwargs):
        """Initialize the Interface object.

        Args:
            substrate (Optional[Atoms], optional): The substrate structure. Defaults to None.
            interface (Optional[Atoms], optional): The interface structure. Defaults to None.
            **kwargs: Additional keyword arguments for future extensions.
        """

        self.substrate = substrate if substrate is not None else Atoms()
        self.interface = interface if interface is not None else self.substrate.copy()

    def __repr__(self):
        return f'== Interface ==\n\tSubstrate: {self.substrate}\n\tInterface: {self.interface}'

    def _parse_substrate(self):
        """Parse the substrate structure. This method can be overridden by subclasses to implement specific parsing logic."""
        pass

    def _parse_interface(self):
        """Parse the interface structure. This method can be overridden by subclasses to implement specific parsing logic."""
        pass
