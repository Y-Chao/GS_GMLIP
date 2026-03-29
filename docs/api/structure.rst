Structure Module
================

.. module:: gs_gmlip.structure

The structure module provides core data types for representing catalyst
slab/adsorbate systems.

SlabAtoms
---------

.. autoclass:: gs_gmlip.structure.atoms.SlabAtoms
   :members:
   :show-inheritance:

Region
------

.. autoclass:: gs_gmlip.structure.region.Region
   :members:

.. autoclass:: gs_gmlip.structure.region.BoxRegion
   :members:
   :show-inheritance:

.. autoclass:: gs_gmlip.structure.region.SphereRegion
   :members:
   :show-inheritance:

Bonds
-----

.. autoclass:: gs_gmlip.structure.bonds.BondData
   :members:

.. autofunction:: gs_gmlip.structure.bonds.get_bond_matrix

.. autofunction:: gs_gmlip.structure.bonds.get_coordination_numbers

.. autofunction:: gs_gmlip.structure.bonds.generate_blmin

Composition
-----------

.. autoclass:: gs_gmlip.structure.composition.MolecularBlock
   :members:

.. autoclass:: gs_gmlip.structure.composition.CompositionConstraint
   :members:

Predefined Blocks
^^^^^^^^^^^^^^^^^

.. autofunction:: gs_gmlip.structure.composition.water_block

.. autofunction:: gs_gmlip.structure.composition.co2_block

.. autofunction:: gs_gmlip.structure.composition.co_block

.. autofunction:: gs_gmlip.structure.composition.oh_block

.. autofunction:: gs_gmlip.structure.composition.hydrogen_block

.. autofunction:: gs_gmlip.structure.composition.oxygen_block

.. autofunction:: gs_gmlip.structure.composition.sulfur_block

.. autofunction:: gs_gmlip.structure.composition.so2_block

.. autofunction:: gs_gmlip.structure.composition.formic_acid_block
