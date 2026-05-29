Interface Module
================

.. module:: gs_gmlip.interface

The interface module provides the central structure object for gs_gmlip:
an :class:`~gs_gmlip.interface.core.Interface` couples a catalyst substrate
with appended adsorbates and clusters in a single object. It is backed by
pymatgen for surface analysis (neighbor-graph bonding, layer detection,
adsorbate/cluster classification) and exposes a conversion, store, and
slab-generator API on top of ASE Atoms.

Interface
---------

.. autoclass:: gs_gmlip.interface.core.Interface
   :members:

Analysis
--------

.. automodule:: gs_gmlip.interface.analysis
   :members:

Builders
--------

.. automodule:: gs_gmlip.interface.builders
   :members:
