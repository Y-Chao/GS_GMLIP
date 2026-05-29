GS_GMLIP Documentation
======================

**Global Searching by Global Machine Learning Interatomic Potential**

A modular Python package for global structure searching of catalyst surfaces
under working conditions, accelerated by machine learning interatomic potentials (MLIP).

.. note::
   GS_GMLIP provides five global search methods — GA, GCMC, GOFEE, SSW,
   and Metadynamics — all with a unified interface and built on top of ASE.

Features
--------

- **Five search methods**: Genetic Algorithm, Grand Canonical Monte Carlo,
  GOFEE (GPR-accelerated), Stochastic Surface Walking, Metadynamics
- **MLIP acceleration**: MACE foundation model integration for fast energy/force evaluation
- **Catalyst-focused**: Designed for slab/adsorbate systems with molecular identity preservation
- **Unified interface**: All search methods share a common ``BaseSearcher`` API
- **ASE-native**: Built on ASE Atoms, calculators, and database infrastructure

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/structure
   api/interface
   api/evaluate
   api/search
   api/data
   api/analysis
   api/utils

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/tutorial_ga_gcmc

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
