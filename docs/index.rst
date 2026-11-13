GS_GMLIP Documentation
======================

**Global Searching by Global Machine Learning Interatomic Potential**

A modular Python package for global structure searching of catalyst surfaces
under working conditions, accelerated by machine learning interatomic potentials (MLIP).

.. note::
   Implemented search methods are the genetic algorithm (GA) and grand-canonical
   Monte Carlo (GCMC). Further methods (GOFEE, SSW, minima hopping, metadynamics)
   plug in through the ``BaseSearcher`` interface.

Features
--------

- **Interactive app**: build bulk, surface, and molecule models in the browser,
  run or prepare simulations, and browse every intermediate structure
- **Search methods**: genetic algorithm and grand-canonical Monte Carlo behind a
  common ``BaseSearcher`` API, extensible through plugins
- **Any ASE calculator**: foundation potentials (MatGL, MACE), DFT (VASP), or
  classical potentials through one calculator registry
- **Catalyst-focused**: slab/adsorbate systems with molecular identity preservation
- **Provenance**: every structure is stored with its origin, role, and level of theory

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

   api/interface
   api/evaluate
   api/search
   api/data
   api/jobs
   api/app
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
