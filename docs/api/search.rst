Search methods
==============

Global structure search. Implemented: genetic algorithm and grand-canonical Monte Carlo. New methods subclass ``BaseSearcher`` and are registered with ``gm2_surf.jobs.register_searcher``.

Searcher base class
-------------------

.. automodule:: gm2_surf.search.base
   :members:
   :show-inheritance:

Placement regions
-----------------

.. automodule:: gm2_surf.search.region
   :members:
   :show-inheritance:

Genetic algorithm: runner
-------------------------

.. automodule:: gm2_surf.search.ga.runner
   :members:
   :show-inheritance:

Genetic algorithm: operators
----------------------------

.. automodule:: gm2_surf.search.ga.operators
   :members:
   :show-inheritance:

Genetic algorithm: population
-----------------------------

.. automodule:: gm2_surf.search.ga.population
   :members:
   :show-inheritance:

Grand-canonical Monte Carlo: runner
-----------------------------------

.. automodule:: gm2_surf.search.gcmc.runner
   :members:
   :show-inheritance:

Grand-canonical Monte Carlo: ensemble
-------------------------------------

.. automodule:: gm2_surf.search.gcmc.ensemble
   :members:
   :show-inheritance:

Grand-canonical Monte Carlo: moves
----------------------------------

.. automodule:: gm2_surf.search.gcmc.moves
   :members:
   :show-inheritance:
