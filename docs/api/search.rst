Search methods
==============

Global structure search. Implemented: genetic algorithm and grand-canonical Monte Carlo. New methods subclass ``BaseSearcher`` and are registered with ``gs_gmlip.jobs.register_searcher``.

Searcher base class
-------------------

.. automodule:: gs_gmlip.search.base
   :members:
   :show-inheritance:

Placement regions
-----------------

.. automodule:: gs_gmlip.search.region
   :members:
   :show-inheritance:

Genetic algorithm: runner
-------------------------

.. automodule:: gs_gmlip.search.ga.runner
   :members:
   :show-inheritance:

Genetic algorithm: operators
----------------------------

.. automodule:: gs_gmlip.search.ga.operators
   :members:
   :show-inheritance:

Genetic algorithm: population
-----------------------------

.. automodule:: gs_gmlip.search.ga.population
   :members:
   :show-inheritance:

Grand-canonical Monte Carlo: runner
-----------------------------------

.. automodule:: gs_gmlip.search.gcmc.runner
   :members:
   :show-inheritance:

Grand-canonical Monte Carlo: ensemble
-------------------------------------

.. automodule:: gs_gmlip.search.gcmc.ensemble
   :members:
   :show-inheritance:

Grand-canonical Monte Carlo: moves
----------------------------------

.. automodule:: gs_gmlip.search.gcmc.moves
   :members:
   :show-inheritance:
