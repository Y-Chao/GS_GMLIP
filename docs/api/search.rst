Search Module
=============

.. module:: gs_gmlip.search

The search module implements five global structure search methods, all
inheriting from a common ``BaseSearcher`` interface.

Base Searcher
-------------

.. autoclass:: gs_gmlip.search.base.BaseSearcher
   :members:

Genetic Algorithm (GA)
----------------------

.. autoclass:: gs_gmlip.search.ga.runner.GARunner
   :members:
   :show-inheritance:

Start Generator
^^^^^^^^^^^^^^^

.. autoclass:: gs_gmlip.search.ga.startgenerator.StartGenerator
   :members:

Population
^^^^^^^^^^

.. autoclass:: gs_gmlip.search.ga.population.Population
   :members:

Operators
^^^^^^^^^

.. autoclass:: gs_gmlip.search.ga.operators.OffspringCreator
   :members:

.. autoclass:: gs_gmlip.search.ga.operators.CutAndSplicePairing
   :members:
   :show-inheritance:

.. autoclass:: gs_gmlip.search.ga.operators.RattleMutation
   :members:
   :show-inheritance:

.. autoclass:: gs_gmlip.search.ga.operators.PermutationMutation
   :members:
   :show-inheritance:

.. autoclass:: gs_gmlip.search.ga.operators.MirrorMutation
   :members:
   :show-inheritance:

.. autoclass:: gs_gmlip.search.ga.operators.OperationSelector
   :members:

Slab Operators
^^^^^^^^^^^^^^

.. autoclass:: gs_gmlip.search.ga.slab_operators.CutSpliceSlabCrossover
   :members:
   :show-inheritance:

.. autoclass:: gs_gmlip.search.ga.slab_operators.RandomCompositionMutation
   :members:
   :show-inheritance:

.. autoclass:: gs_gmlip.search.ga.slab_operators.RandomElementMutation
   :members:
   :show-inheritance:

Comparators
^^^^^^^^^^^

.. autoclass:: gs_gmlip.search.ga.comparators.BaseComparator
   :members:

.. autoclass:: gs_gmlip.search.ga.comparators.InteratomicDistanceComparator
   :members:
   :show-inheritance:

.. autoclass:: gs_gmlip.search.ga.comparators.EnergyComparator
   :members:
   :show-inheritance:

.. autoclass:: gs_gmlip.search.ga.comparators.CompositionComparator
   :members:
   :show-inheritance:

.. autoclass:: gs_gmlip.search.ga.comparators.SequentialComparator
   :members:
   :show-inheritance:

Grand Canonical Monte Carlo (GCMC)
-----------------------------------

.. autoclass:: gs_gmlip.search.gcmc.runner.GCMCRunner
   :members:
   :show-inheritance:

.. autoclass:: gs_gmlip.search.gcmc.ensemble.GCEnsemble
   :members:

Moves
^^^^^

.. autoclass:: gs_gmlip.search.gcmc.moves.MCMove
   :members:

.. autoclass:: gs_gmlip.search.gcmc.moves.InsertMove
   :members:
   :show-inheritance:

.. autoclass:: gs_gmlip.search.gcmc.moves.DeleteMove
   :members:
   :show-inheritance:

.. autoclass:: gs_gmlip.search.gcmc.moves.SwapMove
   :members:
   :show-inheritance:

.. autoclass:: gs_gmlip.search.gcmc.moves.DisplaceMove
   :members:
   :show-inheritance:

GOFEE
-----

.. autoclass:: gs_gmlip.search.gofee.runner.GOFEERunner
   :members:
   :show-inheritance:

.. autoclass:: gs_gmlip.search.gofee.surrogate.SurrogateModel
   :members:

Acquisition Functions
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: gs_gmlip.search.gofee.acquisition.lower_confidence_bound

.. autofunction:: gs_gmlip.search.gofee.acquisition.expected_improvement

.. autofunction:: gs_gmlip.search.gofee.acquisition.probability_of_improvement

.. autofunction:: gs_gmlip.search.gofee.surrogate.get_fingerprint

Stochastic Surface Walking (SSW)
---------------------------------

.. autoclass:: gs_gmlip.search.ssw.runner.SSWRunner
   :members:
   :show-inheritance:

.. autoclass:: gs_gmlip.search.ssw.walker.SSWWalker
   :members:

.. autoclass:: gs_gmlip.search.ssw.minima_hopping.MinimaHopping
   :members:

Metadynamics
------------

.. autoclass:: gs_gmlip.search.metadynamics.runner.MetadynamicsRunner
   :members:
   :show-inheritance:

Collective Variables
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: gs_gmlip.search.metadynamics.collective_vars.CollectiveVariable
   :members:

.. autoclass:: gs_gmlip.search.metadynamics.collective_vars.Distance
   :members:
   :show-inheritance:

.. autoclass:: gs_gmlip.search.metadynamics.collective_vars.Angle
   :members:
   :show-inheritance:

.. autoclass:: gs_gmlip.search.metadynamics.collective_vars.Torsion
   :members:
   :show-inheritance:

.. autoclass:: gs_gmlip.search.metadynamics.collective_vars.CoordinationNumber
   :members:
   :show-inheritance:

.. autoclass:: gs_gmlip.search.metadynamics.collective_vars.Position
   :members:
   :show-inheritance:

.. autofunction:: gs_gmlip.search.metadynamics.collective_vars.build_cvs_from_config

PLUMED Interface
^^^^^^^^^^^^^^^^

.. autoclass:: gs_gmlip.search.metadynamics.plumed_interface.PlumedCalculator
   :members:

.. autofunction:: gs_gmlip.search.metadynamics.plumed_interface.generate_plumed_input

.. autofunction:: gs_gmlip.search.metadynamics.plumed_interface.extract_minima_from_fes
