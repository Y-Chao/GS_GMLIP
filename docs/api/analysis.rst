Analysis Module
===============

.. module:: gs_gmlip.analysis

The analysis module provides tools for post-search structure analysis,
trajectory convergence checking, and result visualization.

Structure Analysis
------------------

.. autoclass:: gs_gmlip.analysis.structure.StructureAnalyzer
   :members:

.. autofunction:: gs_gmlip.analysis.structure.pair_distribution

.. autofunction:: gs_gmlip.analysis.structure.fingerprint_distance

Trajectory Analysis
-------------------

.. autoclass:: gs_gmlip.analysis.trajectory.TrajectoryAnalyzer
   :members:

Visualization
-------------

.. autofunction:: gs_gmlip.analysis.visualization.save_best_structures

.. autofunction:: gs_gmlip.analysis.visualization.write_convergence_csv
