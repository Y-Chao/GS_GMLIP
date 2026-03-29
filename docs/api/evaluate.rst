Evaluate Module
===============

.. module:: gs_gmlip.evaluate

The evaluate module provides energy and force evaluators for candidate
structures, with support for MLIP (MACE) and DFT (VASP) backends.

Base Evaluator
--------------

.. autoclass:: gs_gmlip.evaluate.base.BaseEvaluator
   :members:

MLIP Evaluators
---------------

ASECalculatorEvaluator
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: gs_gmlip.evaluate.mlip.ase_calc.ASECalculatorEvaluator
   :members:
   :show-inheritance:

MACEEvaluator
^^^^^^^^^^^^^

.. autoclass:: gs_gmlip.evaluate.mlip.mace_eval.MACEEvaluator
   :members:
   :show-inheritance:

DFT Evaluators
--------------

VASPEvaluator
^^^^^^^^^^^^^

.. autoclass:: gs_gmlip.evaluate.dft.vasp.VASPEvaluator
   :members:
   :show-inheritance:
