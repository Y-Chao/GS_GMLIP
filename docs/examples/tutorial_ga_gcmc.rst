Tutorial: GA and GCMC on Cu(111)
==================================

This is an interactive Jupyter notebook tutorial demonstrating the
core workflow of GS_GMLIP.

You can find the notebook at ``examples/tutorial_ga_gcmc_workflow.ipynb``
in the repository root. Open it with Jupyter to run interactively:

.. code-block:: bash

   jupyter notebook examples/tutorial_ga_gcmc_workflow.ipynb

The tutorial covers:

1. **Build a Cu(111) slab** — Using ``SlabAtoms`` with proper slab/adsorbate tagging
2. **Define search region** — ``BoxRegion`` above the slab for adsorbate placement
3. **Set up building blocks** — ``MolecularBlock`` for H and O atoms
4. **Configure evaluator** — ``ASECalculatorEvaluator`` wrapping EMT (demo) or MACE (production)
5. **Run Genetic Algorithm** — Manual step-through of ``StartGenerator`` and evaluation
6. **Run GCMC** — ``GCEnsemble`` + ``InsertMove``/``DeleteMove``/``DisplaceMove``
7. **Analyze results** — ``StructureAnalyzer`` and ``TrajectoryAnalyzer``
8. **Visualize structures** — Pair distributions, bond analysis, structure comparison
