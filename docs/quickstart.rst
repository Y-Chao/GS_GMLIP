Quick Start
===========

This guide walks through a minimal workflow: building a slab,
defining adsorbate blocks, and running a Genetic Algorithm search
with EMT (for testing — replace with MACE for production).

1. Build a slab
----------------

.. code-block:: python

   from ase.build import fcc111
   from gs_gmlip.structure import SlabAtoms

   slab = fcc111("Cu", size=(3, 3, 3), vacuum=10.0, periodic=True)
   sa = SlabAtoms.from_atoms(slab)
   sa.tag_slab_by_height(z_threshold=slab.positions[:, 2].mean())
   sa.set_slab_constraints()

2. Define the search region and building blocks
------------------------------------------------

.. code-block:: python

   from gs_gmlip.structure import BoxRegion, MolecularBlock
   from gs_gmlip.structure.composition import water_block, hydrogen_block

   z_top = sa.positions[:, 2].max()
   region = BoxRegion.from_atoms(sa, z_min=z_top + 1.5, z_max=z_top + 5.0)

   blocks = [water_block(mu=-14.0), hydrogen_block(mu=-3.4)]

3. Set up an evaluator
-----------------------

.. code-block:: python

   from ase.calculators.emt import EMT
   from gs_gmlip.evaluate.mlip.ase_calc import ASECalculatorEvaluator

   evaluator = ASECalculatorEvaluator(calculator=EMT())

4. Configure and run GA
------------------------

.. code-block:: python

   from gs_gmlip.search.ga import GARunner

   runner = GARunner(
       slab=sa,
       evaluator=evaluator,
       blocks=blocks,
       region=region,
       population_size=10,
       n_generations=20,
   )
   runner.run()
   results = runner.get_results()

5. Analyze results
-------------------

.. code-block:: python

   from gs_gmlip.analysis import StructureAnalyzer

   energies = [r.get_potential_energy() for r in results]
   analyzer = StructureAnalyzer(results, energies)
   print(analyzer.energy_landscape_summary())
