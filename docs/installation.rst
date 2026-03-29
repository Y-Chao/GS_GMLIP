Installation
============

Requirements
------------

- Python >= 3.12
- ASE >= 3.23.0
- NumPy >= 1.26.0
- SciPy >= 1.12.0
- scikit-learn >= 1.4.0
- PyYAML >= 6.0

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/your-org/GS_GMLIP.git
   cd GS_GMLIP
   pip install -e .

With MLIP support (MACE)
-------------------------

.. code-block:: bash

   pip install -e ".[mlip]"

With Metadynamics support (PLUMED)
----------------------------------

.. code-block:: bash

   pip install -e ".[plumed]"

Install everything (including dev tools)
----------------------------------------

.. code-block:: bash

   pip install -e ".[all]"

Verify installation
-------------------

.. code-block:: python

   import gs_gmlip
   print(gs_gmlip.__version__)  # 1.0.0
