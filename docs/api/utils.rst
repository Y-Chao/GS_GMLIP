Utils Module
============

.. module:: gs_gmlip.utils

``gs_gmlip.utils`` provides a `rich <https://rich.readthedocs.io>`_-based
**levelled printing system** used throughout the package to control terminal
output verbosity and style.

Overview
--------

The system has three layers:

1. **PrintConfig** (global singleton) — holds the current verbosity level and
   Console settings (e.g. output width).
2. **Module-level helpers** — ``get_config()``, ``get_console_settings()``,
   and ``update_print_settings()`` give uniform access to the singleton.
3. **Printer** (main interface) — wraps four output forms: plain text, section
   headers, tables, and panels.

Verbosity levels
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Constant
     - Value
     - Description
   * - ``Printer.STANDARD``
     - ``0``
     - Default level; white text; always visible.
   * - ``Printer.DEBUG``
     - ``1``
     - Debug level; orange text; suppressed unless printer verbosity ≥ 1.

A message is printed only when ``message_verbosity >= printer._verbosity``.

Quick start
-----------

Basic usage
~~~~~~~~~~~

.. code-block:: python

   from gs_gmlip.utils import Printer

   printer = Printer()            # verbosity=STANDARD (0)
   printer("Search started")      # white output
   printer.debug("n_candidates: 42")  # suppressed — _verbosity=0 < DEBUG=1

Enable debug output
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from gs_gmlip.utils import Printer

   printer = Printer(verbosity=Printer.DEBUG)
   printer("Search started")          # white
   printer.debug("n_candidates: 42")  # orange

Structured output
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from gs_gmlip.utils import Printer

   printer = Printer()

   # Section header (rule)
   printer.print_header("Generation 1")

   # Table
   printer.print_table(
       table_column=["Structure", "Energy (eV)", "Origin"],
       table_row=[
           ("candidate_001", "-123.45", "crossover"),
           ("candidate_002", "-122.80", "mutation"),
       ],
       title="Candidates",
   )

   # Panel
   printer.print_panel(
       "Search converged after 200 evaluations.",
       panel_title="Done",
   )

Adjusting output width
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from gs_gmlip.utils import update_print_settings, Printer

   update_print_settings(width=80)   # affects all Printers instantiated after this
   printer = Printer()

   # Equivalent static method on Printer
   Printer.update_settings(width=100)

Integration pattern
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from gs_gmlip.utils import Printer

   class MySearcher:
       def __init__(self, verbose: bool = False):
           level = Printer.DEBUG if verbose else Printer.STANDARD
           self.printer = Printer(verbosity=level)

       def step(self):
           self.printer("Running one step")
           self.printer.debug("internal state: ...")

       def report(self, results):
           self.printer.print_header("Results")
           self.printer.print_table(
               ["Rank", "Energy"],
               [(str(i), f"{e:.3f}") for i, e in enumerate(results)],
           )

Notes
-----

- ``Printer.console_kwargs`` is snapshotted from the global config at
  ``__init__`` time.  Calling ``update_print_settings()`` **after** a Printer
  is created will not affect that instance; update before instantiation, or use
  ``Printer.update_settings()`` before creating the instance.
- The ``verbosity`` property returns ``max(self._verbosity, _config.verbosity)``.
  Raising ``_config.verbosity`` globally promotes all Printers without touching
  individual instances.
- ``print_table`` passes ``**table_kwargs`` directly to ``rich.table.Table``,
  so all Rich Table options (``title``, ``show_lines``, ``border_style``, etc.)
  are available.

API Reference
-------------

PrintConfig
~~~~~~~~~~~

.. autoclass:: gs_gmlip.utils.PrintConfig
   :members:
   :undoc-members:

Module-level helpers
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: gs_gmlip.utils.get_config

.. autofunction:: gs_gmlip.utils.get_console_settings

.. autofunction:: gs_gmlip.utils.update_print_settings

Printer
~~~~~~~

.. autoclass:: gs_gmlip.utils.Printer
   :members:
   :undoc-members:
