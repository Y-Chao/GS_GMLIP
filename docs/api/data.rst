Data Module
===========

.. module:: gs_gmlip.data

The data module provides database management, I/O utilities,
and workflow coordination.

Database
--------

.. autoclass:: gs_gmlip.data.database.CandidateDatabase
   :members:

.. autoclass:: gs_gmlip.data.database.PrepareDatabase
   :members:

I/O Utilities
-------------

.. autofunction:: gs_gmlip.data.io.read_structures

.. autofunction:: gs_gmlip.data.io.write_structures

.. autofunction:: gs_gmlip.data.io.structures_to_db

Workflow Manager
----------------

.. autoclass:: gs_gmlip.data.manager.WorkflowManager
   :members:
