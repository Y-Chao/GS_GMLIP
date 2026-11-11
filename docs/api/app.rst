Interactive app
===============

The server side of the interactive app: the session that holds the windows, the scene sent to the viewer, and the JSON API.

Session
-------

.. automodule:: gs_gmlip.app.session
   :members:
   :show-inheritance:

Viewer scene
------------

.. automodule:: gs_gmlip.app.scene
   :members:
   :show-inheritance:

JSON API server
---------------

``gs_gmlip.app.server`` is a thin FastAPI layer over the session (``create_app``,
``run_app``); it needs the ``app`` extra and is therefore not auto-documented here.
