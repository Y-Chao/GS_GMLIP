"""Plugin loading: add search algorithms, tasks, and calculators without editing gs_gmlip.

A plugin is ordinary Python that calls the registration functions:

- ``gs_gmlip.jobs.register_searcher`` for a ``BaseSearcher`` subclass (a new
  structure-search algorithm),
- ``gs_gmlip.jobs.register_task`` for any other job kind,
- ``gs_gmlip.evaluator.register_calculator`` for an energy engine (any ASE
  calculator).

Plugins are discovered in two places, and loaded by both the app and every
job worker so a registered kind is available wherever a job runs:

1. Installed packages exposing an entry point in the ``gs_gmlip.plugins``
   group. The entry point may name a module (imported) or a callable (called
   with no arguments)::

       [project.entry-points."gs_gmlip.plugins"]
       my_search = "my_package.gs_plugin"

2. Files ``<project>/plugins/*.py``, executed in name order. This is the
   quick route for a single script::

       # my_project/plugins/random_search.py
       from gs_gmlip.jobs import register_searcher
       from gs_gmlip.search.base import BaseSearcher

       class RandomSearch(BaseSearcher):
           def __init__(self, evaluator, start, workdir, seed=0):
               super().__init__(evaluator, workdir, seed)
               self.start = start

           def setup(self):
               self._is_setup = True

           def step(self):
               candidate = self.start.copy()
               candidate.interface.rattle(0.2, seed=self._step)
               self._step += 1
               self.evaluator.evaluate(candidate, relax=True)
               return {"step": self._step}

       register_searcher(
           "random_search",
           lambda evaluator, start, params, workdir: RandomSearch(
               evaluator, start, workdir, seed=params["seed"]
           ),
           defaults={"n_steps": 10, "fmax": 0.1, "relax_steps": 50, "seed": 0},
           description="Rattle and relax",
       )
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from importlib.metadata import entry_points
from pathlib import Path

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "gs_gmlip.plugins"
PLUGIN_DIR = "plugins"

_loaded: set[str] = set()


def _load_entry_points() -> list[str]:
    names = []
    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        key = f"entry_point:{entry_point.name}"
        if key in _loaded:
            continue
        try:
            target = entry_point.load()
            if callable(target):
                target()
        except Exception:  # noqa: BLE001 - a broken plugin must not take the app down
            logger.exception("Failed to load plugin entry point %r", entry_point.name)
            continue
        _loaded.add(key)
        names.append(entry_point.name)
    return names


def _load_file(path: Path) -> bool:
    module_name = f"gs_gmlip_plugin_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return False
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:  # noqa: BLE001 - a broken plugin must not take the app down
        sys.modules.pop(module_name, None)
        logger.exception("Failed to load plugin file %s", path)
        return False
    return True


def load_plugins(project_root: str | Path | None = None) -> list[str]:
    """Load plugins from entry points and from ``<project_root>/plugins/*.py``.

    Safe to call repeatedly: each plugin is loaded once per process. A plugin
    that raises is logged and skipped.

    Args:
        project_root: Project directory to scan for a ``plugins`` folder.

    Returns:
        Names of the plugins loaded by this call.
    """
    names = _load_entry_points()
    if project_root is not None:
        folder = Path(project_root) / PLUGIN_DIR
        for path in sorted(folder.glob("*.py")) if folder.is_dir() else ():
            key = f"file:{path.resolve()}"
            if key in _loaded:
                continue
            if _load_file(path):
                _loaded.add(key)
                names.append(path.stem)
    return names
