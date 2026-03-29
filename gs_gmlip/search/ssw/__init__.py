"""Stochastic Surface Walking (SSW) method for global optimization.

Implements soft-mode-following walks on the PES to discover
new minima and transition states.
"""

from gs_gmlip.search.ssw.runner import SSWRunner

__all__ = ["SSWRunner"]
