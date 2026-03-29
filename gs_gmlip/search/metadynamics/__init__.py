"""Metadynamics for enhanced sampling of rare events.

Wraps PLUMED via the ASE-PLUMED interface for biased MD
along collective variables to explore the PES.
"""

from gs_gmlip.search.metadynamics.runner import MetadynamicsRunner

__all__ = ["MetadynamicsRunner"]
