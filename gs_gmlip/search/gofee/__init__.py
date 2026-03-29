"""Global Optimization with First-principles Energy Expressions (GOFEE).

Implements surrogate-assisted global optimization using Gaussian Process
Regression and acquisition function-based candidate selection.
"""

from gs_gmlip.search.gofee.runner import GOFEERunner

__all__ = ["GOFEERunner"]
