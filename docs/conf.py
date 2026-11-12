# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "GM2-Surf"
copyright = "2026, Chao Yang"
author = "Chao Yang"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- Autodoc configuration ---------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autosummary_generate = True

# Third-party packages (pymatgen, starlette) guard some type imports behind
# TYPE_CHECKING or optional dependencies; sphinx-autodoc-typehints cannot
# resolve those names. They are not errors in this package's documentation.
suppress_warnings = [
    "sphinx_autodoc_typehints.forward_reference",
    "sphinx_autodoc_typehints.guarded_import",
]

# -- Napoleon configuration --------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
# Render ``Attributes:`` sections as fields instead of separate attribute
# entries, which would duplicate the dataclass members autodoc already lists.
napoleon_use_ivar = True

# -- Intersphinx mapping -----------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    # ASE is not listed: its documentation moved and no longer serves an
    # object inventory at a stable address, which fails the build under -W.
}

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
}
html_static_path = ["_static"]
html_logo = "../assets/gm2-surf_icon.svg"
html_favicon = "../assets/png/gm2-surf_icon_64.png"

# -- nbsphinx configuration -------------------------------------------------
nbsphinx_execute = "never"
