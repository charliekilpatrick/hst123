# -*- coding: utf-8 -*-
# hst123 documentation configuration (Sphinx + Read the Docs theme).
#
# This file is execfile()d with the current directory set to its containing dir.

from __future__ import annotations

import os
import re
import sys

# Package import path (repo root)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from hst123 import __version__ as hst123_version
except Exception:
    try:
        from importlib.metadata import version

        hst123_version = version("hst123")
    except Exception:
        hst123_version = "0.0.0+unknown"

# -- General ---------------------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
]

autosummary_generate = True
autosummary_imported_members = False

# numpydoc extension
numpydoc_show_class_members = False
numpydoc_use_blockquotes = True
numpydoc_use_plots = False

try:
    from numpydoc import docscrape_sphinx

    parts = re.split(r"[\(\)|]", docscrape_sphinx.IMPORT_MATPLOTLIB_RE)[1:-1]
except Exception:
    pass
else:
    parts.extend(("fig.show()", "plot.show()"))
    docscrape_sphinx.IMPORT_MATPLOTLIB_RE = r"\b({})\b".format("|".join(parts))

templates_path = ["_templates"]
# MyST-Parser registers the ``myst`` source type for ``.md`` (see MyST docs).
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst",
}

master_doc = "index"
project = "hst123"
copyright = "2025, C. D. Kilpatrick"
author = "C. D. Kilpatrick"
version = ".".join(hst123_version.split(".")[:2]) if hst123_version else "0.0"
release = hst123_version

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "build"]

pygments_style = "sphinx"
todo_include_todos = False

# Autodoc: avoid importing optional heavy stacks when building on minimal envs
autodoc_mock_imports = [
    "astroquery",
    "astroquery.mast",
    "astroscrappy",
    "drizzlepac",
    "drizzlepac.tweakreg",
    "drizzlepac.astrodrizzle",
    "drizzlepac.catalogs",
    "drizzlepac.photeq",
    "jhat",
]

# -- HTML ------------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = f"hst123 {release}"
htmlhelp_basename = "hst123doc"

# -- myst-parser -----------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "substitution",
]

# -- Intersphinx -----------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}

# -- LaTeX / manual (optional) ----------------------------------------------

latex_documents = [
    (master_doc, "hst123.tex", "hst123 Documentation", author, "manual"),
]

man_pages = [(master_doc, "hst123", "hst123 Documentation", [author], 1)]
