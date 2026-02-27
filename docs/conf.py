# Configuration file for the Sphinx documentation builder.
# See https://www.sphinx-doc.org/en/master/usage/configuration.html

project = "hst123"
copyright = "2025"
author = "C. D. Kilpatrick"
try:
    from importlib.metadata import version
    release = version("hst123")
except Exception:
    release = "0.0.0"
version = ".".join(release.split(".")[:2])  # short version for docs

extensions = []

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "hst123"
