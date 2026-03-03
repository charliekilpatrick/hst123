# Sphinx config for hst123. See https://www.sphinx-doc.org/en/master/usage/configuration.html
project = "hst123"
copyright = "2025"
author = "C. D. Kilpatrick"
try:
    from hst123 import __version__
    release = __version__
except Exception:
    try:
        from importlib.metadata import version
        release = version("hst123")
    except Exception:
        release = "0.0.0+unknown"
version = ".".join(release.split(".")[:2]) if release else "0.0"

extensions = []

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "hst123"
