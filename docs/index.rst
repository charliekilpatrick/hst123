##################################
Welcome to hst123's documentation!
##################################

**hst123** is a Python pipeline for Hubble Space Telescope data: download from
MAST, align (TweakReg / JHAT), drizzle, run DOLPHOT, and scrape photometry from
DOLPHOT catalogs. The code uses **NumPy-style docstrings** (see the
:doc:`user_guide`).

Installing hst123
=================

**Requirements:** Python 3.8+, Astropy, DrizzlePac, ``stwcs``, and related
dependencies (see ``pyproject.toml``). DOLPHOT is optional; use
``hst123-install-dolphot`` after activating your environment.

From the repository root:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate hst123
   pip install -e .

Editable install with documentation dependencies:

.. code-block:: bash

   pip install -e ".[docs]"

Build HTML locally:

.. code-block:: bash

   cd docs && make html

The `GitHub Actions documentation workflow
<https://github.com/charliekilpatrick/hst123/actions/workflows/documentation.yml>`__
publishes the built site to **GitHub Pages** (`hosted docs
<https://charliekilpatrick.github.io/hst123/>`__).

Package layout
==============

- **hst123/** — main package: ``_pipeline`` (CLI orchestration), ``primitives/``
  (FITS, photometry, astrometry, DOLPHOT, scrape), ``utils/`` (logging,
  options, WCS, AstroDrizzle helpers, DOLPHOT utilities), ``datamodels/``.
- **docs/** — Sphinx sources (this tree); changelog and zero points as Markdown.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   user_guide
   changelog
   zeropoints
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
