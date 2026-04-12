##################################
Welcome to hst123's documentation!
##################################

**hst123** is a Python pipeline for *Hubble Space Telescope* data: query and
download from MAST, align images (TweakReg / JHAT), combine with AstroDrizzle,
optionally run DOLPHOT for PSF photometry, and scrape DOLPHOT catalogs at a
target position.

* :doc:`installation` — Python 3.12, Conda, DrizzlePac, optional DOLPHOT.
* :doc:`user_guide` — Work-directory layout, CLI overview, outputs, formats.
* :doc:`api` — ``hst123`` class, utilities, DOLPHOT HDF5 export.

Quick install
=============

From the repository root (see :doc:`installation` for detail):

.. code-block:: bash

   conda env create -f environment.yml
   conda activate hst123
   pip install drizzlepac
   pip install -e .

Build this documentation locally:

.. code-block:: bash

   pip install -e ".[docs]"
   cd docs && make html

The `documentation workflow
<https://github.com/charliekilpatrick/hst123/actions/workflows/documentation.yml>`__
publishes to `GitHub Pages <https://charliekilpatrick.github.io/hst123/>`__.

Repository layout
=================

- **hst123/** — ``_pipeline`` (CLI and orchestration), ``primitives/`` (FITS,
  photometry, astrometry, DOLPHOT, scrape), ``utils/`` (options, logging, paths,
  AstroDrizzle helpers, DOLPHOT utilities, bundled STScI WCS under
  ``utils.stwcs``), ``datamodels/``.
- **tests/** — ``pytest`` suite (markers: ``network``, ``dolphot``).
- **docs/** — Sphinx (reStructuredText + MyST Markdown): this site, changelog,
  zeropoints.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   user_guide
   changelog
   zeropoints
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
