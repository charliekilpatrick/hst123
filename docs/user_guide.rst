.. _user_guide:

User guide
==========

Overview
--------

The pipeline is driven by the ``hst123`` command-line interface. Typical stages:

1. **Download** science data from MAST (``--download``).
2. **Align** images to a reference (TweakReg or JHAT); provenance is recorded in
   FITS headers (see the main README).
3. **Drizzle** combined stacks per filter/instrument as configured.
4. **DOLPHOT** (optional) for PSF photometry; reference PSFs are managed via
   ``hst123-install-dolphot``.
5. **Scrape** DOLPHOT catalogs at target coordinates (``--scrape-dolphot``).

Configuration uses ``hst123.utils.options`` (argparse) and ``hst123.settings``.

The :doc:`api` section lists selected modules; Sphinx uses **intersphinx** to
link to Python, NumPy, and Astropy where configured.

DOLPHOT catalog export
------------------------

To pack a DOLPHOT run (``dpXXXX`` base name and sidecars) into a single HDF5 file
with labeled columns and metadata, use
:mod:`hst123.utils.dolphot_catalog_hdf5` (requires ``h5py``; ``pip install .[hdf5]``).

Further reading
---------------

- Repository **README** (installation, CLI options, macOS OpenMP notes).
- :doc:`changelog` and :doc:`zeropoints`.
- :doc:`api` for module reference.
