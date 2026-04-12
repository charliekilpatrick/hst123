.. _api_reference:

API reference
=============

This section documents the public Python API: the CLI entry point, the main
:class:`~hst123._pipeline.hst123` orchestration class, CLI registration in
:mod:`hst123.utils.options`, path helpers, optional HDF5 export for DOLPHOT
tables, and shared primitive utilities.

Entry point
-----------

.. autofunction:: hst123._pipeline.main

Pipeline class
--------------

The class :class:`~hst123._pipeline.hst123` holds pipeline state (options,
coordinates, input lists, reference path) and delegates to **primitives**
(``_fits``, ``_phot``, ``_astrom``, ``_dolphot``, ``_scrape_dolphot``) for
instrument-specific work.

.. autoclass:: hst123._pipeline.hst123
   :members:
   :exclude-members: __weakref__

CLI options and redo helpers
----------------------------

.. automodule:: hst123.utils.options
   :members: add_options, want_redo_astrometry, want_redo_astrodrizzle

Path helpers
------------

.. automodule:: hst123.utils.paths
   :members:
   :undoc-members:

Parallel BLAS / OpenMP guard
----------------------------

.. automodule:: hst123.utils.stdio
   :members:
   :undoc-members:

Work directory cleanup
----------------------

After AstroDrizzle or TweakReg, the pipeline can remove scratch FITS and replay
external logs into the session log. Use ``--keep-drizzle-artifacts`` to retain
intermediates.

.. automodule:: hst123.utils.workdir_cleanup
   :members: remove_files_matching_globs, cleanup_after_astrodrizzle, cleanup_after_tweakreg, remove_superseded_instrument_mask_reference_drizzle

DOLPHOT catalog → HDF5
----------------------

.. automodule:: hst123.utils.dolphot_catalog_hdf5
   :members:
   :undoc-members:

Data model
----------

.. automodule:: hst123.datamodels.hst_model
   :members:
   :undoc-members:

Primitives base class
---------------------

.. autoclass:: hst123.primitives.base.BasePrimitive
   :members:
   :show-inheritance:
