.. _user_guide:

User guide
==========

Overview
--------

**hst123** is a command-line and Python API for *Hubble Space Telescope* data: query and download from MAST, align with TweakReg or JHAT, combine with AstroDrizzle, optionally run DOLPHOT for PSF photometry, and scrape DOLPHOT catalogs at a target coordinate. Configuration is driven by :mod:`hst123.utils.options` (``argparse``) and defaults in :mod:`hst123.settings`.

Typical workflow:

#. **Ingest** — Copy or download calibrated science FITS into ``<work-dir>/raw`` and link or copy into the processing tree (see :ref:`work_directory_layout`).
#. **Filter inputs** — Quality cuts (EXPFLAG, exposure time, date range, filters) via CLI flags.
#. **Table** — Build a per-exposure table (:meth:`~hst123._pipeline.hst123.input_list`) with visit/filter metadata.
#. **Reference** — Choose or build a drizzled reference image (:meth:`~hst123._pipeline.hst123.handle_reference`, :meth:`~hst123._pipeline.hst123.pick_reference`).
#. **Align** — TweakReg or JHAT (``--align-with``); success can be recorded in FITS headers for skip-on-rerun behavior.
#. **Drizzle** — Per-epoch and/or consolidated stacks (:meth:`~hst123._pipeline.hst123.run_astrodrizzle`, :meth:`~hst123._pipeline.hst123.drizzle_all`).
#. **DOLPHOT** (optional) — Mask, split groups, sky subtraction, photometry (:meth:`~hst123._pipeline.hst123.prepare_dolphot`, :meth:`~hst123._pipeline.hst123.run_dolphot`).
#. **Scrape** — Extract photometry at the pipeline coordinate from DOLPHOT output (``--scrape-dolphot``).

The CLI entry point is :func:`~hst123._pipeline.main`; programmatic use starts with :class:`~hst123._pipeline.hst123` and :meth:`~hst123._pipeline.hst123.handle_args`.

.. _work_directory_layout:

Work directory layout
---------------------

Set ``--work-dir`` to an **absolute** path for reproducible behavior (the process may ``chdir`` into subdirectories during alignment).

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Location
     - Role
   * - ``<work-dir>/raw/``
     - Science FITS from MAST or archive; primary cache for downloaded data.
   * - ``<work-dir>/workspace/``
     - Calibrated inputs used for alignment and drizzle, symlinks/copies from ``raw``, and many intermediate TweakReg/AstroDrizzle products. The process working directory is often ``workspace/`` during astrometry.
   * - ``<work-dir>/drizzle/``
     - Consolidated outputs from ``--drizzle-all`` (per-instrument/filter/epoch naming), **not** under ``workspace/``.
   * - ``<work-dir>/logs/``
     - Session log file (mirror of console output for that run).
   * - ``<work-dir>/.mast_download_staging/``
     - Temporary astroquery download tree; removed after files move to ``raw`` or archive.
   * - ``<work-dir>/.hst123_runfiles/``
     - Short-lived replay of external tool logs into the session log (e.g. AstroDrizzle, photeq); typically deleted after replay unless ``--keep-drizzle-artifacts`` applies.

The main **reference** drizzled image (e.g. ``*.ref_*.drc.fits``) and DOLPHOT ``chip*.fits`` / ``chip*.sky.fits`` products are anchored at the **base** ``work-dir`` level; see pipeline helpers in :mod:`hst123.utils.paths`.

Supported instruments and file types
--------------------------------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Instrument
     - File types
   * - WFPC2
     - ``c0m.fits``, ``c1m.fits`` (both)
   * - ACS/WFC
     - ``flc.fits``
   * - ACS/HRC
     - ``flt.fits``
   * - WFC3/UVIS
     - ``flc.fits``
   * - WFC3/IR
     - ``flt.fits``

Photometric zeropoints and AB conversion are described in :doc:`zeropoints`.

Command-line usage
------------------

.. code-block:: bash

   hst123 <ra> <dec> [options]

Examples:

.. code-block:: bash

   hst123 12:30:00 -45.0 --download --work-dir /path/to/run
   hst123 12:22:56.15 +15:49:34.18 --download --drizzle-all --run-dolphot --work-dir /path/to/run

Use ``hst123 --help`` for the full option list. Important groups:

* **Geometry** — ``--work-dir``, ``--raw-dir``, ``--archive``, ``--by-visit``
* **Data selection** — ``--before``, ``--after``, ``--only-filter``, ``--only-wide``
* **Reference** — ``--reference``, ``--reference-filter``, ``--avoid-wfpc2``
* **Alignment** — ``--align-with`` (``tweakreg`` / ``jhat``), ``--skip-tweakreg``, ``--tweak-search``, ``--redo-astrometry``
* **Drizzle** — ``--drizzle-all``, ``--drizzle-dim``, ``--redrizzle``, ``--redo-astrodrizzle``, ``--keep-drizzle-artifacts``
* **Parallelism** — ``--max-cores`` sets AstroDrizzle ``num_cores`` and the DOLPHOT prep thread pool (and parallel MAST downloads when applicable).
* **DOLPHOT** — ``--run-dolphot``, ``--dolphot``, ``--scrape-dolphot``, etc.

Redo and skip behavior
~~~~~~~~~~~~~~~~~~~~~~~~

* ``--redo`` implies both ``--redo-astrometry`` and ``--redo-astrodrizzle``.
* Without redo flags, alignment and drizzle can **skip** when prior success metadata in headers matches the requested reference and method (unless ``--clobber`` forces overwrites).

See :func:`hst123.utils.options.want_redo_astrometry` and :func:`hst123.utils.options.want_redo_astrodrizzle`.

Output data formats
-------------------

FITS products
~~~~~~~~~~~~~

* **Input/output science data** — Standard MAST-calibrated ``flt``/``flc``/``c0m``/``c1m`` FITS; headers are updated for WCS alignment and drizzle provenance where applicable.
* **Drizzled images** — AstroDrizzle products (e.g. ``*_drz.fits``, ``*_drc.fits`` depending on configuration); combined weight maps and header keywords document inputs.

DOLPHOT text products
~~~~~~~~~~~~~~~~~~~~~

A DOLPHOT run uses a base name such as ``dp0000`` (visit suffixes may apply). Sidecar files include:

* ``dpXXXX.columns`` — Human-readable column descriptions (one per measurement column).
* ``dpXXXX.param`` — DOLPHOT parameter file.
* ``dpXXXX.data`` — Numeric catalog (whitespace-separated rows).
* ``dpXXXX.info``, ``dpXXXX.warnings`` — Run metadata and messages.

Optional **HDF5** export with labeled columns and JSON metadata: :mod:`hst123.utils.dolphot_catalog_hdf5` (requires ``h5py``; ``pip install .[hdf5]`` or ``pip install h5py``).

Logs
~~~~

User-facing messages use the ``hst123`` logging namespace. After ``--work-dir`` is set, a session file is written under ``<work-dir>/logs/``.

Python API
----------

Import the package and construct the pipeline class (see :doc:`api`):

.. code-block:: python

   from hst123 import hst123
   p = hst123()
   args = p.handle_args(p.add_options())

Programmatic workflows mirror the CLI: set ``p.options['args']``, populate ``p.input_images`` or call ``get_productlist`` / ``download_files`` as needed, then run alignment and drizzle methods.

Further reading
---------------

* :doc:`installation` — Environment and DOLPHOT setup
* :doc:`changelog` — Release history
* :doc:`zeropoints` — Magnitude system and zeropoints
* :doc:`api` — Module and class reference
