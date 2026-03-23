# Changelog

Notable changes to hst123. Version numbering follows semantic versioning for full releases.

## Pre-releases (historical)

- **v1.00** (2019-02-07) — Base: download, tweakreg, drizzle, dolphot parameters.
- **v1.01** (2019-02-15) — Run dolphot and scrape dolphot output.
- **v1.02** (2019-02-22) — Fake star injection.
- **v1.03** (2019-06-02) — Drizzle-all options and option/syntax cleanup.
- **v1.04** (2020-02-10) — Python 3.7.
- **v1.05** (2020-03-04) — Sanitizing reference images for tweakreg/dolphot.
- **v1.06** (2020-03-07) — Archiving (`--archive`) for large-volume analysis.
- **v1.07** (2020-03-08) — Archiving updates; `construct_hst_archive.py` helper.

## Releases

- **Unreleased** — `hst123.utils.dolphot_catalog_hdf5`: parse DOLPHOT ``*.columns`` / ``*.param`` / ``*.info`` / ``*.data`` / ``*.warnings`` and write **HDF5** (`pip install h5py` or `pip install .[hdf5]`) with the full numeric catalog plus descriptive column names and JSON metadata (merged per ``imgNNNN`` when possible). ``get_dolphot_column`` uses ``parse_column_index_and_description`` for robust column index parsing. `install_psfs` ends with `relocate_all_legacy_psf_into_canonical_layout`: merges legacy `dolphot2.0/...` **`*.psf`** into the canonical tree for **ACS**, **WFC3** (`wfc3/data`, `wfc3/IR`, `wfc3/UVIS`), and **WFPC2** (`wfpc2/data`), including **sibling** `dolphot2.0/` next to `dolphot3.1/` (common conda layout). `relocate_acs_psf_into_canonical_layout` remains for ACS-only use. `hst123-install-dolphot` applies `apply_dolphot_source_patches` before `make`: patches upstream `dolphot.c` (`main` stack buffer 82→4096 bytes) to avoid macOS `sprintf` overflow / SIGTRAP on long absolute output paths; `--no-source-patches` opts out. `tests/test_dolphot_c_python_parity.py` documents C vs Python parity for calcsky, splitgroups, acsmask, wfc3mask, wfpc2mask (optional binaries + DOLPHOT tree; marker `dolphot_parity`). `hst123.utils.progress_log.LoggedProgress`: throttled `logger.info` progress lines (`#---` bar, %, counts, elapsed, est. total duration, remaining) for reuse; wired to calcsky stage 1 in `write_sky_fits_fallback` / `compute_sky_map_dolphot` (Python per-row updates; **Numba: one full parallel pass** then a single progress completion—no sequential batched kernels that ruined throughput). Disable with `HST123_CALCSKY_PROGRESS=0` or `HST123_PROGRESS_LOG=0`. Optional **`pip install .[perf]`** adds `numba`; `environment.yml` includes `numba` for conda users. `*drc.noise.fits` (sky copy for drizzled stacks) removed in `run_dolphot` and `get_dolphot_photometry` primitive cleanup with `respect_keep_artifacts=False` so they are still removed when `--keep-drizzle-artifacts` is set (not in per-filter astrodrizzle cleanup, which would delete prior filters’ noise). Same glob listed in `settings.pipeline_products` for scripted cleanup; see `tests/test_pipeline_cleanup_validation.py`. Removed legacy CLI `--make-clean`, `--no-large-reduction`, `--large-num` and `hst123.make_clean` / `check_large_reduction`. CI: workflow uses `pip install -e ".[test]"`; `_psf_already_satisfied` treats ACS WFC PAM as present if either `make_root` or `source_dir` has a valid `acs/data` payload; scrape-dolphot unit tests supply minimal `pipeline.options`. Pure-Python DOLPHOT `splitgroups` (`hst123.utils.dolphot_splitgroups`); `HST123_DOLPHOT_SPLITGROUPS_EXTERNAL=1` forces the C binary. `DOLPHOT_REQUIRED_SCRIPTS` no longer lists `splitgroups`. DOLPHOT installer: `ACS_WFC_PAM.tar.gz` extracts under `dolphot2.0/acs/data/`; files are now copied to `acs/data/` and `dolphot_acs_data_dir` prefers directories that contain both PAM FITS. Python `calcsky` port: vectorized second pass (summed-area table); Numba first pass uses precomputed annulus offsets and a per-row scratch buffer (no per-pixel `calloc`).

- **v1.1.0** (2025-03-02) — Post–v1.07: refactored pipeline (utils, primitives), setuptools-scm versioning, tests and docs updates.

*Full releases are added here when tagging. Run from the repo root:*
*`python scripts/update_changelog.py "Brief summary of changes"`*
