# hst123

[![Build and Test](https://github.com/charliekilpatrick/hst123/actions/workflows/build-test.yml/badge.svg)](https://github.com/charliekilpatrick/hst123/actions/workflows/build-test.yml)
[![Documentation](https://github.com/charliekilpatrick/hst123/actions/workflows/documentation.yml/badge.svg)](https://github.com/charliekilpatrick/hst123/actions/workflows/documentation.yml)
[![Documentation site](https://img.shields.io/badge/documentation-GitHub%20Pages-4b32b3)](https://charliekilpatrick.github.io/hst123/)

Pipeline for HST data: download from MAST, align (tweakreg/jhat), drizzle, run DOLPHOT, and scrape photometry. Optimized for point-source photometry across multiple visits and filters.

**Alignment skip / provenance:** After a successful TweakReg or JHAT run, the primary header stores **`HIERARCH HST123 ALIGNOK`**, **`ALIGNMT`** (method), and **`ALIGNRF`** (normalized reference id). A later run **skips** re-aligning a file when those match the requested method and reference (**`--clobber`** forces re-alignment). Logs record alignment outcome, reference, and method.

**Issues:** Report bugs and feature requests via the project issue tracker. Other questions or publications using hst123: [Contact](#contact).

---

## Repository status

- **Python:** 3.8+
- **Versioning:** From git tags (setuptools-scm); `hst123 --version`
- **Layout:** Main pipeline in `hst123/_pipeline.py`; helpers in `hst123/primitives/` (FITS, photometry, astrometry, DOLPHOT, scrape) and `hst123/utils/` (options, logging, display, visit, WCS)
- **Tests:** `pytest` in `tests/`; optional markers `network`, `dolphot` (see pyproject.toml)
- **Docs:** Sphinx sources under `docs/` (MyST Markdown + reStructuredText, Read the Docs theme). Build: `pip install -e ".[docs]"` then `cd docs && make html`. **Hosted:** [GitHub Pages](https://charliekilpatrick.github.io/hst123/) (from the [Documentation workflow](https://github.com/charliekilpatrick/hst123/actions/workflows/documentation.yml)).

---

## Installation

**1. Environment (recommended: Conda)**  
From the repo root:

```bash
conda env create -f environment.yml
conda activate hst123
```

**2. Pipeline and heavy deps**

```bash
pip install drizzlepac stwcs
pip install -e .
```

The `hst123` command is available. Download, alignment, and drizzling work without DOLPHOT.

**3. DOLPHOT (only for `--run-dolphot`)**  
DOLPHOT is external; not bundled. With **conda activated**, the helper downloads sources and PSF/PAM reference data under **`$CONDA_PREFIX/opt/hst123-dolphot`**, runs **`make`**, and symlinks **`dolphot`**, **`calcsky`**, and optional mask/split tools into **`$CONDA_PREFIX/bin`**. **Masking** (`acsmask` / `wfc3mask` / `wfpc2mask`) and **`splitgroups`** are implemented in Python by default (see **`hst123.utils.dolphot_mask`**, **`hst123.utils.dolphot_splitgroups`**); set **`HST123_DOLPHOT_MASK_EXTERNAL=1`** or **`HST123_DOLPHOT_SPLITGROUPS_EXTERNAL=1`** to force the C binaries.

```bash
conda activate hst123   # or your env with hst123 installed
hst123-install-dolphot
```

Without conda (or to choose a location), pass **`--dolphot-dir /path/to/dolphot`**; use **`--no-link-conda-bin`** and **`export PATH="/path/to/dolphot:$PATH"`** if you do not want links in the env `bin/`. Manual install: [americano.dolphinsim.com/dolphot](http://americano.dolphinsim.com/dolphot/).

**`calcsky` and drizzled FITS:** Per the [DOLPHOT User’s Guide](http://americano.dolphinsim.com/dolphot/dolphot.pdf) (§4.1 *calcsky*), the sky map is built with an iterative mean and σ-rejection on an annular sample grid, then box-averaged (see upstream **`calcsky.c`** **`getsky`**). **NaN** and **multi-extension** drizzle products can make the `calcsky` binary abort; the pipeline runs it on a temporary **single-HDU** sanitized copy when possible. For **very large** mosaics (default **> 6×10⁶** pixels), the binary is **skipped** (**`HST123_CALCSKY_MAX_PIXELS`**; **`0`** = never skip). If `calcsky` fails or is skipped, **`write_sky_fits_fallback`** in **`hst123.utils.dolphot_sky`** uses a **Python port of that same `getsky` algorithm** (defaults for pixel inclusion match DOLPHOT **`param/fits.param`**: BADPIX/SATURATE, else **−1** / **65535**). The second pass (box smooth) is **vectorized** via a summed-area table; with **`numba`** installed (`pip install .[perf]` or the conda **`environment.yml`**), the first pass uses **one parallel JIT** pass and a **per-row scratch buffer** (no per-pixel allocations). Progress logging uses a **single** stage-1 completion line (batched row updates would serialize work and be much slower). Optional: **`HST123_CALCSKY_LEGACY=1`** restores the older Photutils / median-filter approximation; **`HST123_CALCSKY_NUMBA=0`** disables Numba for the first pass only (second pass stays vectorized; without Numba that pass is slower). The CLI **negative step** quick mode (**`getsky_q`**) is **not** ported—legacy fallback is used instead.

Use **`--all-psfs`** for the full filter set; **`--no-make`** / **`--no-psfs`** as needed. The installer logs **step-by-step progress** (what is downloading, extracting, building, and linking) via **`hst123.utils.logging`** (stderr when using the CLI); **`--quiet`** keeps only DEBUG-level detail for those lines. Re-running the same command **does not re-download** sources or PSF archives that are already installed (metadata under **`.hst123-dolphot/`** in the install tree). For PSFs it also skips when matching **`.psf`** (and PAM **`.fits`**) files are already present under the install tree (e.g. **`dolphot2.0/acs/data/`** or **`WFC/`**), not only when the stamp file exists. Use **`--force-download`** to fetch everything again. See **`hst123-install-dolphot --help`**. The helper fetches **DOLPHOT 3.1** (the **3.0** tarballs often return **HTTP 403** from the server). Before **`make`**, it applies a small **source patch** to **`dolphot.c`** (enlarges a stack buffer in **`main`**) so **long absolute output paths** do not trigger a **macOS buffer-overflow abort** (**SIGTRAP** / `zsh: trace trap`). Use **`--no-source-patches`** to build pristine upstream sources (not recommended on macOS with long paths). PSF tarballs may unpack under **`.../dolphot2.0/...`** (nested or **sibling** of **`dolphot3.1/`**, e.g. conda **`$CONDA_PREFIX/opt/hst123-dolphot/dolphot2.0/`**); the installer merges **`*.psf`** into **`dolphot3.1/acs/data/`**, **`wfc3/{data,IR,UVIS}/`**, and **`wfpc2/data/`**, where DOLPHOT looks at runtime.

**Apple Silicon (arm64) — build DOLPHOT:** You need **Apple’s Xcode Command Line Tools** (`xcode-select --install`) and **Homebrew `libomp`** for the OpenMP link step: **`brew install libomp`**. The installer uses **`-Xclang -fopenmp`** in **`THREAD_CFLAGS`** and, when **`/opt/homebrew/opt/libomp`** exists, sets **`THREAD_LIBS`** to **`-L/opt/homebrew/opt/libomp/lib -lomp`** so the linker does not pick a wrong-arch **`libomp`** (e.g. **x86_64** under **`/usr/local/lib`**). If **`make`** fails with undefined **`___kmpc_*`** / **`omp_*`** and **`ld`** warns about ignoring **`libomp.dylib`**, remove or avoid that stale library and re-run **`hst123-install-dolphot`**.

**DOLPHOT exits with `zsh: trace trap` (SIGTRAP) after “Using N threads”:** This is almost always an **OpenMP / `libomp`** mismatch on macOS (wrong architecture, or broken multithreaded link). **Workaround:** run with one thread — the pipeline sets **`OMP_NUM_THREADS=1`** automatically on **Darwin** unless you already set **`OMP_NUM_THREADS`** or **`HST123_DOLPHOT_OMP_THREADS`**. From the shell: `export OMP_NUM_THREADS=1` then `dolphot …`. Long-term: reinstall DOLPHOT with **`hst123-install-dolphot`** so **`THREAD_LIBS`** points at the correct Homebrew **`libomp`**. The installer now defaults to a **non-threaded build**; use **`--threaded`** only when OpenMP is configured correctly.

**Sources and build:** By default it downloads the **base** tarball plus **ACS, WFC3, WFPC2, Roman, NIRCam, NIRISS, MIRI, and Euclid** module archives and **merges** them into one directory (same layout as unpacking each into the DOLPHOT root). It enables module `USE*` flags and runs **`make`** in the directory that contains **`Makefile`** (including nested **`dolphot3.1/`** if present). The installer defaults to **non-threaded** (`THREADED` lines left commented). Use **`--threaded`** to enable OpenMP (`THREADED`, `THREAD_CFLAGS`, `THREAD_LIBS`) once `libomp` is configured correctly. On **macOS (Intel)**, Homebrew **`libomp`** is under **`/usr/local/opt/libomp`**; the same **`THREAD_LIBS`** logic applies when that path exists. **`--hst-modules-only`** limits downloads and Makefile flags to **HST only** (ACS/WFC3/WFPC2).

**macOS:** If `pip install drizzlepac` fails (e.g. HDF5), install with Homebrew: `brew install hdf5 c-blosc`, set `HDF5_DIR` and `BLOSC_DIR`, then retry.

**Without Conda:** `python3 -m venv .venv && source .venv/bin/activate`, then run the same `pip` steps above.

**`pip install` fails while building NumPy or Astropy** (Meson/ninja, C++ errors such as ``'type_traits' file not found``): this usually means **pip is compiling from source** because there is **no binary wheel** for your Python version. The repo pins **NumPy 1.x** and **Astropy 5.x** (see ``pyproject.toml``); on **Python 3.13+** those often lack wheels, so pip downloads tarballs and the build can fail. **Fix:** use the **conda** env from ``environment.yml`` (**Python 3.10**, with NumPy/Astropy from conda-forge), or use a **venv with Python 3.10–3.12**, or ``conda install 'numpy<2' 'astropy>=5.3,<6'`` (and matching SciPy) **before** ``pip install -e .`` so pip does not try to build the stack. Avoid installing this package into **base** with an unsupported Python unless you know wheels exist.

---

## Usage

```bash
hst123 <ra> <dec> [options]
# Example:
hst123 12:30:00 -45.0 --download
```

- **`--work-dir`** — Absolute path is recommended; defaults to the current directory at startup. **`--raw-dir`** defaults to **`<work-dir>/raw`** (science FITS are written there, then copied into the work directory for reduction). **MAST / astroquery staging** (the temporary **`mastDownload`** tree) is always created under **`<work-dir>/.mast_download_staging/`** and removed after each file is moved to **`raw`** or **`--archive`**, so nothing is left in the directory you launched the command from when **`--work-dir`** points elsewhere.
- **`--download`** — Fetch data from MAST for the given RA/Dec (5′ radius). Use **`--token`** for private data ([MAST auth](https://auth.mast.stsci.edu/info)).
- **`--run-dolphot`** — Run DOLPHOT (requires DOLPHOT on PATH).
- **`--scrape-dolphot`** — Extract photometry at the target from DOLPHOT output.

**DOLPHOT catalog → HDF5:** After a run, the pipeline leaves a base name like **`dp0000`** (no extension) plus sidecars (**`dp0000.columns`**, **`.param`**, **`.info`**, **`.data`**, **`.warnings`**). To pack the full numeric catalog with descriptive column names and merged metadata (including per-**`imgNNNN`** parameters when available), use **`hst123.utils.dolphot_catalog_hdf5.write_dolphot_catalog_hdf5`** (`pip install h5py` or **`pip install .[hdf5]`**). See **`tests/test_dolphot_catalog_hdf5.py`**.

Run in a directory that will hold (or already holds) your images. Without `--download`, hst123 uses existing files in the working directory. Full option list: **`hst123 --help`**.

### Logging

Package output for **`hst123`** and **`hst123-install-dolphot`** goes through **`hst123.utils.logging`**: use **`get_logger(__name__)`** and **`log.info` / `warning` / `error`** instead of **`print`**. The CLI calls **`ensure_cli_logging_configured()`** so a formatted handler is attached to **stderr** if none is present yet.

After **`--work-dir`** is known, the pipeline also creates **`<work-dir>/logs/`** and writes a session file **`hst123_pipeline_<timestamp>_<pid>.log`** with the same formatted records as the console (see **`attach_work_dir_log_file`** in `hst123/utils/logging.py`). **AstroDrizzle**, **photeq**, **TweakReg shift files**, and **`headerlet.log`** are written under **`<work-dir>/.hst123_runfiles/`**, replayed into the session log, then **deleted** (no **`photeq.log`** / **`astrodrizzle.log`** in the work root). Common drizzle scratch files (**`staticMask.fits`**, **`skymask_cat`**, headerlets, etc.) are removed. Use **`--keep-drizzle-artifacts`** to leave those in the work directory.

- **`HST123_LOG_LEVEL`** — default **`INFO`** (compact: few blank lines, one-line startup summary, no multi-line **`HDUList.info()`** dumps). Set **`DEBUG`** for **`@log_calls`**, TweakReg shift-file reads, per-file **`TWEAKSUC`** checks, input-list tables, MAST try-lines, WCS key updates, cache clears, and similar detail.
- **`HST123_REPLAY_SUBLOGS`** — default (**unset** or **`1`**) echoes every line of replayed sublogs into the session log. Set **`0`**, **`false`**, or **`summary`** for a **one-line summary** per file. The pipeline **always** folds AstroDrizzle and photeq runfiles into the session log with **compact whitespace** (this is independent of the env var for those two steps).
- **`HST123_SKY_NUMBA`** — set to **`1`** / **`true`** to use a parallel **Numba** median filter for the large-image Python sky path (when **`numba`** is installed); default is **SciPy** `median_filter`.
- File logging and other options: **`HST123_LOG_ENABLE_FILE`**, **`HST123_LOG_DIR`**, etc. (see **`LogConfig`** in `hst123/utils/logging.py`).
- Third-party libraries may still write directly to the terminal.

**External executables** (`dolphot`, `calcsky`, optional `*mask` / `splitgroups`, `make` in the DOLPHOT installer): stdout/stderr are captured and streamed through the same logging handlers (see `run_external_command` in `hst123/utils/logging.py`). DOLPHOT console text is still written to the usual `.output` / `.fake.output` file via a tee.

**Calibration reference files (WCS / `updatewcs`):** Before `stwcs.updatewcs`, hst123 **aligns SIP headers** on disk: if SIP polynomial keys (`A_ORDER` / `B_ORDER`) are present but `CTYPE*` are `RA---TAN` / `DEC--TAN` without the `-SIP` suffix (common on MAST ACS/WFC3 images), it rewrites them to `RA---TAN-SIP` / `DEC--TAN-SIP` so astropy/stwcs stop emitting long SIP inconsistency messages. It also **removes stale alternate WCS** entries on each **`SCI`** extension when an alternate reuses the primary **`WCSNAME`** but the FITS keywords differ (otherwise `stwcs.updatehdr.update_wcs` / AstrometryDB hits `altwcs.archive_wcs(..., QUIET_ABORT)` and logs non-unique-`wcsname` warnings). **`WCSNAME=TWEAK`** is applied only on **`SCI`** extensions before drizzle (not on every HDU), reducing duplicate-name issues. During `updatewcs`, astropy/stwcs loggers are raised to **ERROR**, `FITSFixedWarning` is ignored, and C-level **stdout** is suppressed so AstrometryDB / IDCTAB chatter does not flood the console.

The pipeline downloads small FITS tables (e.g. IDCTAB, NPOLFILE) into **`<work-dir>/cals/`** and points headers at those paths. Downloads use **HTTPS** first: HST CRDS and the **`https://ssb.stsci.edu/cdbs/`** mirror (with **`jref` / `iref` / `uref`** paths — not the legacy **`jref.old`** segment, which 404s over HTTPS). **FTP** is only tried last and is often blocked on campus networks. If you already use CRDS locally, set **`jref`**, **`iref`**, or **`uref`** to your cache directory so files are **copied** instead of fetched.

---

## Supported instruments

| Instrument | File types |
|------------|------------|
| WFPC2 | c0m.fits, c1m.fits (both) |
| ACS/WFC | flc.fits |
| ACS/HRC | flt.fits |
| WFC3/UVIS | flc.fits |
| WFC3/IR | flt.fits |

You can provide **`--reference`** or let hst123 build one from the data. Alignment: **tweakreg** (default) or **jhat** (**`--align-with`**). Photometry is reported in AB mag by default; see `docs/zeropoints.md`.

---

## Options (summary)

- **Run environment:** `--work-dir`, `--raw-dir`, `--archive`, `--cleanup`, `--keep-drizzle-artifacts`, `--by-visit`
- **Filters/dates:** `--before`, `--after`, `--only-filter`, `--only-wide`
- **Reference:** `--reference` / `--ref`, `--reference-filter`, `--reference-instrument`, `--avoid-wfpc2`
- **Alignment:** `--tweak-search`, `--tweak-min-obj`, `--tweak-thresh`, `--skip-tweakreg`, `--align-with`, `--hierarchical`
- **Drizzle:** `--drizzle-all`, `--drizzle-dim`, `--drizzle-scale`, `--sky-sub`, `--redrizzle`, `--fix-zpt`, `--no-rotation`
- **DOLPHOT:** `--run-dolphot`, `--dolphot` / `--dp`, `--dolphot-lim`, `--do-fake`, `--add-crmask`, `--include-all-splits`
- **Scraping:** `--scrape-dolphot` / `--sd`, `--scrape-all`, `--scrape-radius`, `--no-cuts`, `--brightest`

---

## Documentation

- **Online:** [charliekilpatrick.github.io/hst123](https://charliekilpatrick.github.io/hst123/) (API reference, user guide, changelog, zero points).
- **Local build:** `pip install -e ".[docs]"` then `cd docs && make html` → `docs/build/html/index.html`.
- **Sources:** `docs/index.rst`, `docs/user_guide.rst`, `docs/api.rst`, plus `docs/changelog.md` and `docs/zeropoints.md` (MyST).

---

## Citing and contact

**Citation:** C. D. Kilpatrick, *hst123: HST download, alignment, drizzle, and DOLPHOT photometry pipeline*, GitHub (or project URL). If a DOI (e.g. Zenodo) is assigned to a release, cite that. We welcome notice of papers that use hst123.

**Contact:** Charlie Kilpatrick, ckilpatrick@northwestern.edu. **Bugs and feature requests:** please open an issue on the repository.
