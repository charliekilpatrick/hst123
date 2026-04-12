# hst123

[![Build and Test](https://github.com/charliekilpatrick/hst123/actions/workflows/build-test.yml/badge.svg)](https://github.com/charliekilpatrick/hst123/actions/workflows/build-test.yml)
[![Documentation](https://github.com/charliekilpatrick/hst123/actions/workflows/documentation.yml/badge.svg)](https://github.com/charliekilpatrick/hst123/actions/workflows/documentation.yml)
[![Documentation site](https://img.shields.io/badge/documentation-GitHub%20Pages-4b32b3)](https://charliekilpatrick.github.io/hst123/)

Pipeline for HST data: download from MAST, align (tweakreg/jhat), drizzle, run DOLPHOT, and scrape photometry. Optimized for point-source photometry across multiple visits and filters.

**Issues:** Report bugs and feature requests via the project issue tracker. Other questions or publications using hst123: [Contact](#contact).

---

## Repository status

- **Python:** **3.12** (``requires-python`` in ``pyproject.toml`` pins the **3.12** line: ``>=3.12,<3.13``)
- **Versioning:** From git tags (setuptools-scm); `hst123 --version`
- **Layout:** Main pipeline in `hst123/_pipeline.py`; helpers in `hst123/primitives/` (FITS, photometry, astrometry, DOLPHOT, scrape) and `hst123/utils/` (options, display, visit, WCS)
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
pip install drizzlepac
pip install -e .
```

The **STScI WCS** stack used by hst123 (``updatewcs``, ``HSTWCS``, ``altwcs``) is **bundled** under ``hst123.utils.stwcs`` (see ``STWCS_VENDOR.txt`` there) and reached only via ``hst123.utils.stsci_wcs`` — **PyPI ``stwcs`` is not an hst123 dependency**. DrizzlePac may still install ``stwcs`` for its own use.

The `hst123` command is available. Download, alignment, and drizzling work without DOLPHOT.

**3. DOLPHOT (only for `--run-dolphot`)**
DOLPHOT is external; not bundled. With **conda activated**, the helper downloads sources and PSF/PAM reference data under **`$CONDA_PREFIX/opt/hst123-dolphot`**, runs **`make`**, and symlinks **`dolphot`**, **`calcsky`**, and optional mask/split tools into **`$CONDA_PREFIX/bin`**. **Masking** (`acsmask` / `wfc3mask` / `wfpc2mask`) and **`splitgroups`** are implemented in Python by default (see **`hst123.utils.dolphot_mask`**, **`hst123.utils.dolphot_splitgroups`**); set **`HST123_DOLPHOT_MASK_EXTERNAL=1`** or **`HST123_DOLPHOT_SPLITGROUPS_EXTERNAL=1`** to force the C binaries.

```bash
conda activate hst123   # or your env with hst123 installed
hst123-install-dolphot
```

Without conda (or to choose a location), pass **`--dolphot-dir /path/to/dolphot`**; use **`--no-link-conda-bin`** and **`export PATH="/path/to/dolphot:$PATH"`** if you do not want links in the env `bin/`. Manual install: [americano.dolphinsim.com/dolphot](http://americano.dolphinsim.com/dolphot/).

The helper installs **DOLPHOT 3.1**, merges PSF tables into the tree DOLPHOT expects, and can apply small source patches for long paths on macOS. Re-running **`hst123-install-dolphot`** does not re-download already-installed artifacts; use **`--force-download`** to refresh. See **`hst123-install-dolphot --help`**.

---

## Usage

```bash
hst123 <ra> <dec> [options]
# Example:
hst123 12:30:00 -45.0 --download
```

- **`--work-dir`** — Absolute path is recommended; defaults to the current directory at startup. **`--raw-dir`** defaults to **`<work-dir>/raw`** (science FITS are written there, then copied into the work directory for reduction). **MAST / astroquery staging** (the temporary **`mastDownload`** tree) is always created under **`<work-dir>/.mast_download_staging/`** and removed after each file is moved to **`raw`** or **`--archive`**, so nothing is left in the directory you launched the command from when **`--work-dir`** points elsewhere. Most calibrated inputs and per-epoch drizzle outputs live under **`<work-dir>/workspace/`**; **`--drizzle-all`** writes the consolidated **`drizzle/`** directory under **`<work-dir>/drizzle/`** (the base work directory, not inside **`workspace/`**).
- **`--download`** — Fetch data from MAST for the given RA/Dec (5′ radius). Use **`--token`** for private data ([MAST auth](https://auth.mast.stsci.edu/info)).
- **`--run-dolphot`** — Run DOLPHOT (requires DOLPHOT on PATH).
- **`--scrape-dolphot`** — Extract photometry at the target from DOLPHOT output.

**DOLPHOT catalog → HDF5:** Optional helper **`hst123.utils.dolphot_catalog_hdf5.write_dolphot_catalog_hdf5`** writes the DOLPHOT table to a single **HDF5** file (columnar datasets + metadata). Requires **`h5py`** (`pip install h5py` or **`pip install .[hdf5]`**).

Run in a directory that will hold (or already holds) your images. Without `--download`, hst123 uses existing files in the working directory. Full option list: **`hst123 --help`**.

---

## Supported instruments

| Instrument | File types                |
| ---------- | ------------------------- |
| WFPC2      | c0m.fits, c1m.fits (both) |
| ACS/WFC    | flc.fits                  |
| ACS/HRC    | flt.fits                  |
| WFC3/UVIS  | flc.fits                  |
| WFC3/IR    | flt.fits                  |

You can provide **`--reference`** or let hst123 build one from the data. Alignment: **tweakreg** (default) or **jhat** (**`--align-with`**). Photometry is reported in AB mag by default; see `docs/zeropoints.md`.

---

## Options (summary)

- **Run environment:** `--work-dir`, `--raw-dir`, `--archive`, `--cleanup`, `--keep-drizzle-artifacts`, `--by-visit`
- **Filters/dates:** `--before`, `--after`, `--only-filter`, `--only-wide`
- **Reference:** `--reference` / `--ref`, `--reference-filter`, `--reference-instrument`, `--avoid-wfpc2`
- **Alignment:** `--tweak-search`, `--tweak-min-obj`, `--tweak-thresh`, `--skip-tweakreg`, `--align-with`, `--hierarchical`
- **Drizzle / parallelism:** `--drizzle-all`, `--drizzle-dim`, `--drizzle-scale`, `--sky-sub`, `--redrizzle`, `--fix-zpt`, `--no-rotation`, `--max-cores` (AstroDrizzle `num_cores` and DOLPHOT prep thread pool, capped by exposure count; default min(8, CPU count); `--max-cores 1` forces serial prep and single-worker drizzle)
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

**Suggested references for citing hst123:** If you cite this software in a paper or proposal, you may also reference the following peer-reviewed works that use *Hubble Space Telescope* data heavily and describe the use of **hst123**:

- Kilpatrick et al., “A cool and inflated progenitor candidate for the Type Ib supernova 2019yvr at 2.6 yr before explosion,” *MNRAS* **504**, 2073 (2021). [doi:10.1093/mnras/stab838](https://doi.org/10.1093/mnras/stab838) · [arXiv:2101.03185](https://arxiv.org/abs/2101.03185)
- Kilpatrick et al., “Hubble Space Telescope Observations of GW170817: Complete Light Curves and the Properties of the Galaxy Merger of NGC 4993,” *ApJ* **926**, 49 (2022). [doi:10.3847/1538-4357/ac3e59](https://doi.org/10.3847/1538-4357/ac3e59) · [arXiv:2109.06211](https://arxiv.org/abs/2109.06211)
- Kilpatrick et al., “Type II-P supernova progenitor star initial masses and SN 2020jfo: direct detection, light-curve properties, nebular spectroscopy, and local environment,” *MNRAS* **524**, 2161 (2023). [doi:10.1093/mnras/stad1954](https://doi.org/10.1093/mnras/stad1954) · [arXiv:2307.00550](https://arxiv.org/abs/2307.00550)

**Contact:** Charlie Kilpatrick, ckilpatrick@northwestern.edu. **Bugs and feature requests:** please open an issue on the repository.
