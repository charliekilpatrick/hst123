# hst123

**hst123** is a Python package for HST data: download from MAST, align (tweakreg/jhat), drizzle, run DOLPHOT, and scrape photometry. Optimized for point-source photometry across multiple visits and filters.

**Issues:** Report bugs and feature requests via the project issue tracker. Other questions or publications using hst123: [Contact](#contact).

---

## Repository status

- **Package:** Install with `pip install -e .`; provides the `hst123` CLI and the `hst123-install-dolphot` helper. Run as a module: `python -m hst123`. Import in code: `import hst123` (e.g. `hst123.hst123()`, `hst123.main()`).
- **Python:** 3.8+
- **Versioning:** From git tags (setuptools-scm); `hst123 --version`
- **Layout:** Main pipeline in `hst123/_pipeline.py`; helpers in `hst123/primitives/` (FITS, photometry, astrometry, DOLPHOT, scrape) and `hst123/utils/` (options, logging, display, visit, WCS)
- **Tests:** `pytest` in `tests/`; optional markers `network`, `dolphot` (see pyproject.toml)
- **Docs:** `docs/` (changelog, zeropoints, stwcs); Sphinx with `pip install -e ".[docs]"` then `cd docs && make html`

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

After install, the **`hst123`** command is available (and **`hst123-install-dolphot`** for DOLPHOT). Download, alignment, and drizzling work without DOLPHOT.

**3. DOLPHOT (only for `--run-dolphot`)**  
DOLPHOT is external; not bundled. Install and build from [americano.dolphinsim.com/dolphot](http://americano.dolphinsim.com/dolphot/), or use the helper:

```bash
hst123-install-dolphot --dolphot-dir ~/dolphot
cd ~/dolphot && make
export PATH="$HOME/dolphot:$PATH"
```

Use `--all-psfs` for full filter set; see `hst123-install-dolphot --help`.

**macOS:** If `pip install drizzlepac` fails (e.g. HDF5), install with Homebrew: `brew install hdf5 c-blosc`, set `HDF5_DIR` and `BLOSC_DIR`, then retry.

**Without Conda:** `python3 -m venv .venv && source .venv/bin/activate`, then run the same `pip` steps above.

---

## Usage

Run the pipeline from the command line (after `pip install -e .`):

```bash
hst123 <ra> <dec> [options]
# or as a module:
python -m hst123 <ra> <dec> [options]
```

Example:

```bash
hst123 12:30:00 -45.0 --download
```

- **`--download`** — Fetch data from MAST for the given RA/Dec (5′ radius). Use **`--token`** for private data ([MAST auth](https://auth.mast.stsci.edu/info)).
- **`--run-dolphot`** — Run DOLPHOT (requires DOLPHOT on PATH).
- **`--scrape-dolphot`** — Extract photometry at the target from DOLPHOT output.

Use the package in a directory that will hold (or already holds) your images. Without `--download`, hst123 uses existing files in the working directory. Full option list: **`hst123 --help`**.

**Programmatic use:** `import hst123` then `hst123.hst123()` for the pipeline class or `hst123.main()` for the entry point (same as the CLI).

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

- **Run environment:** `--work-dir`, `--raw-dir`, `--make-clean`, `--archive`, `--cleanup`, `--by-visit`
- **Filters/dates:** `--before`, `--after`, `--only-filter`, `--only-wide`, `--no-large-reduction`, `--large-num`
- **Reference:** `--reference` / `--ref`, `--reference-filter`, `--reference-instrument`, `--avoid-wfpc2`
- **Alignment:** `--tweak-search`, `--tweak-min-obj`, `--tweak-thresh`, `--skip-tweakreg`, `--align-with`, `--hierarchical`
- **Drizzle:** `--drizzle-all`, `--drizzle-dim`, `--drizzle-scale`, `--sky-sub`, `--redrizzle`, `--fix-zpt`, `--no-rotation`
- **DOLPHOT:** `--run-dolphot`, `--dolphot` / `--dp`, `--dolphot-lim`, `--do-fake`, `--add-crmask`, `--include-all-splits`
- **Scraping:** `--scrape-dolphot` / `--sd`, `--scrape-all`, `--scrape-radius`, `--no-cuts`, `--brightest`

---

## Documentation

- **`docs/changelog.md`** — Version history
- **`docs/zeropoints.md`** — AB vs Vega, zero-point formula
- **`docs/stwcs_dependency_analysis.md`** — Why stwcs is required
- **`docs/index.md`** — Doc index and Sphinx build

---

## Citing and contact

**Citation:** C. D. Kilpatrick, *hst123: HST download, alignment, drizzle, and DOLPHOT photometry pipeline* (Python package), GitHub (or project URL). If a DOI (e.g. Zenodo) is assigned to a release, cite that. We welcome notice of papers that use hst123.

**Contact:** Charlie Kilpatrick, ckilpatrick@northwestern.edu. **Bugs and feature requests:** please open an issue on the repository.
