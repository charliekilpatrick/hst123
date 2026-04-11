"""Download DOLPHOT sources and PSF files; build and link into conda env when requested."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from hst123.utils.logging import (
    ensure_cli_logging_configured,
    get_logger,
    run_external_command,
)

log = get_logger(__name__)

# If True, skip user-facing progress lines on stderr (set in main()).
_quiet_progress = False


def _progress(msg: str) -> None:
    """User-visible status via ``hst123`` logging (stderr when CLI configured)."""
    if _quiet_progress:
        log.debug("%s", msg)
    else:
        log.info("%s", msg)


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n / (1024 * 1024):.1f} MiB"


# Relative to CONDA_PREFIX: sources, PSFs, and built executables live here.
CONDA_DOLPHOT_RELATIVE = Path("opt") / "hst123-dolphot"

# Binaries linked into conda bin; pipeline only *requires* those in DOLPHOT_REQUIRED_SCRIPTS
# (dolphot, calcsky). Mask/split C tools remain optional (Python implementations in hst123).
DOLPHOT_EXECUTABLES = (
    "dolphot",
    "calcsky",
    "acsmask",
    "wfc3mask",
    "wfpc2mask",
    "splitgroups",
)


def _resolve_dolphot_executable(source_dir: Path, name: str) -> Path | None:
    """
    Path to a built DOLPHOT program. DOLPHOT 3.x places binaries under ``bin/``;
    older layouts may have them next to the ``Makefile``.
    """
    root = Path(source_dir).resolve()
    for cand in (root / "bin" / name, root / name):
        if cand.is_file():
            return cand
    return None


def dolphot_path_for_shell(make_root: Path) -> Path:
    """
    Directory to add to ``PATH`` so ``dolphot`` is found (``.../dolphot3.1/bin``
    when built, else sensible default under ``make_root``).
    """
    root = Path(make_root).resolve()
    p = _resolve_dolphot_executable(root, "dolphot")
    if p is not None:
        return p.parent
    if (root / "bin").is_dir():
        return root / "bin"
    return root


def get_conda_prefix() -> Path | None:
    """Return active conda env root from CONDA_PREFIX, or None if unset."""
    raw = os.environ.get("CONDA_PREFIX")
    if not raw:
        return None
    p = Path(raw).resolve()
    return p if p.is_dir() else None


def default_dolphot_install_dir() -> Path:
    """
    Default install directory: ``$CONDA_PREFIX/opt/hst123-dolphot`` when in a
    conda environment, else ``./dolphot`` under the current working directory.
    """
    prefix = get_conda_prefix()
    if prefix is not None:
        return prefix / CONDA_DOLPHOT_RELATIVE
    return Path.cwd() / "dolphot"


# ACS/WFC PAM FITS shipped in ACS_WFC_PAM.tar.gz; ``acsmask`` reads these from the DOLPHOT tree.
ACS_WFC_PAM_FILENAMES = ("wfc1_pam.fits", "wfc2_pam.fits")
# Distortion / pixel-area maps required by wfc3mask (Python and C); shipped with
# ``dolphot3.1.WFC3.tar.gz`` under ``wfc3/data/``.
WFC3_MASK_MAP_FILENAMES = (
    "UVIS1wfc3_map.fits",
    "UVIS2wfc3_map.fits",
    "ir_wfc3_map.fits",
)
_MIN_PAM_FILE_BYTES = 512
_MIN_PSF_FILE_BYTES = 512


def dolphot_acs_data_dir(source_dir: Path | str | None = None) -> Path | None:
    """
    Return ``.../acs/data`` under a DOLPHOT source tree if present.

    Tries the usual DOLPHOT 3.x layout (``make_root/acs/data``) and a legacy
    ``dolphot2.0/acs/data`` layout (upstream ``ACS_WFC_PAM.tar.gz`` still uses
    the ``dolphot2.0/`` prefix).

    Prefer a directory that already contains both ``wfc1_pam.fits`` and
    ``wfc2_pam.fits`` so an empty ``acs/data`` from the ACS module does not
    shadow the real PAM location after extraction.
    """
    root = default_dolphot_install_dir() if source_dir is None else Path(source_dir)
    root = root.resolve()
    make_root = dolphot_make_root(root)
    candidates = (
        make_root / "acs" / "data",
        make_root / "dolphot2.0" / "acs" / "data",
    )
    for cand in candidates:
        if _acs_wfc_pam_files_ok_in_data_dir(cand):
            return cand
    for cand in candidates:
        if cand.is_dir():
            return cand
    return None


def dolphot_wfc3_data_dir(source_dir: Path | str | None = None) -> Path | None:
    """
    Return ``.../wfc3/data`` under a DOLPHOT source tree if present.

    Tries DOLPHOT 3.x ``make_root/wfc3/data`` and legacy ``dolphot2.0/wfc3/data``.
    """
    root = default_dolphot_install_dir() if source_dir is None else Path(source_dir)
    root = root.resolve()
    make_root = dolphot_make_root(root)
    for rel in (Path("wfc3") / "data", Path("dolphot2.0") / "wfc3" / "data"):
        d = make_root / rel
        if d.is_dir():
            return d
    return None


def resolve_dolphot_wfc3_data_dir() -> Path | None:
    """First ``wfc3/data`` directory found under plausible DOLPHOT install roots."""
    for r in _candidate_dolphot_source_roots():
        d = dolphot_wfc3_data_dir(r)
        if d is not None:
            return d
    return None


def _wfc3_mask_maps_ok_in_data_dir(data_dir: Path) -> bool:
    """True if UVIS/IR ``*wfc3_map.fits`` exist under *data_dir* with non-trivial size."""
    data_dir = Path(data_dir).resolve()
    if not data_dir.is_dir():
        return False
    for name in WFC3_MASK_MAP_FILENAMES:
        p = data_dir / name
        try:
            if not p.is_file() or p.stat().st_size < _MIN_PAM_FILE_BYTES:
                return False
        except OSError:
            return False
    return True


def verify_wfc3_mask_support_files() -> tuple[bool, list[str]]:
    """
    Return (ok, problem_messages) for WFC3 ``wfc3mask`` distortion maps.

    Python :func:`~hst123.utils.dolphot_mask.apply_wfc3mask` and the C ``wfc3mask``
    binary both read ``UVIS1wfc3_map.fits``, ``UVIS2wfc3_map.fits``, and
    ``ir_wfc3_map.fits`` from ``.../wfc3/data`` (from ``dolphot3.1.WFC3.tar.gz``).
    """
    d = resolve_dolphot_wfc3_data_dir()
    msgs: list[str] = []
    if d is None:
        return False, [
            "DOLPHOT wfc3/data not found (searched CONDA_PREFIX/opt/hst123-dolphot and "
            "default install roots)."
        ]
    if not _wfc3_mask_maps_ok_in_data_dir(d):
        for name in WFC3_MASK_MAP_FILENAMES:
            p = d / name
            if not p.is_file():
                msgs.append(f"missing {p}")
                continue
            try:
                sz = p.stat().st_size
            except OSError as exc:
                msgs.append(f"cannot stat {p}: {exc}")
                continue
            if sz < _MIN_PAM_FILE_BYTES:
                msgs.append(
                    f"{p} is too small ({sz} B); likely corrupt or cloud placeholder — "
                    "re-extract dolphot3.1.WFC3.tar.gz"
                )
        if not msgs:
            msgs.append(f"WFC3 mask maps invalid under {d}")
    return (len(msgs) == 0, msgs)


def relocate_acs_wfc_pam_into_canonical_layout(make_root: Path) -> bool:
    """
    Ensure ``wfc1_pam.fits`` and ``wfc2_pam.fits`` live under *make_root*/acs/data.

    The upstream ``ACS_WFC_PAM.tar.gz`` extracts to
    ``dolphot2.0/acs/data/*.fits``; DOLPHOT 3.x and hst123 expect
    ``<make_root>/acs/data/``. Copies from the legacy tree (or an rglob search)
    when needed.

    Returns
    -------
    bool
        True if *make_root*/acs/data contains both PAM files after this call.
    """
    make_root = Path(make_root).resolve()
    target = make_root / "acs" / "data"
    if _acs_wfc_pam_files_ok_in_data_dir(target):
        return True
    target.mkdir(parents=True, exist_ok=True)

    legacy = make_root / "dolphot2.0" / "acs" / "data"
    if _acs_wfc_pam_files_ok_in_data_dir(legacy):
        for name in ACS_WFC_PAM_FILENAMES:
            shutil.copy2(legacy / name, target / name)
        log.info(
            "ACS WFC PAM: copied %s -> %s (upstream tarball uses dolphot2.0/ prefix)",
            legacy,
            target,
        )
        _progress(f"  ACS WFC PAM placed under {target} (from {legacy.name} layout).")
        return True

    found: dict[str, Path] = {}
    for name in ACS_WFC_PAM_FILENAMES:
        for p in make_root.rglob(name):
            if not p.is_file():
                continue
            try:
                if p.stat().st_size >= _MIN_PAM_FILE_BYTES:
                    found[name] = p
                    break
            except OSError:
                continue
    if len(found) == len(ACS_WFC_PAM_FILENAMES):
        for name, src in found.items():
            shutil.copy2(src, target / name)
        log.info("ACS WFC PAM: copied discovered files into %s", target)
        _progress(f"  ACS WFC PAM placed under {target} (search under {make_root.name}).")
        return _acs_wfc_pam_files_ok_in_data_dir(target)

    return False


def _legacy_dolphot20_roots(make_root: Path) -> list[Path]:
    """
    Roots of legacy ``dolphot2.0/`` trees from upstream tarballs.

    Upstream archives use a ``dolphot2.0/`` prefix. Content may land:

    - Under the Makefile root: ``<make_root>/dolphot2.0/...``
    - Next to ``dolphot3.1`` (common conda layout): ``<make_root>/../dolphot2.0/...``
      when ``make_root`` is ``.../opt/hst123-dolphot/dolphot3.1`` and extraction
      populated ``.../opt/hst123-dolphot/dolphot2.0/...`` instead.

    Returns existing directory roots only (deduplicated).
    """
    make_root = Path(make_root).resolve()
    roots: list[Path] = []
    seen: set[Path] = set()

    def add(p: Path) -> None:
        try:
            r = p.resolve()
        except OSError:
            return
        if r in seen:
            return
        seen.add(r)
        if r.is_dir():
            roots.append(r)

    add(make_root / "dolphot2.0")
    parent = make_root.parent
    if parent.is_dir():
        add(parent / "dolphot2.0")
    return roots


def _acs_psf_legacy_data_dirs(make_root: Path) -> list[Path]:
    """
    Directories that may hold ACS ``*.psf`` from upstream tarballs.

    Upstream archives use a ``dolphot2.0/acs/data`` prefix. Those files may land
    under nested ``<make_root>/dolphot2.0/acs/data`` or sibling
    ``<make_root>/../dolphot2.0/acs/data``.

    Order: nested under *make_root* first, then sibling under the install parent.
    """
    out: list[Path] = []
    for root in _legacy_dolphot20_roots(make_root):
        d = root / "acs" / "data"
        if d.is_dir():
            out.append(d)
    return out


def _copy_psf_files_into_canonical_dir(src_dir: Path, dst_dir: Path) -> int:
    """
    Copy ``*.psf`` from *src_dir* into *dst_dir* when the destination is missing
    or too small (same rules as ACS PSF relocation).

    Returns
    -------
    int
        Number of files copied.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in sorted(src_dir.glob("*.psf")):
        try:
            if src.stat().st_size < _MIN_PSF_FILE_BYTES:
                continue
        except OSError:
            continue
        dst = dst_dir / src.name
        if dst.is_file():
            try:
                if dst.stat().st_size >= _MIN_PSF_FILE_BYTES:
                    continue
            except OSError:
                pass
        shutil.copy2(src, dst)
        copied += 1
    return copied


def relocate_acs_psf_into_canonical_layout(make_root: Path) -> int:
    """
    Copy ACS ``*.psf`` files from legacy ``dolphot2.0/acs/data`` into
    ``<make_root>/acs/data`` when needed.

    DOLPHOT 3.x reads ACS reference files from ``acs/data`` under the Makefile
    root, but PSF tarballs may extract under ``dolphot2.0/acs/data``. Keeping
    them only in the legacy path causes runtime failures like:
    ``Cannot open .../acs/data/F814W.wfc1.psf``.

    Returns
    -------
    int
        Number of files copied into canonical ``acs/data``.
    """
    make_root = Path(make_root).resolve()
    target = make_root / "acs" / "data"
    copied = 0
    for legacy in _acs_psf_legacy_data_dirs(make_root):
        copied += _copy_psf_files_into_canonical_dir(legacy, target)
    if copied:
        log.info(
            "ACS PSF: copied %d file(s) into canonical %s (from legacy layout(s))",
            copied,
            target,
        )
        _progress(f"  ACS PSFs placed under {target} (copied {copied} from legacy layout).")
    return copied


# WFC3 PSFs may live under data/, IR/, and/or UVIS/ in legacy trees.
_WFC3_PSF_LEGACY_SUBDIRS = ("data", "IR", "UVIS")


def relocate_wfc3_psf_into_canonical_layout(make_root: Path) -> int:
    """
    Copy WFC3 ``*.psf`` from legacy ``dolphot2.0/wfc3/{data,IR,UVIS}`` into
    ``<make_root>/wfc3/...`` when needed (nested or sibling ``dolphot2.0``).
    """
    make_root = Path(make_root).resolve()
    copied = 0
    for leg_root in _legacy_dolphot20_roots(make_root):
        for sub in _WFC3_PSF_LEGACY_SUBDIRS:
            src = leg_root / "wfc3" / sub
            if not src.is_dir():
                continue
            dst = make_root / "wfc3" / sub
            copied += _copy_psf_files_into_canonical_dir(src, dst)
    if copied:
        log.info(
            "WFC3 PSF: copied %d file(s) into canonical %s (from legacy layout(s))",
            copied,
            make_root / "wfc3",
        )
        _progress(
            f"  WFC3 PSFs merged under {make_root / 'wfc3'} ({copied} file(s) from legacy layout)."
        )
    return copied


def relocate_wfpc2_psf_into_canonical_layout(make_root: Path) -> int:
    """
    Copy WFPC2 ``*.psf`` from legacy ``dolphot2.0/wfpc2/data`` into
    ``<make_root>/wfpc2/data`` when needed (nested or sibling ``dolphot2.0``).
    """
    make_root = Path(make_root).resolve()
    copied = 0
    for leg_root in _legacy_dolphot20_roots(make_root):
        src = leg_root / "wfpc2" / "data"
        if not src.is_dir():
            continue
        dst = make_root / "wfpc2" / "data"
        copied += _copy_psf_files_into_canonical_dir(src, dst)
    if copied:
        log.info(
            "WFPC2 PSF: copied %d file(s) into canonical %s (from legacy layout(s))",
            copied,
            make_root / "wfpc2" / "data",
        )
        _progress(
            f"  WFPC2 PSFs merged under {make_root / 'wfpc2' / 'data'} "
            f"({copied} file(s) from legacy layout)."
        )
    return copied


def relocate_all_legacy_psf_into_canonical_layout(make_root: Path) -> dict[str, int]:
    """
    Merge ACS, WFC3, and WFPC2 ``*.psf`` from legacy ``dolphot2.0/...`` trees
    (nested under *make_root* or sibling under the install parent) into the
    canonical DOLPHOT 3.x layout under *make_root*.

    Call this after PSF archives are extracted or skipped so runtime lookups
    under ``acs/data``, ``wfc3/...``, and ``wfpc2/data`` succeed.
    """
    make_root = Path(make_root).resolve()
    counts = {
        "acs": relocate_acs_psf_into_canonical_layout(make_root),
        "wfc3": relocate_wfc3_psf_into_canonical_layout(make_root),
        "wfpc2": relocate_wfpc2_psf_into_canonical_layout(make_root),
    }
    total = sum(counts.values())
    if total:
        log.info(
            "Legacy PSF merge into %s: %s (total %d file(s))",
            make_root,
            counts,
            total,
        )
    return counts


def _candidate_dolphot_source_roots() -> list[Path]:
    """Ordered install roots to probe for ``acs/data`` (conda layout, then defaults)."""
    roots: list[Path] = []
    seen: set[Path] = set()

    def add(p: Path) -> None:
        try:
            r = Path(p).resolve()
        except OSError:
            return
        if not r.is_dir() or r in seen:
            return
        seen.add(r)
        roots.append(r)

    wh = shutil.which("acsmask")
    if wh:
        bindir = Path(wh).resolve().parent
        envish = bindir.parent
        add(envish / CONDA_DOLPHOT_RELATIVE)
        add(envish)
    add(default_dolphot_install_dir())
    return roots


def resolve_dolphot_acs_data_dir() -> Path | None:
    """First ``acs/data`` directory found under plausible DOLPHOT install roots."""
    for r in _candidate_dolphot_source_roots():
        d = dolphot_acs_data_dir(r)
        if d is not None:
            return d
    return None


def _acs_wfc_pam_files_ok_in_data_dir(data_dir: Path) -> bool:
    """
    True if both WFC PAM FITS exist directly under *data_dir* (i.e. ``.../acs/data``)
    with non-trivial size.
    """
    data_dir = Path(data_dir).resolve()
    if not data_dir.is_dir():
        return False
    for name in ACS_WFC_PAM_FILENAMES:
        p = data_dir / name
        try:
            if not p.is_file() or p.stat().st_size < _MIN_PAM_FILE_BYTES:
                return False
        except OSError:
            return False
    return True


def _acs_wfc_pam_payload_dir_ok(make_root: Path) -> bool:
    """
    True if both WFC PAM FITS exist under canonical ``make_root/acs/data`` or
    legacy ``make_root/dolphot2.0/acs/data`` (upstream tarball layout).
    """
    make_root = Path(make_root).resolve()
    for rel in (Path("acs") / "data", Path("dolphot2.0") / "acs" / "data"):
        if _acs_wfc_pam_files_ok_in_data_dir(make_root / rel):
            return True
    return False


def verify_acs_wfc_pam_files() -> tuple[bool, list[str]]:
    """
    Return (ok, problem_messages). ``ok`` is True if both WFC PAM files exist,
    are readable, and are larger than a trivial placeholder size.
    """
    d = resolve_dolphot_acs_data_dir()
    msgs: list[str] = []
    if d is None:
        return False, [
            "DOLPHOT acs/data not found (searched CONDA_PREFIX/opt/hst123-dolphot and "
            "directories next to acsmask)."
        ]
    if not _acs_wfc_pam_files_ok_in_data_dir(d):
        for name in ACS_WFC_PAM_FILENAMES:
            p = d / name
            if not p.is_file():
                msgs.append(f"missing {p}")
                continue
            try:
                sz = p.stat().st_size
            except OSError as exc:
                msgs.append(f"cannot stat {p}: {exc}")
                continue
            if sz < _MIN_PAM_FILE_BYTES:
                msgs.append(
                    f"{p} is too small ({sz} B); likely corrupt or cloud placeholder — "
                    "re-extract ACS_WFC_PAM.tar.gz"
                )
        if not msgs:
            msgs.append(f"ACS PAM layout invalid under {d}")
    return (len(msgs) == 0, msgs)


DOLPHOT_BASE_URL = "http://americano.dolphinsim.com/dolphot/"
# 3.0 base/module tarballs currently return HTTP 403 from the upstream host
# while 3.1 downloads succeed; 3.1 modules are unchanged from 3.0 per DOLPHOT site.
DOLPHOT_VERSION = "3.1"

SOURCES_BASE = f"dolphot{DOLPHOT_VERSION}.tar.gz"
# HST-only instrument modules (smaller tree / build).
SOURCES_MODULES_HST = [
    f"dolphot{DOLPHOT_VERSION}.ACS.tar.gz",
    f"dolphot{DOLPHOT_VERSION}.WFC3.tar.gz",
    f"dolphot{DOLPHOT_VERSION}.WFPC2.tar.gz",
]
# Roman / JWST / Euclid modules (same layout; extracted into the same tree as base).
SOURCES_MODULES_EXTENDED = [
    f"dolphot{DOLPHOT_VERSION}.Roman.tar.gz",
    f"dolphot{DOLPHOT_VERSION}.NIRCAM.tar.gz",
    f"dolphot{DOLPHOT_VERSION}.NIRISS.tar.gz",
    f"dolphot{DOLPHOT_VERSION}.MIRI.tar.gz",
    f"dolphot{DOLPHOT_VERSION}.EUCLID.tar.gz",
]
SOURCES_MODULES = SOURCES_MODULES_HST + SOURCES_MODULES_EXTENDED

# Makefile ``export USE*=1`` lines (DOLPHOT base Makefile comments these out by default).
_MAKEFILE_USE_VARS_HST = ("USEWFPC2", "USEACS", "USEWFC3")
_MAKEFILE_USE_VARS_EXTENDED = (
    "USEROMAN",
    "USENIRCAM",
    "USENIRISS",
    "USEMIRI",
    "USEEUCLID",
)

ONE_PSF_PER_INSTRUMENT = {
    "ACS": "ACS_WFC_F435W.tar.gz",
    "WFC3": "WFC3_UVIS_F555W.tar.gz",
    "WFPC2": "WFPC2_F555W.tar.gz",
}

# Metadata under this directory records completed downloads (idempotent re-runs).
DOLPHOT_STAMP_DIR = ".hst123-dolphot"
_SOURCES_STAMP_NAME = "sources.json"


def _stamp_root(root: Path) -> Path:
    return Path(root).resolve() / DOLPHOT_STAMP_DIR


def sources_install_is_complete(
    dest_dir: Path,
    *,
    module_tarballs: list[str] | None = None,
) -> bool:
    """
    True if ``dest_dir`` has a Makefile and our stamp matches the current
    DOLPHOT base + module tarball set (so we can skip re-downloading sources).
    """
    dest_dir = Path(dest_dir).resolve()
    expected_modules = (
        list(module_tarballs)
        if module_tarballs is not None
        else list(SOURCES_MODULES)
    )
    make_root = dolphot_make_root(dest_dir)
    if not (make_root / "Makefile").is_file():
        return False
    stamp = _stamp_root(dest_dir) / _SOURCES_STAMP_NAME
    if not stamp.is_file():
        return False
    try:
        data = json.loads(stamp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        data.get("dolphot_version") == DOLPHOT_VERSION
        and data.get("base_tarball") == SOURCES_BASE
        and data.get("module_tarballs") == expected_modules
    )


def write_sources_stamp(
    dest_dir: Path,
    module_tarballs: list[str] | None = None,
) -> None:
    """Write sources completion stamp after a successful base+modules install."""
    dest_dir = Path(dest_dir).resolve()
    mods = list(module_tarballs) if module_tarballs is not None else list(SOURCES_MODULES)
    root = _stamp_root(dest_dir)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "dolphot_version": DOLPHOT_VERSION,
        "base_tarball": SOURCES_BASE,
        "module_tarballs": mods,
    }
    stamp = root / _SOURCES_STAMP_NAME
    stamp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log.info("Wrote sources stamp %s", stamp)


def dolphot_make_root(source_dir: Path) -> Path:
    """
    Directory that contains the DOLPHOT ``Makefile`` and should be used as
    ``cwd`` for ``make``.

    After a normal install, sources are merged into ``source_dir`` with
    ``Makefile`` at the root. If only ``dolphot3.1/Makefile`` exists (nested
    tree), that subdirectory is used.
    """
    source_dir = Path(source_dir).resolve()
    if (source_dir / "Makefile").is_file():
        return source_dir
    nested = source_dir / f"dolphot{DOLPHOT_VERSION}"
    if (nested / "Makefile").is_file():
        return nested
    for mf in sorted(source_dir.glob("dolphot*/Makefile")):
        if mf.is_file():
            return mf.parent
    return source_dir


def _makefile_thread_libs_line() -> str:
    """
    ``export THREAD_LIBS= ...`` for OpenMP. On macOS, prefer Homebrew ``libomp``
    so the linker does not pick up a wrong-architecture ``libomp`` under
    ``/usr/local/lib`` (common cause of arm64 link failures).
    """
    if sys.platform != "darwin":
        return "export THREAD_LIBS= -lomp"
    for prefix in (Path("/opt/homebrew/opt/libomp"), Path("/usr/local/opt/libomp")):
        libdir = prefix / "lib"
        if libdir.is_dir() and any(libdir.glob("libomp.*")):
            return f"export THREAD_LIBS= -L{libdir} -lomp"
    return "export THREAD_LIBS= -lomp"


def apply_dolphot_source_patches(make_root: Path) -> bool:
    """
    Apply hst123-specific fixes to upstream DOLPHOT sources before ``make``.

    Upstream ``dolphot.c`` declares ``char str[82];`` in ``main`` and uses it
    with ``sprintf`` for paths derived from the output basename. Long absolute
    paths (common on macOS with Dropbox, conda envs, etc.) overflow that stack
    buffer; libc then aborts with **SIGTRAP** / "trace trap" (**``__chk_fail_overflow``**).

    This patch enlarges the buffer to **4096** bytes and is **idempotent**
    (safe to run on every install).

    Parameters
    ----------
    make_root : pathlib.Path
        Directory containing ``Makefile`` and ``dolphot.c`` (see ``dolphot_make_root``).

    Returns
    -------
    bool
        True if ``dolphot.c`` was modified, False if already patched or pattern
        not found.
    """
    make_root = Path(make_root).resolve()
    path = make_root / "dolphot.c"
    if not path.is_file():
        log.debug("apply_dolphot_source_patches: no %s", path)
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    marker = "hst123_dolphot_main_str_buf"
    if marker in text:
        log.debug("apply_dolphot_source_patches: already applied (%s)", path)
        return False
    # Match upstream DOLPHOT 3.1 main() opening; tolerate spaces after `{`.
    pattern = re.compile(
        r"(int main\(int argc,char\*\*argv\) \{\s*)char str\[82\];",
        re.MULTILINE,
    )
    repl = (
        r"\1/* "
        + marker
        + ": avoid sprintf stack overflow on long absolute output paths (macOS). */\n"
        r"   char str[4096];"
    )
    new_text, n_sub = pattern.subn(repl, text, count=1)
    if n_sub == 0:
        if "char str[4096]" in text:
            log.debug(
                "apply_dolphot_source_patches: dolphot.c already has enlarged buffer"
            )
            return False
        log.warning(
            "Could not apply hst123 dolphot.c buffer patch (expected "
            "`char str[82];` after main). Long absolute output paths may crash DOLPHOT."
        )
        return False
    path.write_text(new_text, encoding="utf-8")
    log.info("Applied hst123 patch: %s main() stack buffer 82 -> 4096 bytes", path.name)
    _progress(
        "  Source patch: dolphot.c — enlarged main() path buffer (long absolute paths)."
    )
    return True


def apply_calcsky_source_patches(make_root: Path) -> bool:
    """
    Enlarge ``calcsky.c`` ``main`` stack buffer used for ``sprintf`` path assembly.

    Upstream ``calcsky.c`` declares ``char str[81];`` and does::

        sprintf(str, "%s.fits", argv[1]);
        sprintf(str, "%s.sky.fits", argv[1]);

    So the input basename ``argv[1]`` must stay short (roughly **≤71** characters
    for the second call). Long work directories (Dropbox, conda envs) overflow
    the buffer; on macOS hardened libc this aborts with **SIGTRAP**
    (``__chk_fail_overflow``), often mistaken for an OpenMP issue.

    This patch enlarges the buffer to **4096** bytes (same strategy as
    :func:`apply_dolphot_source_patches`) and is **idempotent**.

    Parameters
    ----------
    make_root : pathlib.Path
        Directory containing ``Makefile`` and ``calcsky.c``.

    Returns
    -------
    bool
        True if ``calcsky.c`` was modified, False if already patched or pattern
        not found.
    """
    make_root = Path(make_root).resolve()
    path = make_root / "calcsky.c"
    if not path.is_file():
        log.debug("apply_calcsky_source_patches: no %s", path)
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    marker = "hst123_calcsky_main_str_buf"
    if marker in text:
        log.debug("apply_calcsky_source_patches: already applied (%s)", path)
        return False
    pattern = re.compile(
        r"(int main\(int argc,char\*\*argv\) \{\s*)char str\[81\];",
        re.MULTILINE,
    )
    repl = (
        r"\1/* "
        + marker
        + ": avoid sprintf stack overflow on long absolute paths (macOS). */\n"
        r"   char str[4096];"
    )
    new_text, n_sub = pattern.subn(repl, text, count=1)
    if n_sub == 0:
        if marker in text or (
            path.name == "calcsky.c" and "char str[4096]" in text
        ):
            log.debug(
                "apply_calcsky_source_patches: calcsky.c already has enlarged buffer"
            )
            return False
        log.warning(
            "Could not apply hst123 calcsky.c buffer patch (expected "
            "`char str[81];` after main). Long paths may crash calcsky with SIGTRAP."
        )
        return False
    path.write_text(new_text, encoding="utf-8")
    log.info("Applied hst123 patch: %s main() stack buffer 81 -> 4096 bytes", path.name)
    _progress(
        "  Source patch: calcsky.c — enlarged main() path buffer (long absolute paths)."
    )
    return True


def configure_dolphot_makefile(
    make_root: Path,
    *,
    threaded: bool = True,
    enable_extended_modules: bool = True,
) -> None:
    """
    Uncomment DOLPHOT ``Makefile`` ``export`` lines for multithreaded OpenMP
    build and instrument modules (matches upstream commented blocks).

    On macOS, uses ``-Xclang -fopenmp`` in ``THREAD_CFLAGS`` as recommended
    for Apple clang. Requires libomp (e.g. ``brew install libomp``) for
    ``-lomp``.
    """
    make_root = Path(make_root).resolve()
    makefile = make_root / "Makefile"
    if not makefile.is_file():
        raise FileNotFoundError(f"No DOLPHOT Makefile at {makefile}")

    _progress("")
    _progress("— Configure Makefile (modules + threading) —")
    _progress(f"  File: {makefile}")

    text = makefile.read_text(encoding="utf-8", errors="replace")
    text = text.replace("\r\n", "\n")

    use_vars: tuple[str, ...] = _MAKEFILE_USE_VARS_HST
    if enable_extended_modules:
        use_vars = _MAKEFILE_USE_VARS_HST + _MAKEFILE_USE_VARS_EXTENDED

    for var in use_vars:
        # Exact upstream form: #export USEACS=1
        old = f"#export {var}=1"
        new = f"export {var}=1"
        if old in text:
            text = text.replace(old, new)
        elif new not in text:
            log.warning("Makefile missing expected line %r (skip)", old)

    if threaded:
        pairs = [
            ("#export THREADED=1", "export THREADED=1"),
        ]
        darwin = sys.platform == "darwin"
        if darwin:
            pairs.append(
                (
                    "#export THREAD_CFLAGS= -DDOLPHOT_THREADED -D_REENTRANT -fopenmp",
                    "export THREAD_CFLAGS= -DDOLPHOT_THREADED -D_REENTRANT -Xclang -fopenmp",
                )
            )
            tl = _makefile_thread_libs_line()
            if "-L" in tl:
                _progress(f"  Makefile: THREAD_LIBS -> {tl.strip()}")
            else:
                _progress(
                    "  Makefile: THREAD_CFLAGS uses -Xclang -fopenmp (Apple clang). "
                    "For arm64, run: brew install libomp (then re-run this installer "
                    "or set THREAD_LIBS to -L/opt/homebrew/opt/libomp/lib -lomp)."
                )
        else:
            pairs.append(
                (
                    "#export THREAD_CFLAGS= -DDOLPHOT_THREADED -D_REENTRANT -fopenmp",
                    "export THREAD_CFLAGS= -DDOLPHOT_THREADED -D_REENTRANT -fopenmp",
                )
            )
            tl = "export THREAD_LIBS= -lomp"
        if darwin:
            pairs.append(
                ("#export THREAD_LIBS= -lomp", tl),
            )
        else:
            pairs.append(
                ("#export THREAD_LIBS= -lomp", "export THREAD_LIBS= -lomp"),
            )
        for old, new in pairs:
            if old in text:
                text = text.replace(old, new)
            elif new not in text:
                log.warning("Makefile missing expected line %r (skip)", old)
        # Fix reruns / hand-edited Makefiles: bare -lomp on macOS without -L
        if threaded and sys.platform == "darwin":
            tl = _makefile_thread_libs_line()
            bare = "export THREAD_LIBS= -lomp"
            if bare in text and tl != bare:
                text = text.replace(bare, tl)

    makefile.write_text(text, encoding="utf-8")
    log.info("Configured DOLPHOT Makefile at %s", makefile)
    _progress(
        "  Enabled: "
        + (f"threaded OpenMP, " if threaded else "")
        + ", ".join(use_vars)
    )


def _psf_stamp_path(source_dir: Path, archive_name: str) -> Path:
    return _stamp_root(source_dir) / "psf" / f"{archive_name}.installed"


def write_psf_stamp(source_dir: Path, archive_name: str) -> None:
    """Record that ``archive_name`` was downloaded and extracted."""
    p = _psf_stamp_path(source_dir, archive_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("", encoding="utf-8")


def psf_install_recorded(source_dir: Path, archive_name: str) -> bool:
    return _psf_stamp_path(source_dir, archive_name).is_file()


def _path_parts_lower(path: Path) -> set[str]:
    return {p.lower() for p in path.parts}


def _psf_stem(archive_name: str) -> str:
    return archive_name.replace(".tar.gz", "")


def _scan_psf_disk_index(source_dir: Path) -> tuple[list[Path], list[Path]]:
    """
    Non-empty ``*.psf`` and likely PAM ``*.fits`` (name contains 'pam') under
    ``source_dir``. Built once per ``install_psfs`` run.
    """
    source_dir = Path(source_dir).resolve()
    psfs: list[Path] = []
    pam_fits: list[Path] = []
    try:
        for p in source_dir.rglob("*"):
            if not p.is_file():
                continue
            try:
                if p.stat().st_size == 0:
                    continue
            except OSError:
                continue
            suf = p.suffix.lower()
            if suf == ".psf":
                psfs.append(p)
            elif suf in (".fits", ".fit") and "pam" in p.name.lower():
                pam_fits.append(p)
    except OSError:
        pass
    return psfs, pam_fits


def _match_acs_wfc_psf(path: Path, token: str) -> bool:
    if path.suffix.lower() != ".psf":
        return False
    if token.lower() not in path.name.lower():
        return False
    parts = _path_parts_lower(path)
    if "wfpc2" in parts or "uvis" in parts:
        return False
    # ACS/WFC PSFs: e.g. .../WFC/..., .../acs/..., .../dolphot2.0/acs/data/...
    if "wfc" in parts or "acs" in parts:
        return True
    return False


def _match_wfc3_ir_psf(path: Path, token: str) -> bool:
    """
    DOLPHOT 2.x often uses ``dolphot2.0/wfc3/data/F105W.ir.psf`` (no ``IR/`` path
    segment). IR and UVIS filter names do not overlap in ``PSF_FILES``, so we can
    use token set + path/filename hints.
    """
    if path.suffix.lower() != ".psf":
        return False
    if token not in WFC3_IR_PSF_TOKENS:
        return False
    if token.lower() not in path.name.lower():
        return False
    parts = _path_parts_lower(path)
    if "wfpc2" in parts or "uvis" in parts:
        return False
    nl = path.name.lower()
    if ".uvis." in nl and ".ir." not in nl:
        return False
    sl = path.as_posix().lower()
    if "/uvis/" in sl:
        return False
    # Path contains wfc3, explicit IR directory, or IR naming in the filename.
    if (
        "wfc3" in sl
        or "ir" in parts
        or ".ir." in nl
        or nl.endswith(".ir.psf")
    ):
        return True
    return False


def _match_wfc3_uvis_psf(path: Path, token: str) -> bool:
    if path.suffix.lower() != ".psf":
        return False
    if token not in WFC3_UVIS_PSF_TOKENS:
        return False
    if token.lower() not in path.name.lower():
        return False
    parts = _path_parts_lower(path)
    if "wfpc2" in parts:
        return False
    nl = path.name.lower()
    if ".ir." in nl and ".uvis." not in nl:
        return False
    sl = path.as_posix().lower()
    if "uvis" in parts or "/uvis/" in sl or ".uvis." in nl:
        return True
    # UVIS PSFs under legacy ``.../wfc3/data/`` without ``uvis`` in the path
    # (token is UVIS-only, so it cannot be an IR PSF).
    if "wfc3" in sl:
        return True
    return False


def _match_wfpc2_psf(path: Path, token: str) -> bool:
    if path.suffix.lower() != ".psf":
        return False
    if token.lower() not in path.name.lower():
        return False
    parts = _path_parts_lower(path)
    return "wfpc2" in parts


def _match_pam_fits(path: Path, scope: str) -> bool:
    parts = _path_parts_lower(path)
    if scope == "acs_wfc":
        if "wfpc2" in parts or "uvis" in parts:
            return False
        return "wfc" in parts or "acs" in parts
    if scope == "wfc3_ir":
        sl = path.as_posix().lower()
        if "wfpc2" in parts or "/uvis/" in sl or "uvis" in parts:
            return False
        return "ir" in parts or "wfc3" in sl
    if scope == "wfc3_uvis":
        sl = path.as_posix().lower()
        if "wfpc2" in parts:
            return False
        return "uvis" in parts or "/uvis/" in sl
    return False


def psf_archive_payload_present(
    archive_name: str,
    *,
    psf_paths: list[Path],
    pam_paths: list[Path],
    make_root: Path | None = None,
) -> bool:
    """
    True if this archive's PSF/PAM data already exists under the DOLPHOT tree
    (e.g. from an older DOLPHOT layout or a prior manual extract), even when
    our ``.installed`` stamp is missing.
    """
    stem = _psf_stem(archive_name)

    if stem == "ACS_WFC_PAM":
        if make_root is not None and _acs_wfc_pam_payload_dir_ok(make_root):
            return True
        return any(_match_pam_fits(p, "acs_wfc") for p in pam_paths)
    if stem.startswith("ACS_WFC_F") or (
        stem.startswith("ACS_WFC_") and not stem.endswith("PAM")
    ):
        token = stem.split("_")[-1]
        return any(_match_acs_wfc_psf(p, token) for p in psf_paths)

    if stem == "WFC3_IR_PAM":
        return any(_match_pam_fits(p, "wfc3_ir") for p in pam_paths)
    if stem.startswith("WFC3_IR_F") or (
        stem.startswith("WFC3_IR_") and not stem.endswith("PAM")
    ):
        token = stem.split("_")[-1]
        return any(_match_wfc3_ir_psf(p, token) for p in psf_paths)

    if stem == "WFC3_UVIS_PAM":
        return any(_match_pam_fits(p, "wfc3_uvis") for p in pam_paths)
    if stem.startswith("WFC3_UVIS_F") or (
        stem.startswith("WFC3_UVIS_") and not stem.endswith("PAM")
    ):
        token = stem.split("_")[-1]
        return any(_match_wfc3_uvis_psf(p, token) for p in psf_paths)

    if stem.startswith("WFPC2_F") or stem.startswith("WFPC2_"):
        token = stem.split("_")[-1]
        return any(_match_wfpc2_psf(p, token) for p in psf_paths)

    return False


def _psf_already_satisfied(
    source_dir: Path,
    archive_name: str,
    *,
    force_download: bool,
    psf_paths: list[Path],
    pam_paths: list[Path],
    make_root: Path,
) -> bool:
    if force_download:
        return False
    # Stale stamp or wrong extract dir: do not skip ACS PAM unless files exist
    # under ``make_root/acs/data`` (where ``acsmask`` reads them) or under the
    # install ``source_dir`` (same layout when Makefile lives in a nested
    # ``dolphot*`` dir but PAM was placed at the install root).
    if _psf_stem(archive_name) == "ACS_WFC_PAM":
        if not (
            _acs_wfc_pam_payload_dir_ok(make_root)
            or _acs_wfc_pam_payload_dir_ok(source_dir)
        ):
            return False
    if psf_install_recorded(source_dir, archive_name):
        return True
    return psf_archive_payload_present(
        archive_name,
        psf_paths=psf_paths,
        pam_paths=pam_paths,
        make_root=make_root,
    )


PSF_FILES = {
    "ACS": [
        "ACS_WFC_PAM.tar.gz",
        "ACS_WFC_F435W.tar.gz",
        "ACS_WFC_F475W.tar.gz",
        "ACS_WFC_F502N.tar.gz",
        "ACS_WFC_F550M.tar.gz",
        "ACS_WFC_F555W.tar.gz",
        "ACS_WFC_F606W.tar.gz",
        "ACS_WFC_F625W.tar.gz",
        "ACS_WFC_F658N.tar.gz",
        "ACS_WFC_F660N.tar.gz",
        "ACS_WFC_F775W.tar.gz",
        "ACS_WFC_F814W.tar.gz",
        "ACS_WFC_F850LP.tar.gz",
        "ACS_WFC_F892N.tar.gz",
    ],
    "WFC3": [
        "WFC3_IR_PAM.tar.gz",
        "WFC3_IR_F105W.tar.gz",
        "WFC3_IR_F110W.tar.gz",
        "WFC3_IR_F125W.tar.gz",
        "WFC3_IR_F140W.tar.gz",
        "WFC3_IR_F160W.tar.gz",
        "WFC3_UVIS_PAM.tar.gz",
        "WFC3_UVIS_F438W.tar.gz",
        "WFC3_UVIS_F475W.tar.gz",
        "WFC3_UVIS_F555W.tar.gz",
        "WFC3_UVIS_F606W.tar.gz",
        "WFC3_UVIS_F775W.tar.gz",
        "WFC3_UVIS_F814W.tar.gz",
        "WFC3_UVIS_F850LP.tar.gz",
    ],
    "WFPC2": [
        "WFPC2_F555W.tar.gz",
        "WFPC2_F606W.tar.gz",
        "WFPC2_F814W.tar.gz",
        "WFPC2_F450W.tar.gz",
        "WFPC2_F439W.tar.gz",
        "WFPC2_F850LP.tar.gz",
    ],
}

# WFC3 IR vs UVIS filter names do not overlap; used to detect on-disk PSFs.
WFC3_IR_PSF_TOKENS = frozenset(
    _psf_stem(fn).split("_")[-1]
    for fn in PSF_FILES["WFC3"]
    if fn.startswith("WFC3_IR_F")
)
WFC3_UVIS_PSF_TOKENS = frozenset(
    _psf_stem(fn).split("_")[-1]
    for fn in PSF_FILES["WFC3"]
    if fn.startswith("WFC3_UVIS_F")
)


def _url_for(filename):
    """Return full URL for a file on the DOLPHOT server."""
    return DOLPHOT_BASE_URL + filename


def download_file(url, dest_path, timeout=60, *, step_label: str | None = None):
    """
    Download a URL to dest_path. Overwrites if exists.

    Parameters
    ----------
    url : str
        Full URL to download.
    dest_path : str or path-like
        Local path to write the file.
    timeout : int
        Request timeout in seconds.
    step_label : str, optional
        Short description for progress output (e.g. ``"[1/4] Base sources"``).

    Raises
    ------
    URLError, HTTPError
        On download failure.
    """
    dest_path = Path(dest_path)
    label = step_label or "Download"
    _progress(f"{label}")
    _progress(f"  URL:    {url}")
    _progress(f"  Saving: {dest_path}")
    req = Request(url, headers={"User-Agent": "hst123-dolphot-install/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        cl = resp.headers.get("Content-Length")
        if cl and cl.isdigit():
            _progress(f"  Expected size: {_fmt_bytes(int(cl))}")
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        body = resp.read()
        with open(dest_path, "wb") as f:
            f.write(body)
    size = dest_path.stat().st_size
    _progress(f"  Done: {_fmt_bytes(size)} written.")
    log.info("Downloaded %s -> %s (%s)", url, dest_path, _fmt_bytes(size))


def extract_tar(
    tar_path,
    dest_dir,
    strip_top_level=True,
    *,
    step_label: str | None = None,
):
    """
    Extract a tarball into dest_dir.

    If strip_top_level is True and the archive contains a single top-level
    directory, its contents are extracted into dest_dir; otherwise members
    are extracted with their path as given.

    Parameters
    ----------
    tar_path : str or path-like
        Path to .tar.gz file.
    dest_dir : str or path-like
        Directory to extract into (created if needed).
    strip_top_level : bool
        If True, strip one leading path component when the archive has
        a single top-level directory.
    step_label : str, optional
        Prefix for progress line (default: archive basename).
    """
    tar_path = Path(tar_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    label = step_label or f"Extract {tar_path.name}"
    mode = "strip top-level directory" if strip_top_level else "keep paths"
    _progress(f"  {label} -> {dest_dir} ({mode})")
    with tarfile.open(tar_path, "r:gz") as tar:
        names = tar.getnames()
        stripped = False
        if strip_top_level and names:
            parts = {n.split("/")[0] for n in names if n}
            if len(parts) == 1:
                prefix = next(iter(parts)) + "/"
                for m in tar.getmembers():
                    if m.name.startswith(prefix):
                        m.name = m.name[len(prefix) :].lstrip("/")
                        if m.name:
                            tar.extract(m, dest_dir)
                stripped = True
        if not stripped:
            tar.extractall(dest_dir)
    _progress(f"  Extracted OK.")
    log.info("Extracted %s -> %s", tar_path, dest_dir)


def install_sources(
    dest_dir,
    timeout=60,
    *,
    force_download: bool = False,
    module_tarballs: list[str] | None = None,
):
    """
    Download and extract DOLPHOT base and instrument modules into dest_dir.

    Creates dest_dir if needed. Base tarball is extracted first; then each
    module tarball is extracted into the same tree (overlaying / merging).

    If a previous run completed successfully (stamp + Makefile), skips all
    source downloads unless ``force_download`` is True.

    Parameters
    ----------
    dest_dir : str or path-like
        Directory that will contain the DOLPHOT source tree (e.g. a
        versioned subdir will be created by the base tarball).
    timeout : int
        Download timeout in seconds.
    force_download : bool
        If True, always download base + modules (ignore completion stamp).
    module_tarballs : list of str, optional
        Module ``.tar.gz`` names to fetch (default: ``SOURCES_MODULES``).

    Returns
    -------
    pathlib.Path
        Path to the install root (``dest_dir``); use ``dolphot_make_root`` for
        the directory containing ``Makefile``.
    """
    dest_dir = Path(dest_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    mods = list(module_tarballs) if module_tarballs is not None else list(SOURCES_MODULES)

    n_steps = 1 + len(mods)
    _progress("")
    _progress("— DOLPHOT source code (base + instrument modules) —")
    _progress(f"  Version: DOLPHOT {DOLPHOT_VERSION}")
    _progress(f"  Target:  {dest_dir}")
    _progress(f"  Modules: {len(mods)} archive(s) after base")

    if not force_download and sources_install_is_complete(
        dest_dir, module_tarballs=mods
    ):
        _progress("  Sources already installed for this DOLPHOT version (skipping download).")
        _progress("  Use --force-download to re-fetch base and module archives.")
        log.info("Skipping source download; stamp matches %s", dest_dir)
        return dest_dir

    # Base: extract so contents land in dest_dir (strip top-level dir if present)
    base_url = _url_for(SOURCES_BASE)
    base_tar = dest_dir / SOURCES_BASE
    download_file(
        base_url,
        base_tar,
        timeout=timeout,
        step_label=f"[1/{n_steps}] Download base archive ({SOURCES_BASE})",
    )
    extract_tar(base_tar, dest_dir, strip_top_level=True, step_label="Unpack base")
    source_dir = dest_dir

    # Modules: extract into the same source tree (merged with base)
    for i, mod in enumerate(mods, start=2):
        url = _url_for(mod)
        local = dest_dir / mod
        download_file(
            url,
            local,
            timeout=timeout,
            step_label=f"[{i}/{n_steps}] Download module ({mod})",
        )
        extract_tar(local, source_dir, strip_top_level=True, step_label=f"Unpack {mod}")
        local.unlink(missing_ok=True)
    base_tar.unlink(missing_ok=True)
    write_sources_stamp(dest_dir, module_tarballs=mods)
    mr = dolphot_make_root(source_dir)
    _progress(f"Sources ready (Makefile: {mr / 'Makefile'}).")

    return source_dir


def install_psfs(
    source_dir,
    instruments=None,
    all_psfs=False,
    one_per_instrument=False,
    timeout=60,
    *,
    force_download: bool = False,
):
    """
    Download and extract PSF (and PAM) tarballs into the DOLPHOT source directory.

    Skips download/extract for any archive that was already installed in a
    previous run (per-archive stamp under ``.hst123-dolphot/psf/``), unless
    ``force_download`` is True.

    Parameters
    ----------
    source_dir : str or path-like
        Path to the DOLPHOT source directory (Makefile directory).
    instruments : list of str, optional
        Instruments to install PSFs for: "ACS", "WFC3", "WFPC2".
        If None, defaults to all three.
    all_psfs : bool
        If True, download full PSF list per instrument; else only one per instrument.
    one_per_instrument : bool
        If True, download exactly one PSF per instrument (for testing).
    timeout : int
        Download timeout in seconds.
    force_download : bool
        If True, re-download and re-extract every planned archive.
    """
    source_dir = Path(source_dir).resolve()
    make_root = dolphot_make_root(source_dir)
    if instruments is None:
        instruments = list(ONE_PSF_PER_INSTRUMENT)
    allowed = set(ONE_PSF_PER_INSTRUMENT)
    for inv in instruments:
        if inv not in allowed:
            raise ValueError(f"Unknown instrument {inv!r}; allowed: {sorted(allowed)}")

    if one_per_instrument:
        # ACS acsmask requires wfc1_pam.fits + wfc2_pam.fits from ACS_WFC_PAM.tar.gz
        # in addition to a band-specific PSF tarball.
        files_to_get = []
        for inv in instruments:
            if inv == "ACS":
                files_to_get.append("ACS_WFC_PAM.tar.gz")
            files_to_get.append(ONE_PSF_PER_INSTRUMENT[inv])
    elif all_psfs:
        files_to_get = []
        for inv in instruments:
            files_to_get.extend(PSF_FILES.get(inv, []))
    else:
        files_to_get = [ONE_PSF_PER_INSTRUMENT[inv] for inv in instruments]

    seen = set()
    ordered = []
    for filename in files_to_get:
        if filename in seen:
            continue
        seen.add(filename)
        ordered.append(filename)
    psf_paths, pam_paths = _scan_psf_disk_index(make_root)
    total = len(ordered)
    n_cached = sum(
        1
        for f in ordered
        if _psf_already_satisfied(
            source_dir,
            f,
            force_download=force_download,
            psf_paths=psf_paths,
            pam_paths=pam_paths,
            make_root=make_root,
        )
    )
    n_to_fetch = total - n_cached
    _progress("")
    _progress("— PSF / PAM reference data —")
    _progress(
        f"  Instruments: {', '.join(instruments)}  |  "
        f"{total} archive(s) in plan ({n_cached} already present, {n_to_fetch} to download)"
    )
    _progress(f"  Into: {make_root} (DOLPHOT Makefile root)")

    for idx, filename in enumerate(ordered, start=1):
        psf_paths, pam_paths = _scan_psf_disk_index(make_root)
        if _psf_already_satisfied(
            source_dir,
            filename,
            force_download=force_download,
            psf_paths=psf_paths,
            pam_paths=pam_paths,
            make_root=make_root,
        ):
            if psf_install_recorded(source_dir, filename):
                _progress(
                    f"[PSF {idx}/{total}] {filename} — already installed (stamp, skip download)"
                )
                log.info("Skipping PSF archive (stamp): %s", filename)
            else:
                write_psf_stamp(source_dir, filename)
                _progress(
                    f"[PSF {idx}/{total}] {filename} — PSF/PAM files already on disk "
                    "(skip download; wrote stamp)"
                )
                log.info("Skipping PSF archive (existing files under tree): %s", filename)
            if _psf_stem(filename) == "ACS_WFC_PAM":
                relocate_acs_wfc_pam_into_canonical_layout(make_root)
            continue
        url = _url_for(filename)
        local = source_dir / filename
        try:
            download_file(
                url,
                local,
                timeout=timeout,
                step_label=f"[PSF {idx}/{total}] {filename}",
            )
            extract_tar(
                local,
                make_root,
                strip_top_level=True,
                step_label=f"Unpack {filename}",
            )
            if _psf_stem(filename) == "ACS_WFC_PAM":
                if not relocate_acs_wfc_pam_into_canonical_layout(make_root):
                    log.warning(
                        "ACS_WFC_PAM.tar.gz was extracted but wfc1_pam.fits / wfc2_pam.fits "
                        "were not found under %s (or dolphot2.0/acs/data). "
                        "acsmask / Python ACS mask may fail until these exist.",
                        make_root / "acs" / "data",
                    )
            write_psf_stamp(source_dir, filename)
        except (URLError, HTTPError) as e:
            log.warning("Failed to download %s: %s", filename, e)
            raise
        finally:
            if local.exists():
                local.unlink()
    # PSFs may exist only under legacy ``.../dolphot2.0/...`` (nested under the
    # Makefile root or sibling next to ``dolphot3.1``, common conda layout).
    # Always merge ACS / WFC3 / WFPC2 ``*.psf`` into the canonical tree for runtime.
    relocate_all_legacy_psf_into_canonical_layout(make_root)
    _progress("PSF / PAM step finished.")


def run_make(
    make_root: Path,
    jobs: int | None = None,
    targets: list[str] | None = None,
    *,
    always_make: bool = False,
) -> None:
    """
    Run ``make`` in the DOLPHOT tree root (directory containing ``Makefile``).

    Parameters
    ----------
    make_root : pathlib.Path
        Same as ``dolphot_make_root(install_dir)`` after extracting base + modules.
    jobs : int, optional
        If set, ``make -j <jobs>`` is used.
    targets : list of str, optional
        If set, only these Makefile targets are built (e.g. ``["bin/calcsky"]``).
        If ``None``, runs a full ``make`` (default ``all``).
    always_make : bool, optional
        If True, pass ``make -B`` (unconditionally remake), used for calcsky-only
        rebuilds so the binary is re-linked even when timestamps look current.
    """
    make_root = Path(make_root).resolve()
    makefile = make_root / "Makefile"
    if not makefile.is_file():
        raise FileNotFoundError(
            f"No Makefile in {make_root}; extract DOLPHOT sources first."
        )
    cmd = ["make"]
    if always_make:
        cmd.append("-B")
    if jobs is not None and jobs > 0:
        cmd.extend(["-j", str(jobs)])
    if targets:
        cmd.extend(targets)
    _progress("")
    _progress("— Build (make) —")
    _progress(f"  Directory: {make_root}")
    _progress(f"  Command:   {' '.join(cmd)}")
    _progress("  (compiler output follows)")
    log.info("Running in %s: %s", make_root, " ".join(cmd))
    run_external_command(cmd, cwd=str(make_root), log=log, check=True)
    _progress("  make completed successfully.")


def _calcsky_make_target(make_root: Path) -> str:
    """
    Return the Makefile target name for ``calcsky`` (``bin/calcsky`` vs ``calcsky``).

    DOLPHOT 3.x typically defines ``bin/calcsky``; older layouts may use ``calcsky``.
    """
    mf = Path(make_root).resolve() / "Makefile"
    if not mf.is_file():
        return "bin/calcsky"
    text = mf.read_text(encoding="utf-8", errors="replace")
    for pat in (r"^\s*(bin/calcsky)\s*:", r"^\s*(calcsky)\s*:"):
        m = re.search(pat, text, re.MULTILINE)
        if m:
            return m.group(1)
    return "bin/calcsky"


def rebuild_calcsky_in_tree(
    make_root: Path,
    jobs: int | None = None,
    *,
    apply_patch: bool = True,
) -> None:
    """
    Apply the hst123 ``calcsky.c`` buffer patch and rebuild only ``calcsky``.

    Does not run a full ``make``; use after a normal install when you need an
    updated ``calcsky`` binary (e.g. long-path ``sprintf`` fix). CLI:
    ``hst123-install-dolphot --calcsky-only``.
    """
    make_root = Path(make_root).resolve()
    if apply_patch:
        apply_calcsky_source_patches(make_root)
    tgt = _calcsky_make_target(make_root)
    try:
        run_make(
            make_root, jobs=jobs, targets=[tgt], always_make=True
        )
    except subprocess.CalledProcessError:
        if tgt == "bin/calcsky":
            log.warning(
                "make target bin/calcsky failed; retrying with target calcsky"
            )
            run_make(
                make_root, jobs=jobs, targets=["calcsky"], always_make=True
            )
        else:
            raise


def link_executables_to_conda_bin(
    source_dir: Path,
    conda_prefix: Path | None = None,
    *,
    force: bool = False,
    only: tuple[str, ...] | None = None,
) -> list[str]:
    """
    Symlink (or copy) DOLPHOT executables from ``source_dir`` into
    ``<conda_prefix>/bin`` so they are on PATH when the env is active.

    Parameters
    ----------
    source_dir : pathlib.Path
        DOLPHOT build root (directory with ``Makefile``). Binaries are taken
        from ``source_dir/bin/`` when present (DOLPHOT 3.x), else ``source_dir/``.
    conda_prefix : pathlib.Path, optional
        Conda environment root; default ``CONDA_PREFIX``.
    force : bool
        Replace existing files/symlinks in ``bin``.
    only : tuple of str, optional
        If set, link only these executable names (e.g. ``("calcsky",)``).

    Returns
    -------
    list of str
        Names successfully linked.
    """
    if conda_prefix is None:
        conda_prefix = get_conda_prefix()
    if conda_prefix is None:
        log.warning("CONDA_PREFIX not set; skipping link to conda bin.")
        _progress("Skipping conda bin links (CONDA_PREFIX not set).")
        return []
    bin_dir = conda_prefix / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    linked: list[str] = []
    source_dir = Path(source_dir).resolve()
    _progress("")
    _progress("— Install executables on PATH —")
    sample = _resolve_dolphot_executable(source_dir, "dolphot")
    from_dir = sample.parent if sample is not None else source_dir
    _progress(f"  From: {from_dir}")
    _progress(f"  To:   {bin_dir}")
    names: tuple[str, ...] = only if only is not None else DOLPHOT_EXECUTABLES
    for name in names:
        src = _resolve_dolphot_executable(source_dir, name)
        if src is None:
            log.warning(
                "Executable not found (build may have failed): %s or %s",
                source_dir / "bin" / name,
                source_dir / name,
            )
            _progress(f"  ! missing (skip): {name}")
            continue
        dst = bin_dir / name
        if dst.exists() or dst.is_symlink():
            if not force:
                log.info("Skip %s (already exists in %s)", name, bin_dir)
                _progress(f"  skip (exists): {name}")
                continue
            dst.unlink()
        try:
            os.symlink(src, dst)
            _progress(f"  symlink: {name}")
        except OSError as e:
            log.debug("Symlink failed (%s); copying instead", e)
            shutil.copy2(src, dst)
            os.chmod(dst, dst.stat().st_mode | 0o111)
            _progress(f"  copy:    {name} (symlink not available)")
        linked.append(name)
        log.info("Installed %s -> %s", src, dst)
    if linked:
        _progress(f"  Linked/copied {len(linked)} program(s).")
    return linked


def main():
    """Entry point for the hst123-install-dolphot script."""
    conda = get_conda_prefix()
    default_dir = default_dolphot_install_dir()
    parser = argparse.ArgumentParser(
        description="Download DOLPHOT sources, PSF reference data, build, and "
        "optionally link executables into the active conda env's bin/. "
        "Default install path is $CONDA_PREFIX/opt/hst123-dolphot when "
        "CONDA_PREFIX is set, otherwise ./dolphot.",
        epilog="DOLPHOT: http://americano.dolphinsim.com/dolphot/",
    )
    parser.add_argument(
        "--dolphot-dir",
        type=Path,
        default=None,
        help=(
            "Directory for DOLPHOT tree (sources, PSFs, built binaries). "
            f"Default: {default_dir} "
            + ("(conda env)" if conda else "(cwd ./dolphot)")
        ),
    )
    parser.add_argument(
        "--psf-only",
        action="store_true",
        help="Only download PSF files; do not download base or module sources. "
        "--dolphot-dir must point to an existing DOLPHOT source directory (with Makefile).",
    )
    parser.add_argument(
        "--calcsky-only",
        action="store_true",
        help=(
            "Only apply the hst123 calcsky.c buffer patch and rebuild the calcsky "
            "binary (no source downloads, no PSF steps). Requires an existing DOLPHOT "
            "tree at --dolphot-dir (default: $CONDA_PREFIX/opt/hst123-dolphot). "
            "Use --force-bin to replace calcsky in conda bin."
        ),
    )
    parser.add_argument(
        "--instruments",
        nargs="+",
        choices=["ACS", "WFC3", "WFPC2"],
        default=["ACS", "WFC3", "WFPC2"],
        help="Instruments to install PSFs for (default: all)",
    )
    parser.add_argument(
        "--all-psfs",
        action="store_true",
        help="Download full set of PSF files per instrument (default: one per instrument)",
    )
    parser.add_argument(
        "--no-psfs",
        action="store_true",
        help="Do not download any PSF files (only base + modules)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging (debug messages on stderr)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress step-by-step progress lines (errors still shown)",
    )
    make_grp = parser.add_mutually_exclusive_group()
    make_grp.add_argument(
        "--make",
        dest="run_make",
        action="store_true",
        help="Run make after downloading sources (default)",
    )
    make_grp.add_argument(
        "--no-make",
        dest="run_make",
        action="store_false",
        help="Do not run make (only download/extract)",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        metavar="N",
        help="Parallel make jobs (make -j N)",
    )
    link_grp = parser.add_mutually_exclusive_group()
    link_grp.add_argument(
        "--link-conda-bin",
        dest="link_conda_bin",
        action="store_true",
        help=(
            "Symlink executables into $CONDA_PREFIX/bin (default when "
            "CONDA_PREFIX is set)"
        ),
    )
    link_grp.add_argument(
        "--no-link-conda-bin",
        dest="link_conda_bin",
        action="store_false",
        help="Do not add symlinks in conda bin (leave executables only in dolphot dir)",
    )
    parser.add_argument(
        "--force-bin",
        action="store_true",
        help="Replace existing executables in conda bin when linking",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help=(
            "Re-download DOLPHOT sources and PSF archives even if a previous "
            "run already installed them (ignore local completion stamps)"
        ),
    )
    parser.add_argument(
        "--hst-modules-only",
        action="store_true",
        help=(
            "Download and build only HST modules (ACS, WFC3, WFPC2); skip "
            "Roman, NIRCam, NIRISS, MIRI, Euclid tarballs and Makefile flags"
        ),
    )
    thread_grp = parser.add_mutually_exclusive_group()
    thread_grp.add_argument(
        "--threaded",
        dest="no_threaded",
        action="store_false",
        help=(
            "Enable OpenMP threading in the Makefile "
            "(THREADED=1, THREAD_CFLAGS, THREAD_LIBS)."
        ),
    )
    thread_grp.add_argument(
        "--no-threaded",
        dest="no_threaded",
        action="store_true",
        help=(
            "Compile without OpenMP: leave Makefile threading lines commented "
            "(default)."
        ),
    )
    parser.add_argument(
        "--no-source-patches",
        action="store_true",
        help=(
            "Do not patch upstream DOLPHOT sources before build. Default: apply "
            "fixes to dolphot.c and calcsky.c (main() path buffers) so long absolute "
            "paths do not crash with SIGTRAP on macOS."
        ),
    )
    parser.set_defaults(
        run_make=True,
        link_conda_bin=(conda is not None),
        no_threaded=True,
    )
    args = parser.parse_args()

    global _quiet_progress
    _quiet_progress = args.quiet
    if args.quiet and args.verbose:
        parser.error("Use only one of --quiet and --verbose")

    ensure_cli_logging_configured(
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    dest = (args.dolphot_dir if args.dolphot_dir is not None else default_dir).resolve()

    if args.calcsky_only and args.psf_only:
        parser.error("--calcsky-only cannot be used with --psf-only")

    if args.calcsky_only:
        if not dest.is_dir():
            parser.error(
                f"--calcsky-only requires an existing DOLPHOT directory: {dest}"
            )
        make_root = dolphot_make_root(dest)
        if not (make_root / "Makefile").is_file():
            parser.error(
                f"No Makefile under {make_root}; run a full hst123-install-dolphot first."
            )
        # Ensure visible output even if the hst123 logger has no stderr handler yet.
        print(
            "hst123-install-dolphot: calcsky-only (patch + make bin/calcsky) …",
            file=sys.stderr,
            flush=True,
        )
        _progress("")
        _progress("— calcsky-only (patch + partial make) —")
        _progress(f"  Makefile root: {make_root}")
        try:
            if args.run_make:
                rebuild_calcsky_in_tree(
                    make_root,
                    jobs=args.jobs,
                    apply_patch=not args.no_source_patches,
                )
            elif not args.no_source_patches:
                apply_calcsky_source_patches(make_root)
        except subprocess.CalledProcessError as e:
            log.error("make failed with exit code %s", e.returncode)
            _progress(f"make failed with exit code {e.returncode}.")
            sys.exit(e.returncode)
        except FileNotFoundError as e:
            log.error("%s", e)
            _progress(str(e))
            sys.exit(1)
        if args.link_conda_bin:
            linked = link_executables_to_conda_bin(
                make_root, force=args.force_bin, only=("calcsky",)
            )
            if linked:
                log.info(
                    "Linked DOLPHOT executables into conda bin: %s",
                    ", ".join(linked),
                )
            elif get_conda_prefix() is None:
                path_dir = dolphot_path_for_shell(make_root)
                log.info(
                    "To use calcsky, add to PATH: export PATH=%s:$PATH",
                    path_dir,
                )
                _progress("")
                _progress("Add DOLPHOT bin to PATH:")
                _progress(f"  export PATH={path_dir}:$PATH")
        else:
            path_dir = dolphot_path_for_shell(make_root)
            log.info(
                "calcsky built under %s (not linked to conda bin)",
                path_dir,
            )
            _progress("")
            _progress("Add DOLPHOT to PATH (or use --link-conda-bin):")
            _progress(f"  export PATH={path_dir}:$PATH")
        _progress("")
        _progress("Done (--calcsky-only).")
        log.info("DOLPHOT root: %s", make_root)
        print(
            f"hst123-install-dolphot: calcsky-only finished (DOLPHOT root {make_root}).",
            file=sys.stderr,
            flush=True,
        )
        return

    _progress("hst123-install-dolphot")
    _progress(f"  Install root: {dest}")
    if conda:
        _progress(f"  Conda env:    {conda}")
    else:
        _progress("  Conda env:    (none — not linking to $CONDA_PREFIX/bin unless you use conda)")
    _progress(
        "  Plan: "
        + ("PSF archives only (--psf-only)" if args.psf_only else "full install")
        + ("" if args.psf_only else f" | PSFs: {'all filters' if args.all_psfs else 'one per instrument'}")
        + ("" if args.psf_only or not args.no_psfs else " | PSFs skipped (--no-psfs)")
        + (
            ""
            if args.psf_only
            else f" | make: {'yes' if args.run_make else 'no (--no-make)'}"
        )
        + (
            ""
            if args.psf_only
            else f" | conda bin: {'yes' if args.link_conda_bin else 'no (--no-link-conda-bin)'}"
        )
        + (" | force-download: yes" if args.force_download else "")
        + (" | HST modules only" if args.hst_modules_only else "")
        + (" | no threaded build (default)" if args.no_threaded else " | threaded OpenMP build")
        + (" | no source patches" if args.no_source_patches else " | source patches (dolphot.c + calcsky.c buffers)")
    )

    if args.psf_only:
        if not dest.is_dir():
            parser.error("--psf-only requires an existing directory: %s" % dest)
        install_psfs(
            dest,
            instruments=args.instruments,
            all_psfs=args.all_psfs,
            one_per_instrument=not args.all_psfs,
            timeout=60,
            force_download=args.force_download,
        )
        log.info("PSF installation complete in %s", dest)
        _progress("")
        _progress("Done (--psf-only).")
        log.info("DOLPHOT root: %s", dest)
        return

    module_tarballs = (
        list(SOURCES_MODULES_HST)
        if args.hst_modules_only
        else list(SOURCES_MODULES)
    )
    source_dir = install_sources(
        dest,
        timeout=60,
        force_download=args.force_download,
        module_tarballs=module_tarballs,
    )
    log.info("DOLPHOT sources installed in %s", source_dir)
    make_root = dolphot_make_root(source_dir)
    if not args.no_source_patches:
        try:
            apply_dolphot_source_patches(make_root)
            apply_calcsky_source_patches(make_root)
        except OSError as exc:
            log.warning("Could not apply DOLPHOT source patches: %s", exc)
            _progress(f"  Warning: source patch failed ({exc})")
    else:
        _progress("  Skipping source patches (--no-source-patches).")
    try:
        configure_dolphot_makefile(
            make_root,
            threaded=not args.no_threaded,
            enable_extended_modules=not args.hst_modules_only,
        )
    except FileNotFoundError as e:
        log.warning("Could not configure Makefile: %s", e)
        _progress(f"  Warning: {e}")
    if not args.no_psfs:
        install_psfs(
            source_dir,
            instruments=args.instruments,
            all_psfs=args.all_psfs,
            one_per_instrument=not args.all_psfs,
            timeout=60,
            force_download=args.force_download,
        )
        log.info("PSF / reference data installed under %s", source_dir)
    else:
        _progress("")
        _progress("Skipping PSF/PAM downloads (--no-psfs).")

    if args.run_make:
        try:
            run_make(make_root, jobs=args.jobs)
        except subprocess.CalledProcessError as e:
            log.error("make failed with exit code %s", e.returncode)
            _progress(f"make failed with exit code {e.returncode}.")
            sys.exit(e.returncode)
        except FileNotFoundError as e:
            log.error("%s", e)
            _progress(str(e))
            sys.exit(1)
    else:
        _progress("")
        _progress("Skipping build (--no-make). Run make yourself in:")
        _progress(f"  cd {make_root} && make")

    if args.link_conda_bin:
        linked = link_executables_to_conda_bin(
            make_root, force=args.force_bin
        )
        if linked:
            log.info(
                "Linked DOLPHOT executables into conda bin: %s",
                ", ".join(linked),
            )
        elif get_conda_prefix() is None:
            path_dir = dolphot_path_for_shell(make_root)
            log.info(
                "To use DOLPHOT, add to PATH: export PATH=%s:$PATH",
                path_dir,
            )
            _progress("")
            _progress("Add DOLPHOT to PATH:")
            _progress(f"  export PATH={path_dir}:$PATH")
    else:
        path_dir = dolphot_path_for_shell(make_root)
        log.info(
            "Add DOLPHOT to PATH: export PATH=%s:$PATH",
            path_dir,
        )
        _progress("")
        _progress("Add DOLPHOT to PATH:")
        _progress(f"  export PATH={path_dir}:$PATH")

    _progress("")
    _progress("All steps finished.")
    log.info("DOLPHOT root: %s", make_root)


if __name__ == "__main__":
    main()
