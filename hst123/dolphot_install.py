"""Download DOLPHOT sources and PSF files; see README for build steps."""

import argparse
import logging
import tarfile
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

log = logging.getLogger(__name__)

DOLPHOT_BASE_URL = "http://americano.dolphinsim.com/dolphot/"
DOLPHOT_VERSION = "3.0"

SOURCES_BASE = f"dolphot{DOLPHOT_VERSION}.tar.gz"
SOURCES_MODULES = [
    f"dolphot{DOLPHOT_VERSION}.ACS.tar.gz",
    f"dolphot{DOLPHOT_VERSION}.WFC3.tar.gz",
    f"dolphot{DOLPHOT_VERSION}.WFPC2.tar.gz",
]

ONE_PSF_PER_INSTRUMENT = {
    "ACS": "ACS_WFC_F435W.tar.gz",
    "WFC3": "WFC3_UVIS_F555W.tar.gz",
    "WFPC2": "WFPC2_F555W.tar.gz",
}

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


def _url_for(filename):
    """Return full URL for a file on the DOLPHOT server."""
    return DOLPHOT_BASE_URL + filename


def download_file(url, dest_path, timeout=60):
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

    Raises
    ------
    URLError, HTTPError
        On download failure.
    """
    req = Request(url, headers={"User-Agent": "hst123-dolphot-install/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(resp.read())
    log.info("Downloaded %s -> %s", url, dest_path)


def extract_tar(tar_path, dest_dir, strip_top_level=True):
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
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        names = tar.getnames()
        if strip_top_level and names:
            # Check for single top-level directory (e.g. "dolphot3.0/")
            parts = {n.split("/")[0] for n in names}
            if len(parts) == 1:
                prefix = names[0].split("/")[0] + "/"
                for m in tar.getmembers():
                    if m.name.startswith(prefix):
                        m.name = m.name[len(prefix) :].lstrip("/")
                        if m.name:
                            tar.extract(m, dest_dir)
                    continue
        tar.extractall(dest_dir)
    log.info("Extracted %s -> %s", tar_path, dest_dir)


def install_sources(dest_dir, timeout=60):
    """
    Download and extract DOLPHOT base and ACS/WFC3/WFPC2 modules into dest_dir.

    Creates dest_dir if needed. Base tarball is extracted first; then each
    module tarball is extracted into the same tree (overlaying files).

    Parameters
    ----------
    dest_dir : str or path-like
        Directory that will contain the DOLPHOT source tree (e.g. a
        versioned subdir will be created by the base tarball).
    timeout : int
        Download timeout in seconds.

    Returns
    -------
    pathlib.Path
        Path to the actual source directory (e.g. dest_dir / "dolphot3.0").
    """
    dest_dir = Path(dest_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Base: extract so contents land in dest_dir (strip top-level dir if present)
    base_url = _url_for(SOURCES_BASE)
    base_tar = dest_dir / SOURCES_BASE
    download_file(base_url, base_tar, timeout=timeout)
    extract_tar(base_tar, dest_dir, strip_top_level=True)
    source_dir = dest_dir

    # Modules: extract into the same source tree
    for mod in SOURCES_MODULES:
        url = _url_for(mod)
        local = dest_dir / mod
        download_file(url, local, timeout=timeout)
        extract_tar(local, source_dir, strip_top_level=True)
        local.unlink(missing_ok=True)
    base_tar.unlink(missing_ok=True)

    return source_dir


def install_psfs(source_dir, instruments=None, all_psfs=False, one_per_instrument=False, timeout=60):
    """
    Download and extract PSF (and PAM) tarballs into the DOLPHOT source directory.

    Parameters
    ----------
    source_dir : str or path-like
        Path to the DOLPHOT source directory (e.g. .../dolphot3.0).
    instruments : list of str, optional
        Instruments to install PSFs for: "ACS", "WFC3", "WFPC2".
        If None, defaults to all three.
    all_psfs : bool
        If True, download full PSF list per instrument; else only one per instrument.
    one_per_instrument : bool
        If True, download exactly one PSF per instrument (for testing).
    timeout : int
        Download timeout in seconds.
    """
    source_dir = Path(source_dir).resolve()
    if instruments is None:
        instruments = list(ONE_PSF_PER_INSTRUMENT)
    allowed = set(ONE_PSF_PER_INSTRUMENT)
    for inv in instruments:
        if inv not in allowed:
            raise ValueError(f"Unknown instrument {inv!r}; allowed: {sorted(allowed)}")

    if one_per_instrument:
        files_to_get = [ONE_PSF_PER_INSTRUMENT[inv] for inv in instruments]
    elif all_psfs:
        files_to_get = []
        for inv in instruments:
            files_to_get.extend(PSF_FILES.get(inv, []))
    else:
        files_to_get = [ONE_PSF_PER_INSTRUMENT[inv] for inv in instruments]

    seen = set()
    for filename in files_to_get:
        if filename in seen:
            continue
        seen.add(filename)
        url = _url_for(filename)
        local = source_dir / filename
        try:
            download_file(url, local, timeout=timeout)
            extract_tar(local, source_dir, strip_top_level=True)
        except (URLError, HTTPError) as e:
            log.warning("Failed to download %s: %s", filename, e)
            raise
        finally:
            if local.exists():
                local.unlink()


def main():
    """Entry point for the hst123-install-dolphot script."""
    parser = argparse.ArgumentParser(
        description="Download DOLPHOT sources and optional PSF files for hst123. "
        "Build with 'make' in the source directory after running this script.",
        epilog="DOLPHOT: http://americano.dolphinsim.com/dolphot/",
    )
    parser.add_argument(
        "--dolphot-dir",
        type=Path,
        default=Path.cwd() / "dolphot",
        help="Directory to install DOLPHOT into (default: ./dolphot)",
    )
    parser.add_argument(
        "--psf-only",
        action="store_true",
        help="Only download PSF files; do not download base or module sources. "
        "--dolphot-dir must point to an existing DOLPHOT source directory (e.g. dolphot3.0).",
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
        help="Verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    dest = args.dolphot_dir.resolve()
    if args.psf_only:
        if not dest.is_dir():
            parser.error("--psf-only requires an existing directory: %s" % dest)
        install_psfs(
            dest,
            instruments=args.instruments,
            all_psfs=args.all_psfs,
            one_per_instrument=not args.all_psfs,
            timeout=60,
        )
        log.info("PSF installation complete in %s", dest)
        return

    source_dir = install_sources(dest, timeout=60)
    log.info("DOLPHOT sources installed in %s", source_dir)
    if not args.no_psfs:
        install_psfs(
            source_dir,
            instruments=args.instruments,
            all_psfs=args.all_psfs,
            one_per_instrument=not args.all_psfs,
            timeout=60,
        )
    print("Next step: cd %s && make" % source_dir)


if __name__ == "__main__":
    main()
