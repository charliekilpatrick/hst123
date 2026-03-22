"""
Fetch HST calibration reference files (IDCTAB, NPOLFILE, D2IMFILE, …).

STScI serves these over **HTTPS** on ``hst-crds.stsci.edu`` and ``ssb.stsci.edu``.
Legacy **FTP** URLs are often blocked on modern networks and ``jref.old`` paths
404 on the HTTPS mirror (files live under ``jref/``, not ``jref.old/``).

This module tries, in order:

1. Local CRDS-style env dirs (``jref``, ``iref``, ``uref``) if the file exists there
2. CRDS ``unchecked_get`` URL
3. ``https://ssb.stsci.edu/cdbs/<subdir>/<file>`` (``jref.old`` → ``jref``)
4. Legacy FTP (last resort)
5. ``astropy.utils.data.download_file`` as a final fallback for odd URLs
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from astropy.utils.data import download_file as astropy_download_file
except Exception:  # pragma: no cover
    astropy_download_file = None  # type: ignore[misc, assignment]

try:
    from hst123 import __version__ as _HST123_VERSION
except Exception:  # pragma: no cover
    _HST123_VERSION = "unknown"


def cdbs_env_to_https_subdir(ref_env: str) -> str:
    """
    Map FITS/env style directory name to ssb.stsci.edu/cdbs/ subdirectory.

    Examples
    --------
    ``jref.old`` → ``jref``; ``iref.old`` → ``iref``; ``uref`` → ``uref``.
    """
    if ref_env.endswith(".old"):
        return ref_env[: -len(".old")]
    return ref_env


def ref_prefix_for_header(ref_env: str) -> str:
    """Prefix used in FITS headers before ``$`` (e.g. ``jref$file.fits``)."""
    return cdbs_env_to_https_subdir(ref_env)


def _local_reference_path(ref_env: str, ref_file: str) -> Optional[str]:
    """If ``$jref``/``$iref``/``$uref`` (etc.) contains ``ref_file``, return that path."""
    sub = cdbs_env_to_https_subdir(ref_env)
    for key in (sub, sub.upper()):
        root = os.environ.get(key)
        if not root:
            continue
        root = os.path.expanduser(root)
        candidate = os.path.join(root, ref_file)
        if os.path.isfile(candidate):
            return candidate
    return None


def build_calibration_reference_urls(
    global_defaults: Dict[str, Any], ref_env: str, ref_file: str
) -> List[str]:
    """
    Build ordered URL list for a reference basename (e.g. ``4bb1536mj_idc.fits``).
    """
    urls: List[str] = []
    crds = str(global_defaults.get("crds", "")).rstrip("/") + "/"
    if crds.startswith("http"):
        urls.append(crds + ref_file)

    cdbs_https = str(global_defaults.get("cdbs_https", "")).rstrip("/") + "/"
    if cdbs_https.startswith("http"):
        sub = cdbs_env_to_https_subdir(ref_env)
        urls.append(cdbs_https + sub + "/" + ref_file)

    cdbs_ftp = str(global_defaults.get("cdbs", "")).rstrip("/") + "/"
    if cdbs_ftp.startswith("ftp://"):
        urls.append(cdbs_ftp + ref_env + "/" + ref_file)

    # De-dupe while preserving order
    seen = set()
    out: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _requests_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=4,
        connect=3,
        backoff_factor=0.6,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s


def _http_headers() -> Dict[str, str]:
    return {
        "User-Agent": f"hst123/{_HST123_VERSION} (calibration-reference-fetch; +https://github.com)",
        "Accept": "*/*",
    }


def _download_http_stream(
    url: str, dest_path: str, timeout: Tuple[float, float], log: logging.Logger
) -> None:
    """GET url and stream to dest_path via a temp file + atomic replace."""
    dest_dir = os.path.dirname(os.path.abspath(dest_path)) or "."
    os.makedirs(dest_dir, exist_ok=True)
    sess = _requests_session()
    fd, tmp = tempfile.mkstemp(
        prefix=".refdl_", suffix=".partial", dir=dest_dir
    )
    os.close(fd)
    try:
        with sess.get(
            url, headers=_http_headers(), stream=True, timeout=timeout
        ) as r:
            r.raise_for_status()
            with open(tmp, "wb") as out:
                for chunk in r.iter_content(chunk_size=1024 * 512):
                    if chunk:
                        out.write(chunk)
        os.replace(tmp, dest_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _download_ftp(url: str, dest_path: str, timeout: float, log: logging.Logger) -> None:
    dest_dir = os.path.dirname(os.path.abspath(dest_path)) or "."
    os.makedirs(dest_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".refdl_", suffix=".partial", dir=dest_dir
    )
    os.close(fd)
    try:
        # urllib handles ftp://; short timeout avoids hanging on blocked FTP
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            with open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out, length=64 * 1024)
        os.replace(tmp, dest_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def download_calibration_reference(
    urls: Iterable[str],
    dest_path: str,
    *,
    timeout_http: Tuple[float, float] = (45.0, 300.0),
    timeout_ftp: float = 60.0,
    log: Optional[logging.Logger] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Try each URL / strategy until the file is saved to ``dest_path``.

    Returns
    -------
    ok : bool
    last_error : str or None
        Brief message from the last failed attempt (for logging).
    """
    log = log or logging.getLogger(__name__)
    last_err: Optional[str] = None
    dest_path = os.path.abspath(dest_path)

    for url in urls:
        if not url:
            continue
        try:
            if url.startswith("ftp://"):
                _download_ftp(url, dest_path, timeout_ftp, log)
            else:
                _download_http_stream(url, dest_path, timeout_http, log)
            log.info("Fetched calibration reference from %s", url)
            return True, None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            log.debug("Reference download failed for %s — %s", url, last_err, exc_info=True)

    if astropy_download_file is not None:
        for url in urls:
            if not url or url.startswith("ftp://"):
                continue
            try:
                dat = astropy_download_file(
                    url,
                    cache=True,
                    show_progress=False,
                    timeout=int(timeout_http[0] + timeout_http[1]),
                )
                shutil.move(dat, dest_path)
                return True, None
            except Exception as e:
                last_err = f"astropy:{type(e).__name__}: {e}"
                log.debug(
                    "Astropy download_file failed for %s — %s",
                    url,
                    last_err,
                    exc_info=True,
                )

    return False, last_err


def fetch_calibration_reference(
    global_defaults: Dict[str, Any],
    ref_env: str,
    ref_file: str,
    dest_path: str,
    *,
    log: Optional[logging.Logger] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Copy from local ``$jref``/etc. if present, else download using
    :func:`build_calibration_reference_urls` and :func:`download_calibration_reference`.
    """
    log = log or logging.getLogger(__name__)
    ref_file = os.path.basename(ref_file.strip())
    local = _local_reference_path(ref_env, ref_file)
    if local:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(dest_path)) or ".", exist_ok=True)
            shutil.copy2(local, dest_path)
            log.info("Using local reference file %s -> %s", local, dest_path)
            return True, None
        except OSError as e:
            log.warning("Could not copy local reference %s: %s", local, e)

    urls = build_calibration_reference_urls(global_defaults, ref_env, ref_file)
    return download_calibration_reference(urls, dest_path, log=log)
