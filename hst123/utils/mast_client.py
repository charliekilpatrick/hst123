"""
Helpers for MAST / astroquery: extended timeouts, retries, and batched calls.

astroquery's default 600 s portal timeout is easy to exceed on large cone searches.
The portal raises built-in ``TimeoutError``, not ``astroquery.exceptions.TimeoutError``.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Iterator, TypeVar

import astroquery.exceptions
import requests
from astroquery.mast import Observations
from astroquery.mast import conf as mast_conf

T = TypeVar("T")

MAST_TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    astroquery.exceptions.RemoteServiceError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
    TimeoutError,
)
if hasattr(astroquery.exceptions, "TimeoutError"):
    MAST_TRANSIENT_ERRORS = MAST_TRANSIENT_ERRORS + (
        astroquery.exceptions.TimeoutError,
    )

DEFAULT_QUERY_TIMEOUT = 1800
DEFAULT_RETRIES = 3
DEFAULT_RETRY_PAUSE = 15.0
OBS_PRODUCT_BATCH = 75
DOWNLOAD_BATCH = 40


@contextmanager
def mast_extended_timeout(seconds: int = DEFAULT_QUERY_TIMEOUT):
    """Temporarily raise MAST portal HTTP timeout (conf and live connection)."""
    old_conf = mast_conf.timeout
    portal = Observations._portal_api_connection
    old_portal = portal.TIMEOUT
    try:
        mast_conf.timeout = seconds
        portal.TIMEOUT = seconds
        yield
    finally:
        mast_conf.timeout = old_conf
        portal.TIMEOUT = old_portal


def mast_call_with_retries(
    fn: Callable[..., T],
    *args,
    retries: int = DEFAULT_RETRIES,
    pause: float = DEFAULT_RETRY_PAUSE,
    **kwargs,
) -> T:
    """Call *fn* with exponential backoff on transient MAST/network errors."""
    last_exc: BaseException | None = None
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except MAST_TRANSIENT_ERRORS as exc:
            last_exc = exc
            if attempt + 1 >= retries:
                raise
            time.sleep(pause * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("mast_call_with_retries: no attempts made")


def iter_table_batches(table, batch_size: int) -> Iterator:
    """Yield successive slices of an astropy Table."""
    n = len(table)
    for start in range(0, n, batch_size):
        yield table[start : start + batch_size]
