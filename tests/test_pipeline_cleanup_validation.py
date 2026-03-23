"""
Validate that key drizzle sidecars are covered by cleanup paths.

- ``{inst}.ref.drc.fits`` (e.g. ``acs_wfc_full.ref.drc.fits``): removed after a
  successful mask-reference drizzle by
  :func:`hst123.utils.workdir_cleanup.remove_superseded_instrument_mask_reference_drizzle`
  from :meth:`hst123.hst123.pick_reference` (unless ``--keep-drizzle-artifacts``).

- ``*.drc.noise.fits`` (sky copy for DOLPHOT): removed in DOLPHOT primitive
  cleanup via glob ``*drc.noise.fits`` in ``run_dolphot`` /
  ``get_dolphot_photometry`` (``respect_keep_artifacts=False``).

- :data:`hst123.settings.pipeline_products` lists common pipeline output globs
  (including ``*drc.fits`` and ``*drc.noise.fits``) for tooling or manual cleanup.

Note: CLI ``--cleanup`` removes ``hst.input_images`` and, under ``--work-dir``,
patterns in :data:`hst123.settings.cleanup_extra_globs` (e.g. ``*.drc.noise.fits``).
:func:`hst123.utils.workdir_cleanup.remove_files_matching_globs` resolves globs
under ``--work-dir``, not the shell CWD.
"""
from __future__ import annotations

import glob
from pathlib import Path

import pytest

from hst123 import settings


@pytest.mark.parametrize(
    "basename",
    [
        "acs_wfc_full.ref.drc.fits",
        "acs.f814w.stack.drc.fits",
    ],
)
def test_pipeline_products_drc_glob_matches_logical_drc_names(tmp_path, monkeypatch, basename):
    """``*drc.fits`` in pipeline_products matches instrument ref and stack .drc.fits."""
    monkeypatch.chdir(tmp_path)
    Path(basename).write_bytes(b"0")
    matched = []
    for pattern in settings.pipeline_products:
        if pattern == "*drc.fits":
            matched.extend(glob.glob(pattern))
    assert basename in matched


def test_pipeline_products_includes_drc_noise_pattern():
    assert "*drc.noise.fits" in settings.pipeline_products


def test_drc_noise_glob_matches_example_ut_filename(tmp_path, monkeypatch):
    """Ephemeral DOLPHOT noise sidecar name used in the field matches ``*drc.noise.fits``."""
    name = "acs.f814w.ut221110_0001.drc.noise.fits"
    monkeypatch.chdir(tmp_path)
    Path(name).write_bytes(b"0")
    assert name in glob.glob("*drc.noise.fits")
    noise_hits = []
    for pattern in settings.pipeline_products:
        if pattern == "*drc.noise.fits":
            noise_hits.extend(glob.glob(pattern))
    assert name in noise_hits
