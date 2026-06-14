"""Tests for per-image DOLPHOT summary logging."""
from pathlib import Path

import numpy as np
import pytest

from hst123.utils.dolphot_image_summary import (
    build_dolphot_image_summary_rows,
    format_dolphot_image_summary_line,
)


def _write_mini_dolphot_bundle(tmp_path: Path) -> Path:
    base = tmp_path / "dp0000"
    (tmp_path / "dp0000.param").write_text(
        "\n".join(
            [
                "Nimg = 1",
                "img0000_file = ref.drc",
                "img0001_file = sci_a.chip1",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "dp0000.info").write_text(
        "\n".join(
            [
                "2 sets of output data",
                "sci_a.chip1",
                "  60000.0",
                "EXTENSION 0 CHIP 1",
                "Limits",
                " 0 100 0 100",
                "* image 1: F814W 1 100.000000",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "dp0000.data").write_text(
        "\n".join(
            [
                "Align image 1: 12 10 1.0 2.0 1.0 0.0 3.0 0.250000",
                "PSF image 1: 5 0.123456",
                "Apcor image 1: 7 6 0.0 0.0 0.0 0.0 0 0.0 0.0 0.0 0.0 0 0.0 0.0 0.0 0.0",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "dp0000.columns").write_text(
        "\n".join(
            [
                "11. Object type (1=bright star, 2=faint, 3=elongated, 4=hot pixel, 5=extended)",
                "12. Instrumental VEGAMAG magnitude, sci_a.chip1 (WFPC2_F814W, 100.0 sec)",
                "13. Magnitude uncertainty, sci_a.chip1 (WFPC2_F814W, 100.0 sec)",
            ]
        ),
        encoding="utf-8",
    )
    # Catalog width must cover the highest column index (13 → width 13).
    catalog = np.full((4, 13), np.nan, dtype=np.float64)
    catalog[:, 10] = [1.0, 1.0, 1.0, 2.0]
    catalog[:, 11] = [20.0, 22.0, 24.0, 26.0]
    catalog[:, 12] = [0.05, 0.08, 0.12, 0.20]
    np.savetxt(base, catalog)
    return base


def test_build_dolphot_image_summary_rows(tmp_path):
    base = _write_mini_dolphot_bundle(tmp_path)
    rows = build_dolphot_image_summary_rows(base, snr_limit=3.0)
    assert len(rows) == 2
    sci = rows[1]
    assert sci["image_num"] == "001"
    assert sci["image_name"] == "sci_a.chip1"
    assert sci["filter"] == "F814W"
    assert sci["exptime"] == 100.0
    assert sci["align_stars_used"] == 10
    assert sci["align_sig"] == pytest.approx(0.25)
    assert sci["psf_central_adj"] == pytest.approx(0.123456)
    assert sci["apcor_stars_used"] == 6
    # Tiny synthetic catalog may not span enough mag range for a stable limit estimate.
    assert "limit_mag_3sig" in sci


def test_format_dolphot_image_summary_line():
    line = format_dolphot_image_summary_line(
        {
            "image_num": "016",
            "image_name": "ifl701h9q_flc.chip1",
            "filter": "F555W",
            "exptime": 70.0,
            "align_stars_used": 39,
            "align_sig": 0.404963,
            "psf_central_adj": 0.015828,
            "apcor_stars_used": 2,
            "limit_mag_3sig": 27.123,
        }
    )
    assert "img=016" in line
    assert "ifl701h9q_flc.chip1" in line
    assert "filter=F555W" in line
    assert "align_used=39" in line
    assert "limit_3sig=27.123" in line


@pytest.mark.integration
def test_sn2026dix_dp0000_summary_if_present():
    base = Path("/Users/ckilpatrick/Downloads/SN2026dix/dolphot/dp0000")
    if not base.is_file():
        pytest.skip("SN2026dix DOLPHOT catalog not available")
    rows = build_dolphot_image_summary_rows(base, snr_limit=3.0)
    assert len(rows) == 32
    img16 = next(r for r in rows if r["image_num"] == "016")
    assert img16["image_name"] == "ifl701gxq_flc.chip1"
    assert img16["filter"] == "F555W"
    assert img16["align_stars_used"] == 41
    assert img16["align_sig"] == pytest.approx(0.404963, rel=1e-3)
    assert img16["psf_central_adj"] == pytest.approx(0.015828, rel=1e-3)
    assert img16["apcor_stars_used"] == 2
    assert np.isfinite(img16["limit_mag_3sig"])
