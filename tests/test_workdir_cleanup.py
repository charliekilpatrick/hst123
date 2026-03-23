"""Tests for workdir_cleanup (intermediate drizzle/tweakreg file handling)."""
import logging

from hst123.utils.workdir_cleanup import (
    cleanup_after_astrodrizzle,
    cleanup_after_tweakreg,
    remove_superseded_instrument_mask_reference_drizzle,
)


def test_cleanup_after_astrodrizzle_removes_static_mask_and_archives_log(tmp_path):
    wd = tmp_path / "w"
    wd.mkdir()
    (wd / "staticMask.fits").write_bytes(b"0")
    (wd / "astrodrizzle.log").write_text("log line\n", encoding="utf-8")
    log = logging.getLogger("test_cleanup")

    cleanup_after_astrodrizzle(str(wd), log=log, keep_artifacts=False)

    assert not (wd / "staticMask.fits").exists()
    assert not (wd / "astrodrizzle.log").exists()
    logs = wd / "logs"
    assert logs.is_dir()
    archived = list(logs.glob("astrodrizzle_*.log"))
    assert len(archived) == 1


def test_remove_superseded_instrument_mask_ref_drizzle(tmp_path):
    """``{inst}.ref.drc.fits`` plus internal .drz and sidecars are removed."""
    wd = tmp_path / "w"
    wd.mkdir()
    log = logging.getLogger("test_inst_ref")
    drc = wd / "acs_wfc_full.ref.drc.fits"
    drz = wd / "acs_wfc_full.ref.drz.fits"
    drc.write_bytes(b"0")
    drz.write_bytes(b"0")
    (wd / "acs_wfc_full.ref.drz_sci.fits").write_bytes(b"0")
    (wd / "acs_wfc_full.ref.drz_med.fits").write_bytes(b"0")

    remove_superseded_instrument_mask_reference_drizzle(
        str(drc), log=log, keep_artifacts=False
    )

    assert not drc.exists()
    assert not drz.exists()
    assert not (wd / "acs_wfc_full.ref.drz_sci.fits").exists()
    assert not (wd / "acs_wfc_full.ref.drz_med.fits").exists()


def test_remove_superseded_instrument_mask_ref_respects_keep(tmp_path):
    wd = tmp_path / "w"
    wd.mkdir()
    log = logging.getLogger("test_inst_ref2")
    drc = wd / "acs_wfc_full.ref.drc.fits"
    drc.write_bytes(b"0")
    remove_superseded_instrument_mask_reference_drizzle(
        str(drc), log=log, keep_artifacts=True
    )
    assert drc.exists()


def test_cleanup_respects_keep_artifacts(tmp_path):
    wd = tmp_path / "w"
    wd.mkdir()
    (wd / "staticMask.fits").write_bytes(b"0")
    log = logging.getLogger("test_cleanup2")

    cleanup_after_astrodrizzle(str(wd), log=log, keep_artifacts=True)

    assert (wd / "staticMask.fits").exists()


def test_cleanup_after_tweakreg_ingests_and_removes_shifts(tmp_path):
    wd = tmp_path / "w"
    wd.mkdir()
    (wd / "drizzle_shifts.txt").write_text("# ref\n", encoding="utf-8")
    log = logging.getLogger("test_cleanup3")

    cleanup_after_tweakreg(str(wd), log=log, keep_artifacts=False)

    assert not (wd / "drizzle_shifts.txt").exists()
    # Ingested into hst123.tweakreg log; no separate copies under logs/


def test_cleanup_after_tweakreg_removes_numbered_shift_files(tmp_path):
    """Per-filter TweakReg writes drizzle_shifts_0.txt, etc.; ingested then deleted."""
    wd = tmp_path / "w"
    wd.mkdir()
    (wd / "drizzle_shifts_0.txt").write_text("a\n", encoding="utf-8")
    (wd / "drizzle_shifts_1.txt").write_text("b\n", encoding="utf-8")
    log = logging.getLogger("test_cleanup4")

    cleanup_after_tweakreg(str(wd), log=log, keep_artifacts=False)

    assert not (wd / "drizzle_shifts_0.txt").exists()
    assert not (wd / "drizzle_shifts_1.txt").exists()
