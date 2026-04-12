"""Tests for workdir_cleanup (intermediate drizzle/tweakreg file handling)."""
import logging
import os

from hst123.utils.workdir_cleanup import (
    cleanup_after_astrodrizzle,
    cleanup_after_tweakreg,
    remove_files_matching_globs,
    remove_superseded_instrument_mask_reference_drizzle,
)


def test_cleanup_after_astrodrizzle_removes_prefixed_drizzle_masks(tmp_path):
    """DrizzlePac writes root-prefixed mask names, not only bare crmask.fits."""
    wd = tmp_path / "w"
    wd.mkdir()
    (wd / "foo_crmask.fits").write_bytes(b"0")
    (wd / "bar_dqmask.fits").write_bytes(b"0")
    (wd / "wfpc2.f555w.ref_800x800_1_staticMask.fits").write_bytes(b"0")
    log = logging.getLogger("test_cleanup_masks")

    cleanup_after_astrodrizzle(str(wd), log=log, keep_artifacts=False)

    assert not (wd / "foo_crmask.fits").exists()
    assert not (wd / "bar_dqmask.fits").exists()
    assert not (wd / "wfpc2.f555w.ref_800x800_1_staticMask.fits").exists()


def test_cleanup_after_astrodrizzle_removes_drizzle_subdir_interstitials(tmp_path):
    """``--drizzle-all`` writes outputs under work_dir/drizzle/; scrub there too."""
    wd = tmp_path / "w"
    wd.mkdir()
    drizzle = wd / "drizzle"
    drizzle.mkdir()
    (drizzle / "staticMask.fits").write_bytes(b"0")
    (drizzle / "drz_med.fits").write_bytes(b"0")
    (drizzle / "single_sci.fits").write_bytes(b"0")
    (drizzle / "wfpc2.f555w.ut260411_0001_1_staticMask.fits").write_bytes(b"0")
    log = logging.getLogger("test_cleanup_drizzle_subdir")

    cleanup_after_astrodrizzle(str(wd), log=log, keep_artifacts=False)

    assert not (drizzle / "staticMask.fits").exists()
    assert not (drizzle / "drz_med.fits").exists()
    assert not (drizzle / "single_sci.fits").exists()
    assert not (drizzle / "wfpc2.f555w.ut260411_0001_1_staticMask.fits").exists()


def test_cleanup_after_astrodrizzle_removes_hst123drz_scratch_glob(tmp_path):
    wd = tmp_path / "w"
    wd.mkdir()
    (wd / "u2465107t_hst123drz12345_c0m.fits").write_bytes(b"0")
    (wd / "u2465107t_hst123drz12345_c1m.fits").write_bytes(b"0")
    log = logging.getLogger("test_hst123drz")

    cleanup_after_astrodrizzle(str(wd), log=log, keep_artifacts=False)

    assert not (wd / "u2465107t_hst123drz12345_c0m.fits").exists()
    assert not (wd / "u2465107t_hst123drz12345_c1m.fits").exists()


def test_cleanup_after_astrodrizzle_base_work_dir_removes_interstitials_in_base(tmp_path):
    """When AstroDrizzle runs with outdir=workspace/, DrizzlePac may leave staticMask in base."""
    base = tmp_path / "job"
    base.mkdir()
    ws = base / "workspace"
    ws.mkdir()
    (base / "staticMask.fits").write_bytes(b"0")
    (ws / "single_sci.fits").write_bytes(b"0")
    log = logging.getLogger("test_cleanup_base_ws")

    cleanup_after_astrodrizzle(
        str(ws), log=log, keep_artifacts=False, base_work_dir=str(base)
    )

    assert not (base / "staticMask.fits").exists()
    assert not (ws / "single_sci.fits").exists()


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


def test_remove_files_matching_globs_uses_work_dir_not_cwd(tmp_path):
    """Products under ``work_dir`` are found even when CWD is elsewhere."""
    work = tmp_path / "test_data"
    work.mkdir()
    noise = work / "acs.f814w.ref_0001.drc.noise.fits"
    noise.write_bytes(b"0")
    other = tmp_path / "noise_elsewhere.drc.noise.fits"
    other.write_bytes(b"0")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    old = os.getcwd()
    try:
        os.chdir(cwd)
        n = remove_files_matching_globs(str(work), ("*drc.noise.fits",))
        assert n == 1
        assert not noise.exists()
        assert other.exists()
    finally:
        os.chdir(old)


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
