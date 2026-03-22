"""Tests for workdir_cleanup (intermediate drizzle/tweakreg file handling)."""
import logging

from hst123.utils.workdir_cleanup import (
    cleanup_after_astrodrizzle,
    cleanup_after_tweakreg,
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


def test_cleanup_respects_keep_artifacts(tmp_path):
    wd = tmp_path / "w"
    wd.mkdir()
    (wd / "staticMask.fits").write_bytes(b"0")
    log = logging.getLogger("test_cleanup2")

    cleanup_after_astrodrizzle(str(wd), log=log, keep_artifacts=True)

    assert (wd / "staticMask.fits").exists()


def test_cleanup_after_tweakreg_archives_shifts(tmp_path):
    wd = tmp_path / "w"
    wd.mkdir()
    (wd / "drizzle_shifts.txt").write_text("# ref\n", encoding="utf-8")
    log = logging.getLogger("test_cleanup3")

    cleanup_after_tweakreg(str(wd), log=log, keep_artifacts=False)

    assert not (wd / "drizzle_shifts.txt").exists()
    assert list((wd / "logs").glob("drizzle_shifts_*.txt"))


def test_cleanup_after_tweakreg_archives_numbered_shift_files(tmp_path):
    """Per-filter TweakReg writes drizzle_shifts_0.txt, etc."""
    wd = tmp_path / "w"
    wd.mkdir()
    (wd / "drizzle_shifts_0.txt").write_text("a\n", encoding="utf-8")
    (wd / "drizzle_shifts_1.txt").write_text("b\n", encoding="utf-8")
    log = logging.getLogger("test_cleanup4")

    cleanup_after_tweakreg(str(wd), log=log, keep_artifacts=False)

    assert not (wd / "drizzle_shifts_0.txt").exists()
    assert not (wd / "drizzle_shifts_1.txt").exists()
    logs = wd / "logs"
    assert len(list(logs.glob("drizzle_shifts_0_*.txt"))) == 1
    assert len(list(logs.glob("drizzle_shifts_1_*.txt"))) == 1
