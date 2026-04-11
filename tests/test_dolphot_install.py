"""Tests for DOLPHOT download and install (hst123.dolphot_install)."""
import io
import tarfile
from pathlib import Path

import pytest

import hst123.dolphot_install as dolphot_install
from hst123.dolphot_install import (
    ACS_WFC_PAM_FILENAMES,
    _calcsky_make_target,
    apply_calcsky_source_patches,
    apply_dolphot_source_patches,
    CONDA_DOLPHOT_RELATIVE,
    DOLPHOT_BASE_URL,
    DOLPHOT_EXECUTABLES,
    DOLPHOT_STAMP_DIR,
    DOLPHOT_VERSION,
    ONE_PSF_PER_INSTRUMENT,
    PSF_FILES,
    SOURCES_BASE,
    SOURCES_MODULES,
    SOURCES_MODULES_HST,
    configure_dolphot_makefile,
    dolphot_acs_data_dir,
    dolphot_make_root,
    dolphot_path_for_shell,
    _url_for,
    default_dolphot_install_dir,
    download_file,
    extract_tar,
    get_conda_prefix,
    install_psfs,
    install_sources,
    link_executables_to_conda_bin,
    relocate_acs_psf_into_canonical_layout,
    relocate_acs_wfc_pam_into_canonical_layout,
    relocate_all_legacy_psf_into_canonical_layout,
    relocate_wfc3_psf_into_canonical_layout,
    relocate_wfpc2_psf_into_canonical_layout,
    psf_archive_payload_present,
    psf_install_recorded,
    sources_install_is_complete,
    verify_acs_wfc_pam_files,
    verify_wfc3_mask_support_files,
    WFC3_MASK_MAP_FILENAMES,
    write_psf_stamp,
    write_sources_stamp,
)


def test_verify_wfc3_mask_support_files_ok(tmp_path, monkeypatch):
    base = tmp_path / "opt" / "hst123-dolphot"
    base.mkdir(parents=True)
    (base / "Makefile").write_text("all:\n", encoding="utf-8")
    wfc3d = base / "wfc3" / "data"
    wfc3d.mkdir(parents=True)
    for name in WFC3_MASK_MAP_FILENAMES:
        (wfc3d / name).write_bytes(b"x" * 600)
    monkeypatch.setattr(
        dolphot_install,
        "_candidate_dolphot_source_roots",
        lambda: [base.resolve()],
    )
    ok, msgs = verify_wfc3_mask_support_files()
    assert ok
    assert msgs == []


def test_verify_wfc3_mask_support_files_missing_map(tmp_path, monkeypatch):
    base = tmp_path / "opt" / "hst123-dolphot"
    base.mkdir(parents=True)
    (base / "Makefile").write_text("all:\n", encoding="utf-8")
    wfc3d = base / "wfc3" / "data"
    wfc3d.mkdir(parents=True)
    (wfc3d / "UVIS1wfc3_map.fits").write_bytes(b"x" * 600)
    monkeypatch.setattr(
        dolphot_install,
        "_candidate_dolphot_source_roots",
        lambda: [base.resolve()],
    )
    ok, msgs = verify_wfc3_mask_support_files()
    assert not ok
    assert any("UVIS2wfc3_map" in m or "ir_wfc3_map" in m for m in msgs)


def test_url_for():
    """URLs are built correctly from base and filename."""
    assert _url_for("dolphot3.1.tar.gz") == DOLPHOT_BASE_URL + "dolphot3.1.tar.gz"
    assert _url_for("ACS_WFC_F435W.tar.gz") == DOLPHOT_BASE_URL + "ACS_WFC_F435W.tar.gz"


def test_sources_constants():
    """Base and module tarball names match DOLPHOT_VERSION."""
    assert SOURCES_BASE == "dolphot3.1.tar.gz"
    assert "dolphot3.1.ACS.tar.gz" in SOURCES_MODULES
    assert "dolphot3.1.WFC3.tar.gz" in SOURCES_MODULES
    assert "dolphot3.1.WFPC2.tar.gz" in SOURCES_MODULES
    assert "dolphot3.1.NIRCAM.tar.gz" in SOURCES_MODULES
    assert set(SOURCES_MODULES_HST) <= set(SOURCES_MODULES)
    assert len(SOURCES_MODULES) == len(SOURCES_MODULES_HST) + 5


def test_dolphot_make_root_nested(tmp_path):
    """Prefer dolphot3.1/Makefile when install root has no Makefile."""
    root = tmp_path / "opt"
    nested = root / "dolphot3.1"
    nested.mkdir(parents=True)
    (nested / "Makefile").write_text("all:\n")
    assert dolphot_make_root(root) == nested


def test_configure_dolphot_makefile_uncomments(tmp_path):
    """Makefile export lines are activated like a manual uncomment."""
    mf = tmp_path / "Makefile"
    mf.write_text(
        "#export THREADED=1\n"
        "#export THREAD_CFLAGS= -DDOLPHOT_THREADED -D_REENTRANT -fopenmp\n"
        "#export THREAD_LIBS= -lomp\n"
        "#export USEACS=1\n"
        "#export USEROMAN=1\n",
        encoding="utf-8",
    )
    configure_dolphot_makefile(
        tmp_path, threaded=True, enable_extended_modules=True
    )
    t = mf.read_text(encoding="utf-8")
    assert "export THREADED=1" in t
    assert "export USEACS=1" in t
    assert "export USEROMAN=1" in t
    assert "#export THREADED=1" not in t


def test_apply_dolphot_source_patches_main_buffer(tmp_path):
    """Installer enlarges main() sprintf buffer so long absolute paths do not SIGTRAP."""
    dc = tmp_path / "dolphot.c"
    dc.write_text(
        "int main(int argc,char**argv) {\n   char str[82];\n   return 0;\n}\n",
        encoding="utf-8",
    )
    assert apply_dolphot_source_patches(tmp_path) is True
    t = dc.read_text(encoding="utf-8")
    assert "char str[4096]" in t
    assert "hst123_dolphot_main_str_buf" in t
    assert "char str[82]" not in t
    assert apply_dolphot_source_patches(tmp_path) is False


def test_apply_dolphot_source_patches_idempotent_when_already_4096(tmp_path):
    dc = tmp_path / "dolphot.c"
    dc.write_text(
        "int main(int argc,char**argv) {\n"
        "   /* hst123_dolphot_main_str_buf: x */\n"
        "   char str[4096];\n}\n",
        encoding="utf-8",
    )
    assert apply_dolphot_source_patches(tmp_path) is False


def test_calcsky_make_target_prefers_bin_prefix(tmp_path):
    mf = tmp_path / "Makefile"
    mf.write_text("all:\n\nbin/calcsky: calcsky.c\n\tgcc\n", encoding="utf-8")
    assert _calcsky_make_target(tmp_path) == "bin/calcsky"


def test_calcsky_make_target_fallback(tmp_path):
    mf = tmp_path / "Makefile"
    mf.write_text("calcsky:\n\tgcc\n", encoding="utf-8")
    assert _calcsky_make_target(tmp_path) == "calcsky"


def test_apply_calcsky_source_patches_main_buffer(tmp_path):
    """calcsky.c uses a tiny sprintf buffer; long argv[1] overflows → SIGTRAP on macOS."""
    cc = tmp_path / "calcsky.c"
    cc.write_text(
        "int main(int argc,char**argv) {\n   char str[81];\n   return 0;\n}\n",
        encoding="utf-8",
    )
    assert apply_calcsky_source_patches(tmp_path) is True
    t = cc.read_text(encoding="utf-8")
    assert "char str[4096]" in t
    assert "hst123_calcsky_main_str_buf" in t
    assert "char str[81]" not in t
    assert apply_calcsky_source_patches(tmp_path) is False


def test_configure_dolphot_makefile_hst_only_skips_extended(tmp_path):
    mf = tmp_path / "Makefile"
    mf.write_text(
        "#export USEACS=1\n"
        "#export USEROMAN=1\n",
        encoding="utf-8",
    )
    configure_dolphot_makefile(
        tmp_path, threaded=False, enable_extended_modules=False
    )
    t = mf.read_text(encoding="utf-8")
    assert "export USEACS=1" in t
    assert "#export USEROMAN=1" in t


def test_one_psf_per_instrument():
    """One PSF filename per instrument is defined."""
    assert ONE_PSF_PER_INSTRUMENT["ACS"] == "ACS_WFC_F435W.tar.gz"
    assert ONE_PSF_PER_INSTRUMENT["WFC3"] == "WFC3_UVIS_F555W.tar.gz"
    assert ONE_PSF_PER_INSTRUMENT["WFPC2"] == "WFPC2_F555W.tar.gz"


def test_one_per_instrument_acs_plan_includes_pam_tarball():
    """ACS one-per-instrument install must fetch PAM + one PSF (acsmask needs PAMs)."""
    instruments = ["ACS"]
    files_to_get = []
    for inv in instruments:
        if inv == "ACS":
            files_to_get.append("ACS_WFC_PAM.tar.gz")
        files_to_get.append(ONE_PSF_PER_INSTRUMENT[inv])
    assert files_to_get == ["ACS_WFC_PAM.tar.gz", "ACS_WFC_F435W.tar.gz"]


def test_psf_files_lists():
    """PSF_FILES has non-empty lists for ACS, WFC3, WFPC2."""
    for inv in ("ACS", "WFC3", "WFPC2"):
        assert inv in PSF_FILES
        assert len(PSF_FILES[inv]) > 0
        assert ONE_PSF_PER_INSTRUMENT[inv] in PSF_FILES[inv]


def test_extract_tar_strip_top_level(tmp_path):
    """extract_tar strips a single top-level directory when requested."""
    tar_path = tmp_path / "test.tar.gz"
    dest = tmp_path / "dest"
    dest.mkdir()
    with tarfile.open(tar_path, "w:gz") as tar:
        # Simulate versioned top dir + Makefile layout
        import io
        buf = io.BytesIO(b"makefile content")
        info = tarfile.TarInfo(name="dolphot3.1/Makefile")
        info.size = len(buf.getvalue())
        tar.addfile(info, buf)
    extract_tar(tar_path, dest, strip_top_level=True)
    assert (dest / "Makefile").read_text() == "makefile content"


def test_extract_tar_no_strip(tmp_path):
    """extract_tar preserves paths when not stripping."""
    tar_path = tmp_path / "test.tar.gz"
    dest = tmp_path / "dest"
    dest.mkdir()
    with tarfile.open(tar_path, "w:gz") as tar:
        import io
        buf = io.BytesIO(b"content")
        info = tarfile.TarInfo(name="top/file.txt")
        info.size = len(buf.getvalue())
        tar.addfile(info, buf)
    extract_tar(tar_path, dest, strip_top_level=False)
    assert (dest / "top" / "file.txt").read_text() == "content"


def test_install_psfs_invalid_instrument(tmp_path):
    """install_psfs raises for unknown instrument."""
    with pytest.raises(ValueError, match="Unknown instrument"):
        install_psfs(tmp_path, instruments=["INVALID"], one_per_instrument=True)


def test_sources_install_is_complete_requires_makefile_and_stamp(tmp_path):
    """sources_install_is_complete needs Makefile + valid stamp JSON."""
    d = tmp_path / "dolphot"
    d.mkdir()
    assert not sources_install_is_complete(d)
    (d / "Makefile").write_text("all:\n")
    assert not sources_install_is_complete(d)
    write_sources_stamp(d)
    assert sources_install_is_complete(d)


def test_sources_install_is_complete_rejects_bad_stamp(tmp_path):
    """Corrupt or mismatched stamp is ignored."""
    d = tmp_path / "dolphot"
    d.mkdir()
    (d / "Makefile").write_text("all:\n")
    stamp_dir = d / DOLPHOT_STAMP_DIR
    stamp_dir.mkdir(parents=True)
    (stamp_dir / "sources.json").write_text('{"dolphot_version": "0.0"}', encoding="utf-8")
    assert not sources_install_is_complete(d)


def test_install_sources_skips_when_stamp_matches(tmp_path, monkeypatch):
    """Second run does not call download_file when stamp + Makefile exist."""
    d = tmp_path / "dolphot"
    d.mkdir()
    (d / "Makefile").write_text("all:\n")
    write_sources_stamp(d)

    def boom(*_a, **_k):
        raise AssertionError("download_file should not be called")

    monkeypatch.setattr(dolphot_install, "download_file", boom)
    out = install_sources(d, force_download=False)
    assert out == d.resolve()


def test_install_sources_force_download_invokes_download(tmp_path, monkeypatch):
    """force_download=True fetches archives even when stamp is valid."""

    def _tiny_tar(path: Path, prefix: str) -> None:
        with tarfile.open(path, "w:gz") as tar:
            s = b"x"
            ti = tarfile.TarInfo(name=f"{prefix}/stub.txt")
            ti.size = len(s)
            tar.addfile(ti, io.BytesIO(s))

    d = tmp_path / "dolphot"
    d.mkdir()
    (d / "Makefile").write_text("all:\n")
    write_sources_stamp(d)

    downloads = []

    def fake_download(_url, dest_path, timeout=60, step_label=None):
        downloads.append(Path(dest_path).name)
        _tiny_tar(Path(dest_path), "pkg")

    monkeypatch.setattr(dolphot_install, "download_file", fake_download)
    install_sources(d, force_download=True)
    assert SOURCES_BASE in downloads
    assert set(SOURCES_MODULES) <= set(downloads)
    assert len(downloads) == 1 + len(SOURCES_MODULES)


def test_psf_archive_payload_present_acs_under_dolphot20_acs_data(tmp_path):
    """Detect ACS/WFC PSFs in legacy dolphot2.0/acs/data/ layout (no stamp)."""
    d = tmp_path / "dolphot"
    p = d / "dolphot2.0" / "acs" / "data"
    p.mkdir(parents=True)
    psf = p / "F435W.std.psf"
    psf.write_bytes(b"x" * 100)
    psf_paths, pam_paths = dolphot_install._scan_psf_disk_index(d)
    assert psf_archive_payload_present(
        "ACS_WFC_F435W.tar.gz", psf_paths=psf_paths, pam_paths=pam_paths
    )


def test_psf_archive_payload_present_wfc3_ir_dolphot20_wfc3_data(tmp_path):
    """DOLPHOT 2.x WFC3 IR layout: dolphot2.0/wfc3/data/F105W.ir.psf"""
    d = tmp_path / "dolphot"
    p = d / "dolphot2.0" / "wfc3" / "data"
    p.mkdir(parents=True)
    psf = p / "F105W.ir.psf"
    psf.write_bytes(b"x" * 100)
    psf_paths, pam_paths = dolphot_install._scan_psf_disk_index(d)
    assert psf_archive_payload_present(
        "WFC3_IR_F105W.tar.gz", psf_paths=psf_paths, pam_paths=pam_paths
    )


def test_psf_archive_payload_present_wfc3_uvis_under_wfc3_data(tmp_path):
    """UVIS-only filter under legacy wfc3/data (no uvis/ segment)."""
    d = tmp_path / "dolphot"
    p = d / "wfc3" / "data"
    p.mkdir(parents=True)
    psf = p / "F555W.std.psf"
    psf.write_bytes(b"x" * 100)
    psf_paths, pam_paths = dolphot_install._scan_psf_disk_index(d)
    assert psf_archive_payload_present(
        "WFC3_UVIS_F555W.tar.gz", psf_paths=psf_paths, pam_paths=pam_paths
    )
    assert not psf_archive_payload_present(
        "WFC3_IR_F105W.tar.gz", psf_paths=psf_paths, pam_paths=pam_paths
    )


def test_psf_archive_payload_present_wfpc2_dolphot20_wfpc2_data(tmp_path):
    """DOLPHOT 2.x WFPC2 layout: dolphot2.0/wfpc2/data/*.psf"""
    d = tmp_path / "dolphot"
    p = d / "dolphot2.0" / "wfpc2" / "data"
    p.mkdir(parents=True)
    psf = p / "F555W.std.psf"
    psf.write_bytes(b"x" * 100)
    psf_paths, pam_paths = dolphot_install._scan_psf_disk_index(d)
    assert psf_archive_payload_present(
        "WFPC2_F555W.tar.gz", psf_paths=psf_paths, pam_paths=pam_paths
    )


def test_psf_archive_payload_present_acs_wfc_pam_requires_both_under_make_root(tmp_path):
    """ACS_WFC_PAM is satisfied only when wfc1/wfc2 PAMs sit under make_root/acs/data."""
    root = tmp_path / "hst123-dolphot"
    data = root / "dolphot3.1" / "acs" / "data"
    data.mkdir(parents=True)
    (root / "dolphot3.1" / "Makefile").write_text("all:\n", encoding="utf-8")
    make_root = dolphot_make_root(root)
    psf_paths, pam_paths = dolphot_install._scan_psf_disk_index(make_root)
    assert not psf_archive_payload_present(
        "ACS_WFC_PAM.tar.gz",
        psf_paths=psf_paths,
        pam_paths=pam_paths,
        make_root=make_root,
    )
    for name in ACS_WFC_PAM_FILENAMES:
        (data / name).write_bytes(b"0" * 2048)
    psf_paths, pam_paths = dolphot_install._scan_psf_disk_index(make_root)
    assert psf_archive_payload_present(
        "ACS_WFC_PAM.tar.gz",
        psf_paths=psf_paths,
        pam_paths=pam_paths,
        make_root=make_root,
    )


def test_psf_archive_payload_present_false_when_only_wrong_instrument(tmp_path):
    """WFPC2 tree must not satisfy ACS_WFC archive."""
    d = tmp_path / "dolphot"
    p = d / "WFPC2"
    p.mkdir(parents=True)
    psf = p / "F435W.psf"
    psf.write_bytes(b"x" * 10)
    psf_paths, pam_paths = dolphot_install._scan_psf_disk_index(d)
    assert not psf_archive_payload_present(
        "ACS_WFC_F435W.tar.gz", psf_paths=psf_paths, pam_paths=pam_paths
    )


def test_install_psfs_skips_download_when_psf_stamp_exists(tmp_path, monkeypatch):
    """PSF archives with a stamp file are not re-downloaded."""
    src = tmp_path / "dolphot"
    src.mkdir()
    fn = ONE_PSF_PER_INSTRUMENT["ACS"]
    write_psf_stamp(src, fn)
    write_psf_stamp(src, "ACS_WFC_PAM.tar.gz")
    pam_dir = src / "acs" / "data"
    pam_dir.mkdir(parents=True)
    for name in ACS_WFC_PAM_FILENAMES:
        (pam_dir / name).write_bytes(b"0" * 2048)

    def boom(*_a, **_k):
        raise AssertionError("download_file should not be called")

    monkeypatch.setattr(dolphot_install, "download_file", boom)
    install_psfs(
        src,
        instruments=["ACS"],
        one_per_instrument=True,
        force_download=False,
    )
    assert psf_install_recorded(src, fn)


def test_install_psfs_skips_when_matching_psf_on_disk_no_stamp(tmp_path, monkeypatch):
    """Existing ACS PSF files satisfy the archive without an .installed stamp."""
    src = tmp_path / "dolphot"
    p = src / "dolphot2.0" / "acs" / "data"
    p.mkdir(parents=True)
    (p / "F435W.on_disk.psf").write_bytes(b"x" * 50)
    for name in ACS_WFC_PAM_FILENAMES:
        (p / name).write_bytes(b"0" * 2048)

    def boom(*_a, **_k):
        raise AssertionError("download_file should not be called")

    monkeypatch.setattr(dolphot_install, "download_file", boom)
    install_psfs(
        src,
        instruments=["ACS"],
        one_per_instrument=True,
        force_download=False,
    )
    assert psf_install_recorded(src, ONE_PSF_PER_INSTRUMENT["ACS"])
    assert psf_install_recorded(src, "ACS_WFC_PAM.tar.gz")


def test_get_conda_prefix_none(monkeypatch):
    """get_conda_prefix returns None when CONDA_PREFIX is unset."""
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    assert get_conda_prefix() is None


def test_get_conda_prefix_set(monkeypatch, tmp_path):
    """get_conda_prefix resolves CONDA_PREFIX when set."""
    fake = tmp_path / "env"
    fake.mkdir()
    monkeypatch.setenv("CONDA_PREFIX", str(fake))
    assert get_conda_prefix() == fake.resolve()


def test_default_dolphot_install_dir_conda(monkeypatch, tmp_path):
    """With CONDA_PREFIX, default install dir is under opt/hst123-dolphot."""
    fake = tmp_path / "myenv"
    fake.mkdir()
    monkeypatch.setenv("CONDA_PREFIX", str(fake))
    monkeypatch.chdir(tmp_path)
    expected = fake.resolve() / CONDA_DOLPHOT_RELATIVE
    assert default_dolphot_install_dir() == expected


def test_default_dolphot_install_dir_no_conda(monkeypatch, tmp_path):
    """Without CONDA_PREFIX, default is ./dolphot under cwd."""
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.chdir(tmp_path)
    assert default_dolphot_install_dir() == (tmp_path / "dolphot").resolve()


def test_link_executables_to_conda_bin(tmp_path):
    """Built executables are symlinked into conda_prefix/bin."""
    src_dir = tmp_path / "dolphot_build"
    src_dir.mkdir()
    for name in DOLPHOT_EXECUTABLES:
        p = src_dir / name
        p.write_text("#!/bin/sh\necho ok\n")
        p.chmod(0o755)
    conda = tmp_path / "conda_env"
    (conda / "bin").mkdir(parents=True)
    linked = link_executables_to_conda_bin(src_dir, conda_prefix=conda)
    assert set(linked) == set(DOLPHOT_EXECUTABLES)
    for name in DOLPHOT_EXECUTABLES:
        dst = conda / "bin" / name
        assert dst.is_symlink() or dst.is_file()
        assert dst.exists()


def test_link_executables_to_conda_bin_from_dolphot3_bin(tmp_path):
    """DOLPHOT 3.x layout: programs live under build root ``bin/``."""
    src_dir = tmp_path / "dolphot3.1"
    bindir = src_dir / "bin"
    bindir.mkdir(parents=True)
    for name in DOLPHOT_EXECUTABLES:
        p = bindir / name
        p.write_text("#!/bin/sh\necho ok\n")
        p.chmod(0o755)
    conda = tmp_path / "conda_env"
    (conda / "bin").mkdir(parents=True)
    linked = link_executables_to_conda_bin(src_dir, conda_prefix=conda)
    assert set(linked) == set(DOLPHOT_EXECUTABLES)
    for name in DOLPHOT_EXECUTABLES:
        dst = conda / "bin" / name
        assert dst.is_symlink() or dst.is_file()
        assert dst.resolve() == (bindir / name).resolve()


def test_dolphot_path_for_shell_prefers_bin(tmp_path):
    """PATH hint uses ``bin/`` when ``dolphot`` is there (DOLPHOT 3.x)."""
    root = tmp_path / "dolphot3.1"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "dolphot").write_text("#!/bin/sh\n", encoding="utf-8")
    assert dolphot_path_for_shell(root) == (root / "bin").resolve()


def test_dolphot_path_for_shell_flat_layout(tmp_path):
    """PATH hint uses build root when binaries are not under ``bin/``."""
    root = tmp_path / "dolphot_old"
    root.mkdir()
    (root / "dolphot").write_text("#!/bin/sh\n", encoding="utf-8")
    assert dolphot_path_for_shell(root) == root.resolve()


def test_link_executables_skip_without_conda_prefix(monkeypatch, tmp_path):
    """link_executables_to_conda_bin returns [] when no conda prefix."""
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    assert link_executables_to_conda_bin(src_dir, conda_prefix=None) == []


def test_dolphot_acs_data_dir_under_nested_dolphot31(tmp_path):
    """acs/data is found under dolphot_make_root (nested dolphot3.1)."""
    root = tmp_path / "hst123-dolphot"
    data = root / "dolphot3.1" / "acs" / "data"
    data.mkdir(parents=True)
    (root / "dolphot3.1" / "Makefile").write_text("all:\n", encoding="utf-8")
    assert dolphot_acs_data_dir(root) == data.resolve()


def test_dolphot_acs_data_dir_prefers_tree_with_pam_files(tmp_path):
    """Empty acs/data must not hide wfc*_pam.fits under dolphot2.0/acs/data."""
    root = tmp_path / "hst123-dolphot"
    (root / "dolphot3.1" / "acs" / "data").mkdir(parents=True)
    legacy = root / "dolphot3.1" / "dolphot2.0" / "acs" / "data"
    legacy.mkdir(parents=True)
    (root / "dolphot3.1" / "Makefile").write_text("all:\n", encoding="utf-8")
    for name in ACS_WFC_PAM_FILENAMES:
        (legacy / name).write_bytes(b"0" * 2048)
    assert dolphot_acs_data_dir(root) == legacy.resolve()


def test_relocate_acs_wfc_pam_into_canonical_layout(tmp_path):
    """ACS_WFC_PAM tarball layout (dolphot2.0/acs/data) -> make_root/acs/data."""
    root = tmp_path / "hst123-dolphot"
    mr = root / "dolphot3.1"
    leg = mr / "dolphot2.0" / "acs" / "data"
    leg.mkdir(parents=True)
    (mr / "Makefile").write_text("all:\n", encoding="utf-8")
    for name in ACS_WFC_PAM_FILENAMES:
        (leg / name).write_bytes(b"0" * 2048)
    make_root = dolphot_make_root(root)
    assert relocate_acs_wfc_pam_into_canonical_layout(make_root) is True
    for name in ACS_WFC_PAM_FILENAMES:
        assert (make_root / "acs" / "data" / name).is_file()


def test_relocate_acs_psf_into_canonical_layout(tmp_path):
    """Legacy ACS PSFs under dolphot3.1/dolphot2.0/acs/data are copied to acs/data."""
    root = tmp_path / "hst123-dolphot"
    mr = root / "dolphot3.1"
    leg = mr / "dolphot2.0" / "acs" / "data"
    leg.mkdir(parents=True)
    (mr / "Makefile").write_text("all:\n", encoding="utf-8")
    (leg / "F814W.wfc1.psf").write_bytes(b"x" * 2048)
    (leg / "F814W.wfc2.psf").write_bytes(b"x" * 2048)
    make_root = dolphot_make_root(root)
    copied = relocate_acs_psf_into_canonical_layout(make_root)
    assert copied == 2
    assert (make_root / "acs" / "data" / "F814W.wfc1.psf").is_file()
    assert (make_root / "acs" / "data" / "F814W.wfc2.psf").is_file()


def test_relocate_acs_psf_from_sibling_dolphot20_layout(tmp_path):
    """PSFs under opt/hst123-dolphot/dolphot2.0/acs/data (sibling of dolphot3.1)."""
    root = tmp_path / "hst123-dolphot"
    mr = root / "dolphot3.1"
    mr.mkdir(parents=True)
    (mr / "Makefile").write_text("all:\n", encoding="utf-8")
    sib = root / "dolphot2.0" / "acs" / "data"
    sib.mkdir(parents=True)
    (sib / "F814W.wfc1.psf").write_bytes(b"x" * 2048)
    (sib / "F814W.wfc2.psf").write_bytes(b"x" * 2048)
    make_root = dolphot_make_root(root)
    copied = relocate_acs_psf_into_canonical_layout(make_root)
    assert copied == 2
    assert (make_root / "acs" / "data" / "F814W.wfc1.psf").is_file()
    assert (make_root / "acs" / "data" / "F814W.wfc2.psf").is_file()


def test_relocate_wfc3_psf_into_canonical_layout_nested(tmp_path):
    """WFC3 PSFs under dolphot3.1/dolphot2.0/wfc3/IR -> wfc3/IR."""
    root = tmp_path / "hst123-dolphot"
    mr = root / "dolphot3.1"
    leg = mr / "dolphot2.0" / "wfc3" / "IR"
    leg.mkdir(parents=True)
    (mr / "Makefile").write_text("all:\n", encoding="utf-8")
    (leg / "F105W.ir.psf").write_bytes(b"x" * 2048)
    make_root = dolphot_make_root(root)
    assert relocate_wfc3_psf_into_canonical_layout(make_root) == 1
    assert (make_root / "wfc3" / "IR" / "F105W.ir.psf").is_file()


def test_relocate_wfc3_psf_from_sibling_dolphot20_layout(tmp_path):
    """WFC3 PSFs under sibling dolphot2.0/wfc3/data (next to dolphot3.1)."""
    root = tmp_path / "hst123-dolphot"
    mr = root / "dolphot3.1"
    mr.mkdir(parents=True)
    (mr / "Makefile").write_text("all:\n", encoding="utf-8")
    sib = root / "dolphot2.0" / "wfc3" / "data"
    sib.mkdir(parents=True)
    (sib / "F275W.uvis.psf").write_bytes(b"y" * 2048)
    make_root = dolphot_make_root(root)
    assert relocate_wfc3_psf_into_canonical_layout(make_root) == 1
    assert (make_root / "wfc3" / "data" / "F275W.uvis.psf").is_file()


def test_relocate_wfpc2_psf_into_canonical_layout_nested(tmp_path):
    """WFPC2 PSFs under dolphot3.1/dolphot2.0/wfpc2/data -> wfpc2/data."""
    root = tmp_path / "hst123-dolphot"
    mr = root / "dolphot3.1"
    leg = mr / "dolphot2.0" / "wfpc2" / "data"
    leg.mkdir(parents=True)
    (mr / "Makefile").write_text("all:\n", encoding="utf-8")
    (leg / "F555W_WF2.psf").write_bytes(b"z" * 2048)
    make_root = dolphot_make_root(root)
    assert relocate_wfpc2_psf_into_canonical_layout(make_root) == 1
    assert (make_root / "wfpc2" / "data" / "F555W_WF2.psf").is_file()


def test_relocate_wfpc2_psf_from_sibling_dolphot20_layout(tmp_path):
    """WFPC2 PSFs under sibling dolphot2.0/wfpc2/data."""
    root = tmp_path / "hst123-dolphot"
    mr = root / "dolphot3.1"
    mr.mkdir(parents=True)
    (mr / "Makefile").write_text("all:\n", encoding="utf-8")
    sib = root / "dolphot2.0" / "wfpc2" / "data"
    sib.mkdir(parents=True)
    (sib / "F814W_WF3.psf").write_bytes(b"w" * 2048)
    make_root = dolphot_make_root(root)
    assert relocate_wfpc2_psf_into_canonical_layout(make_root) == 1
    assert (make_root / "wfpc2" / "data" / "F814W_WF3.psf").is_file()


def test_relocate_all_legacy_psf_into_canonical_layout(tmp_path):
    """Single merge pass copies ACS, WFC3, and WFPC2 from nested + sibling legacy trees."""
    root = tmp_path / "hst123-dolphot"
    mr = root / "dolphot3.1"
    mr.mkdir(parents=True)
    (mr / "Makefile").write_text("all:\n", encoding="utf-8")
    nested = mr / "dolphot2.0"
    (nested / "acs" / "data").mkdir(parents=True)
    (nested / "acs" / "data" / "F606W.wfc1.psf").write_bytes(b"a" * 2048)
    (nested / "wfc3" / "UVIS").mkdir(parents=True)
    (nested / "wfc3" / "UVIS" / "F336W.uvis.psf").write_bytes(b"b" * 2048)
    sib = root / "dolphot2.0"
    (sib / "wfpc2" / "data").mkdir(parents=True)
    (sib / "wfpc2" / "data" / "F300W_WF4.psf").write_bytes(b"c" * 2048)
    make_root = dolphot_make_root(root)
    counts = relocate_all_legacy_psf_into_canonical_layout(make_root)
    assert counts == {"acs": 1, "wfc3": 1, "wfpc2": 1}
    assert (make_root / "acs" / "data" / "F606W.wfc1.psf").is_file()
    assert (make_root / "wfc3" / "UVIS" / "F336W.uvis.psf").is_file()
    assert (make_root / "wfpc2" / "data" / "F300W_WF4.psf").is_file()


def test_verify_acs_wfc_pam_files_ok(monkeypatch, tmp_path):
    root = tmp_path / "hst123-dolphot"
    data = root / "dolphot3.1" / "acs" / "data"
    data.mkdir(parents=True)
    (root / "dolphot3.1" / "Makefile").write_text("all:\n", encoding="utf-8")
    for name in ("wfc1_pam.fits", "wfc2_pam.fits"):
        (data / name).write_bytes(b"0" * 2048)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dolphot_install, "default_dolphot_install_dir", lambda: root)
    monkeypatch.setattr(dolphot_install.shutil, "which", lambda _x: None)
    ok, msgs = verify_acs_wfc_pam_files()
    assert ok is True
    assert msgs == []


def test_verify_acs_wfc_pam_files_missing_one(monkeypatch, tmp_path):
    root = tmp_path / "hst123-dolphot"
    data = root / "dolphot3.1" / "acs" / "data"
    data.mkdir(parents=True)
    (root / "dolphot3.1" / "Makefile").write_text("all:\n", encoding="utf-8")
    (data / "wfc1_pam.fits").write_bytes(b"0" * 2048)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dolphot_install, "default_dolphot_install_dir", lambda: root)
    monkeypatch.setattr(dolphot_install.shutil, "which", lambda _x: None)
    ok, msgs = verify_acs_wfc_pam_files()
    assert ok is False
    assert any("wfc2_pam" in m for m in msgs)


def test_verify_acs_wfc_pam_files_too_small(monkeypatch, tmp_path):
    root = tmp_path / "hst123-dolphot"
    data = root / "dolphot3.1" / "acs" / "data"
    data.mkdir(parents=True)
    (root / "dolphot3.1" / "Makefile").write_text("all:\n", encoding="utf-8")
    for name in ("wfc1_pam.fits", "wfc2_pam.fits"):
        (data / name).write_bytes(b"x")
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dolphot_install, "default_dolphot_install_dir", lambda: root)
    monkeypatch.setattr(dolphot_install.shutil, "which", lambda _x: None)
    ok, msgs = verify_acs_wfc_pam_files()
    assert ok is False
    assert any("too small" in m for m in msgs)


# --- Network tests: download one PSF per instrument ---


@pytest.mark.network
def test_download_single_psf_acs(tmp_path):
    """Download a single ACS PSF file and extract into target dir."""
    install_psfs(tmp_path, instruments=["ACS"], one_per_instrument=True, timeout=30)
    # After extract, tarball is removed; we expect some extracted content
    assert list(tmp_path.iterdir())  # directory not empty
    # ACS_WFC_PAM.tar.gz ships files under dolphot2.0/...; installer copies to acs/data/
    pam_dir = tmp_path / "acs" / "data"
    for name in ACS_WFC_PAM_FILENAMES:
        assert (pam_dir / name).is_file(), f"expected {pam_dir / name} after install_psfs"
        assert (pam_dir / name).stat().st_size >= 512


@pytest.mark.network
def test_download_single_psf_wfc3(tmp_path):
    """Download a single WFC3 PSF file and extract into target dir."""
    install_psfs(tmp_path, instruments=["WFC3"], one_per_instrument=True, timeout=30)
    assert list(tmp_path.iterdir())


@pytest.mark.network
def test_download_single_psf_wfpc2(tmp_path):
    """Download a single WFPC2 PSF file and extract into target dir."""
    install_psfs(tmp_path, instruments=["WFPC2"], one_per_instrument=True, timeout=30)
    assert list(tmp_path.iterdir())
