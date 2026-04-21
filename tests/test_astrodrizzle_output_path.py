"""AstroDrizzle output path normalization / recovery (truncated drizname regression)."""
import logging

from hst123.utils.astrodrizzle_helpers import astrodrizzle_chdir_bundle_for_drizzlepac
from hst123.utils.astrodrizzle_paths import (
    astrodrizzle_output_exists,
    logical_driz_to_internal_astrodrizzle,
    normalize_astrodrizzle_output_path,
    recover_drizzlepac_linear_output,
)


def test_normalize_appends_drz_when_no_fits_suffix(tmp_path):
    log = logging.getLogger("t_norm")
    root = str(tmp_path / "acs.f555w.ut")
    out = normalize_astrodrizzle_output_path(root, log)
    assert out == root + ".drz.fits"


def test_normalize_unchanged_when_already_fits(tmp_path):
    log = logging.getLogger("t_norm2")
    p = str(tmp_path / "x.drz.fits")
    assert normalize_astrodrizzle_output_path(p, log) == p


def test_normalize_drc_unchanged(tmp_path):
    log = logging.getLogger("t_norm_drc")
    p = str(tmp_path / "x.drc.fits")
    assert normalize_astrodrizzle_output_path(p, log) == p


def test_logical_driz_to_internal():
    assert logical_driz_to_internal_astrodrizzle("/a/b/stack.drc.fits") == "/a/b/stack.drz.fits"
    assert logical_driz_to_internal_astrodrizzle("/a/b/stack.drz.fits") == "/a/b/stack.drz.fits"


def test_recover_renames_drz_bundle(tmp_path):
    log = logging.getLogger("t_rec")
    root = str(tmp_path / "acs.f555w.ut")
    (tmp_path / "acs.f555w.ut_drz_sci.fits").write_bytes(b"SIMPLE")
    (tmp_path / "acs.f555w.ut_drz_wht.fits").write_bytes(b"SIMPLE")
    (tmp_path / "acs.f555w.ut_drz_ctx.fits").write_bytes(b"SIMPLE")

    canon = recover_drizzlepac_linear_output(root, log)
    assert canon == root + ".drz.fits"
    assert (tmp_path / "acs.f555w.ut.drz_sci.fits").is_file()
    assert (tmp_path / "acs.f555w.ut.drz_wht.fits").is_file()
    assert (tmp_path / "acs.f555w.ut.drz_ctx.fits").is_file()


def test_recover_noop_when_target_exists(tmp_path):
    log = logging.getLogger("t_rec2")
    f = tmp_path / "exists.drz.fits"
    f.write_bytes(b"x")
    assert recover_drizzlepac_linear_output(str(f), log) == str(f)


def test_astrodrizzle_output_exists_final_file(tmp_path):
    f = tmp_path / "out.drz.fits"
    f.write_bytes(b"x")
    assert astrodrizzle_output_exists(str(f)) is True


def test_astrodrizzle_output_exists_only_sci_sidecar(tmp_path):
    """build=False leaves *_sci.fits before pipeline rename."""
    final_path = tmp_path / "out.drz.fits"
    sci = tmp_path / "out.drz_sci.fits"
    sci.write_bytes(b"x")
    assert astrodrizzle_output_exists(str(final_path)) is True


def test_astrodrizzle_output_exists_missing(tmp_path):
    assert astrodrizzle_output_exists(str(tmp_path / "nope.drz.fits")) is False


def test_astrodrizzle_output_exists_drc_file(tmp_path):
    f = tmp_path / "out.drc.fits"
    f.write_bytes(b"x")
    assert astrodrizzle_output_exists(str(f)) is True


def test_astrodrizzle_output_exists_drc_via_internal_sci(tmp_path):
    """Logical .drc not written yet but internal *_sci.fits present."""
    drc = tmp_path / "out.drc.fits"
    sci = tmp_path / "out.drz_sci.fits"
    sci.write_bytes(b"y")
    assert astrodrizzle_output_exists(str(drc)) is True


def test_astrodrizzle_chdir_bundle_basename_and_relpath(tmp_path):
    """DrizzlePac outroot bug: underscore parents must not appear in output stem."""
    drizzle = tmp_path / "SNII" / "SN1961F" / "drizzle"
    workspace = tmp_path / "SNII" / "SN1961F" / "workspace"
    drizzle.mkdir(parents=True)
    workspace.mkdir(parents=True)
    internal = drizzle / "wfc3_uvis_full.ref.drz.fits"
    inp_a = workspace / "a_flc.drztmp.fits"
    inp_b = workspace / "b_flc.drztmp.fits"
    inp_a.write_bytes(b"x")
    inp_b.write_bytes(b"x")
    rd, ob, rels = astrodrizzle_chdir_bundle_for_drizzlepac(
        str(internal), [str(inp_a), str(inp_b)]
    )
    assert rd == str(drizzle.resolve())
    assert ob == "wfc3_uvis_full.ref.drz.fits"
    assert rels == ["../workspace/a_flc.drztmp.fits", "../workspace/b_flc.drztmp.fits"]
