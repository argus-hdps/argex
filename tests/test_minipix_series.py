"""Client.minipix_series: decoding the /minipix/series stream into a
MinipixSeries, the photocal zeropoint model, DQ flags under an aperture,
the per-frame records table, paging and resuming, and finding a minipix by
position."""

import math

import httpx
import numpy as np
import pytest
from astropy.io import fits

import argex
from argex import sky
from conftest import (
    CR_FRAME,
    MINIPIX,
    RED,
    SEG_A,
    SEG_B,
    STAR_XY,
    UNCALIBRATED,
    FakeServer,
    star_zp,
)


def _client(**kw):
    server = FakeServer(minipix=True, **kw)
    return server, argex.Client(
        "http://fake/", transport=httpx.MockTransport(server), backoff=0
    )


@pytest.fixture
def series():
    _server, c = _client(stack_cap=100)
    with c:
        yield c.minipix_series(*MINIPIX)


def test_series_decodes_frames_and_records(series):
    assert (series.seg, series.idx) == MINIPIX
    assert (series.x0, series.y0) == (5 * 32, 7 * 32)
    assert series.data.shape == (24, 32, 32)
    assert list(series.epochs) == sorted(series.epochs)
    rho = series.select(band="rho")
    assert len(rho) == 12 and {f.telescope_config for f in rho.frames} == {RED}
    assert isinstance(rho, argex.MinipixSeries)
    assert rho.has_dq.all()
    assert list(np.flatnonzero(~rho.has_photocal)) == [UNCALIBRATED]
    assert rho.dq_or[CR_FRAME].tolist() == [0, 0, 1 << 15, 0]
    assert rho.n_flagged[CR_FRAME].tolist() == [0, 0, 3, 0]
    assert rho.zp[0, 0] == star_zp(0)
    assert np.isnan(rho.zp[0, 1:]).all() and np.isnan(rho.zp[UNCALIBRATED]).all()
    assert rho.frames[0].data_file.endswith(f"minipix_{MINIPIX[1]:04d}.arg")


def test_zeropoint_model(series):
    rho = series.select(band="rho")
    assert rho.center == (15.5, 15.5)
    np.testing.assert_allclose(rho.zeropoint()[:3], [star_zp(i) for i in range(3)])
    # 0.001 mag per 32 px in x, -0.002 in y, 0.05 per mag of g - r
    got = rho.zeropoint(15.5 + 32, 15.5 - 16, gr=1.6)[0]
    assert got == pytest.approx(star_zp(0) + 0.001 + 0.001 + 0.05)
    assert np.isnan(rho.zeropoint()[UNCALIBRATED])


def test_dq_under_an_aperture(series):
    rho = series.select(band="rho")
    flags = rho.dq(*STAR_XY, 4.0)  # the star sits in the NW quadrant
    assert flags[CR_FRAME] == argex.dq.Flags.COSMIC_RAY
    assert argex.dq.names(flags[CR_FRAME]) == ["COSMIC_RAY"]
    assert (np.delete(flags, CR_FRAME) == 0).all()
    assert rho.dq(4.0, 4.0, 3.0)[CR_FRAME] == 0  # SW only
    assert argex.dq.quadrants(15.5, 15.5, 1.0) == [0, 1, 2, 3]


def test_series_pages_and_resumes():
    server, c = _client(stack_cap=5, drop_stack_after=2)
    with c:
        got = c.minipix_series(*MINIPIX)
    assert len(got) == 24 and len({f.frame_key for f in got.frames}) == 24
    calls = [r for r in server.requests if r.url.path.startswith("/minipix/series")]
    assert len(calls) > math.ceil(24 / 5)  # every page, plus the resumed drop


def test_series_window_and_band():
    _server, c = _client(stack_cap=100)
    with c:
        all_frames = c.minipix_series(*MINIPIX)
        got = c.minipix_series(
            *MINIPIX, t1=all_frames.epochs[4], t2=all_frames.epochs[9], band="beta"
        )
    assert {f.band for f in got.frames} == {"beta"}
    assert list(got.epochs) == sorted(set(all_frames.epochs[4:10]))


def test_unknown_minipix_and_missing_feature():
    _server, c = _client()
    with c, pytest.raises(argex.NotFound):
        c.minipix_series(SEG_A, MINIPIX[1] + 1)
    server = FakeServer(stack=True)
    with argex.Client("http://fake/", transport=httpx.MockTransport(server)) as c:
        with pytest.raises(argex.ExplorerError, match="minipix/series"):
            c.minipix_series(*MINIPIX)


def test_series_at_tries_each_segment():
    # a position in the overlap of SEG_A and SEG_B, deeper inside SEG_B
    ra, dec = sky.pixel_to_radec(SEG_A, 2047.0 - 5, 1000.0)
    hits = sky.segments_containing(ra, dec)
    assert [h[0] for h in hits][:2] == [SEG_B, SEG_A]
    px, py = hits[1][1], hits[1][2]
    idx = sky.minipix_index(px, py)
    server, c = _client(minipix_at=(SEG_A, idx), stack_cap=100)
    with c:
        series, x, y = c.minipix_series_at(ra, dec, band="rho")
        with pytest.raises(argex.NoData):
            c.minipix_series_at(*sky.pixel_to_radec(SEG_A, 100.0, 100.0))
    assert (series.seg, series.idx) == (SEG_A, idx)
    assert (x, y) == pytest.approx((px - series.x0, py - series.y0))
    tried = [r.url.path for r in server.requests if "series" in r.url.path]
    assert tried[0].startswith(f"/minipix/series/{SEG_B}/")


def test_fits_frames_table_carries_the_records(series, tmp_path):
    path = tmp_path / "series.fits"
    series.writeto(path)
    with fits.open(path) as hdul:
        table = hdul["FRAMES"]
        assert table.header["MINIPIX"] == MINIPIX[1]
        assert table.header["GRPIVOT"] == 0.6
        rows = table.data
        assert rows["DQ_OR"].shape == (24, 4) and rows["ZP"].shape == (24, 3)
        np.testing.assert_array_equal(rows["HAS_CAL"], series.has_photocal)


def test_records_table(series):
    rho = series.select(band="rho")
    table = rho.records(*STAR_XY, gr=1.1)
    assert len(table) == 12 and table.meta["idx"] == MINIPIX[1]
    assert list(table["frame_key"]) == [f.frame_key for f in rho.frames]
    assert table["dq"][CR_FRAME] == argex.dq.Flags.COSMIC_RAY
    assert table["dq_names"][CR_FRAME] == "COSMIC_RAY" and table["dq_names"][0] == ""
    assert table["n_flagged"][CR_FRAME].tolist() == [0, 0, 3, 0]
    assert not table["has_photocal"][UNCALIBRATED] and np.isnan(
        table["zp"][UNCALIBRATED]
    )
    assert table["zp"][0] == star_zp(0)
    np.testing.assert_allclose(table["zp_at"], rho.zeropoint(*STAR_XY, 1.1))
    assert table["utc"][0].startswith("2026-")
    assert "zp_at" not in rho.records().colnames


def test_read_round_trips_the_fits_products(series, tmp_path):
    path = tmp_path / "series.fits"
    series.writeto(path)
    got = argex.read(path)
    assert isinstance(got, argex.MinipixSeries) and got.idx == series.idx
    np.testing.assert_array_equal(got.data, series.data)
    assert got.frames == series.frames
    for name in ("dq_or", "n_flagged", "has_dq", "has_photocal"):
        np.testing.assert_array_equal(getattr(got, name), getattr(series, name))
    rho, rho_got = series.select(band="rho"), got.select(band="rho")
    np.testing.assert_array_equal(
        rho_got.zeropoint(*STAR_XY, 1.1), rho.zeropoint(*STAR_XY, 1.1)
    )
    assert rho_got.dq(*STAR_XY, 4.0)[CR_FRAME] == argex.dq.Flags.COSMIC_RAY

    stack = argex.CutoutStack(
        series.seg, series.x0, series.y0, series.data, series.frames, series.provenance
    )
    stack.writeto(tmp_path / "stack.fits")
    got = argex.read(tmp_path / "stack.fits")
    assert type(got) is argex.CutoutStack and got.frames == stack.frames

    series[3].writeto(tmp_path / "cut.fits")
    got = argex.read(tmp_path / "cut.fits")
    assert isinstance(got, argex.Cutout) and got.frame == series.frames[3]
    np.testing.assert_array_equal(got.data, series.data[3])
