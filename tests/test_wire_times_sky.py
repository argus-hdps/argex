"""Wire decoders, time conversions and sky geometry, without a server."""

import importlib.util
import os
import pathlib
import struct
from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pytest
from astropy.time import Time

from argex import sky, times, wire
from argex.errors import DecodeError


# ── wire ─────────────────────────────────────────────────────────────────


def test_tiles_round_trip_and_nan_epoch():
    a = np.arange(16, dtype=np.float16).reshape(4, 4)
    blob = wire.encode_tiles(123.5, {(1, 2): a, (0, 0): -a})
    assert struct.unpack_from("<dI", blob) == (123.5, 2)
    epoch, tiles = wire.decode_tiles(blob, 4)
    assert epoch == 123.5 and np.array_equal(tiles[(1, 2)], a)
    assert tiles[(0, 0)].dtype == np.float32
    epoch, tiles = wire.decode_tiles(wire.encode_tiles(None, {}), 4)
    assert epoch is None and tiles == {}
    with pytest.raises(DecodeError):
        wire.decode_tiles(blob[:-1], 4)


def test_tile_preserves_nan():
    a = np.array([[np.nan, 1.0], [2.0, np.inf]], np.float16)
    out = wire.decode_tile(a.astype("<f2").tobytes(), 2)
    assert np.isnan(out[0, 0]) and out[1, 0] == 2.0
    with pytest.raises(DecodeError):
        wire.decode_tile(b"\0" * 6, 2)


def test_segments_and_coverage():
    blob = (
        np.array([5, 7], "<u4").tobytes()
        + np.array([3, 4], "<u4").tobytes()
        + np.array([1.0, 2.0], "<f8").tobytes()
        + np.array([9.0, 8.0], "<f8").tobytes()
    )
    cols = wire.decode_segments(blob)
    assert cols["seg"].tolist() == [5, 7] and cols["epoch_max"].tolist() == [9.0, 8.0]
    with pytest.raises(DecodeError):
        wire.decode_segments(blob[:-1])
    assert wire.decode_coverage(np.array([1, 0], "<u4").tobytes(), 2).tolist() == [1, 0]
    with pytest.raises(DecodeError):
        wire.decode_coverage(np.array([1], "<u4").tobytes(), 2)


# ── times ────────────────────────────────────────────────────────────────


def test_origin_is_tt_new_year():
    assert times.to_epoch("2024-12-31T23:58:50.816") == pytest.approx(0, abs=1e-6)
    assert times.to_iso(0.0) == "2024-12-31T23:58:50.816"
    assert times.to_iso(0.0, "tt") == "2025-01-01T00:00:00.000"
    assert times.to_time(0.0).scale == "tt"


def test_tt_epoch_round_trips_to_utc_in_2026():
    """A 2026 epoch maps to the UTC instant 69.184 s (TT - UTC) earlier on
    the clock than the same TT reading, and back again."""
    e = 56_022_308.24609375
    tt = times.to_iso(e, "tt")
    utc = times.to_iso(e, "utc")
    assert tt == "2026-10-11T09:45:08.246"
    assert utc == "2026-10-11T09:43:59.062"
    gap = Time(tt, scale="tt").datetime - Time(utc, scale="utc").datetime
    assert gap.total_seconds() == pytest.approx(69.184, abs=1e-3)
    assert times.to_epoch(utc) == pytest.approx(e, abs=1e-3)
    assert times.to_epoch(Time(tt, scale="tt")) == pytest.approx(e, abs=1e-3)


def test_to_epoch_forms_agree():
    ref = times.to_epoch("2026-10-15T06:30:00")
    assert times.to_epoch(datetime(2026, 10, 15, 6, 30)) == pytest.approx(ref)
    est = timezone(timedelta(hours=-4))
    aware = datetime(2026, 10, 15, 2, 30, tzinfo=est)
    assert times.to_epoch(aware) == pytest.approx(ref)
    assert times.to_epoch("2026-10-15T02:30:00-04:00") == pytest.approx(ref)
    assert times.to_epoch("2026-10-15T06:30:00Z") == pytest.approx(ref)
    assert times.to_epoch(np.datetime64("2026-10-15T06:30:00")) == pytest.approx(ref)
    t = Time("2026-10-15T06:30:00", scale="utc")
    for scaled in (t, t.tt, t.tai):
        assert times.to_epoch(scaled) == pytest.approx(ref, abs=1e-6)
    assert times.to_epoch(ref) == ref  # numbers are epochs already
    for bad in (None, True, object()):
        with pytest.raises(TypeError):
            times.to_epoch(bad)


def test_leap_second_is_counted():
    """Across the 2016-12-31 leap second, UTC clock readings 1 s apart are
    2 TT seconds apart -- astropy's table, not a fixed offset."""
    a = times.to_epoch("2016-12-31T23:59:59")
    b = times.to_epoch("2017-01-01T00:00:00")
    assert b - a == pytest.approx(2.0, abs=1e-6)


def test_round_trip_datetime_and_mjd():
    e = 56_623_014.71484375
    dt = times.to_datetime(e)
    assert dt.tzinfo == UTC
    assert times.to_epoch(dt) == pytest.approx(e, abs=1e-6)
    assert times.to_mjd(0.0) == 60676.0
    arr = times.to_time(np.array([0.0, 86400.0]))
    assert arr.tt.isot.tolist() == [
        "2025-01-01T00:00:00.000",
        "2025-01-02T00:00:00.000",
    ]


# ── sky ──────────────────────────────────────────────────────────────────


def _server_scheme():
    """The server's sky_scheme.py from an hdps checkout named by HDPS_REPO
    (it is numpy-only, so it loads standalone)."""
    repo = os.environ.get("HDPS_REPO")
    if not repo:
        pytest.skip("HDPS_REPO not set")
    path = (
        pathlib.Path(repo)
        / "applications/data_explorer/src/data_explorer/sky_scheme.py"
    )
    if not path.exists():
        pytest.skip(f"no sky_scheme.py under {repo}")
    spec = importlib.util.spec_from_file_location("_server_sky_scheme", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matches_server_scheme():
    ref = _server_scheme()
    rng = np.random.default_rng(1)
    segs = np.array(ref.northern_sky_ids(dec_floor=-60.0))[::97]
    ra, dec = sky.seg_center(segs)
    rra, rdec = ref.seg_center(segs)
    assert np.array_equal(ra, rra) and np.array_equal(dec, rdec)
    for seg in segs[:40]:
        px, py = rng.uniform(-10, 2058, 2)
        r, d = sky.pixel_to_radec(int(seg), px, py)
        assert (r, d) == pytest.approx(
            tuple(map(float, ref.seg_pixel_to_radec(seg, px, py)))
        )
        assert sky.radec_to_pixel(int(seg), r, d) == pytest.approx((px, py), abs=1e-6)
    for box in [(10, 12, 40, 41), (359.5, 0.5, -1, 1), (100, 101, 88, 89)]:
        assert sky.segments_in_box(*box) == ref.segments_in_box(*box)


def test_segment_of_owns_the_point():
    rng = np.random.default_rng(2)
    for ra, dec in zip(rng.uniform(0, 360, 300), rng.uniform(-88, 88, 300)):
        seg = sky.segment_of(ra, dec)
        hits = sky.segments_containing(ra, dec)
        assert seg in [h[0] for h in hits]
        px, py = sky.radec_to_pixel(seg, ra, dec)
        assert sky.edge_margin(px, py) > 0
    assert sky.segment_of(0.0, 89.9) is None


def test_segments_containing_overlap():
    a = 383117
    ra, dec = sky.pixel_to_radec(a, 2045.0, 1000.0)  # inside a's west overlap
    hits = sky.segments_containing(ra, dec)
    assert len(hits) == 2 and hits[0][0] != a  # the neighbour holds it deeper
    assert hits[1][0] == a and hits[1][1] == pytest.approx(2045.0)


def test_wcs_header_and_minipix():
    header = sky.wcs_header(383117, 100, 200)
    assert header["CRPIX1"] == 924.5 and header["CRPIX2"] == 824.5
    assert header["CD1_1"] == -sky.PIXEL_SCALE_DEG
    assert sky.minipix_index(0, 0) == 0
    assert sky.minipix_index(33.0, 64.5) == 2 * 64 + 1
    assert sky.minipix_index(2047, 2047) == 64 * 64 - 1


def test_astropy_wcs_matches_pixel_math():
    w = sky.wcs(383117, 100, 200, (50, 60))
    assert w.pixel_shape == (60, 50)
    for x, y in [(0, 0), (12.3, 45.6), (59, 49)]:
        world = w.pixel_to_world(x, y)
        ra, dec = sky.pixel_to_radec(383117, x + 100, y + 200)
        assert (world.ra.deg, world.dec.deg) == pytest.approx((ra, dec), abs=1e-10)


def test_minipix_windows():
    assert sky.minipix_origin(64 * 3 + 5) == (160, 96)
    assert sky.minipix_window(7) == (224, 0, 32, 32)
    group = [64 * 3 + 5, 64 * 4 + 6, 64 * 3 + 6]
    assert sky.minipix_window(group) == (160, 96, 64, 64)
    mask = sky.minipix_mask(group)
    assert mask.sum() == 3 * 32 * 32 and not mask[32:, :32].any()
    w = sky.minipix_wcs(383117, group)
    assert w.pixel_shape == (64, 64)
    assert w.wcs.crpix[0] == pytest.approx(sky.CENTER_PX + 1 - 160)
    with pytest.raises(ValueError):
        sky.minipix_origin(4096)
