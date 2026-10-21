"""Smoke test against a running explorer server. Skipped unless EXPLORER_URL
is set, e.g. ``EXPLORER_URL=http://localhost:8765/ pytest tests/test_live.py``.
Issues a handful of small requests."""

import os

import numpy as np
import pytest

import argex

URL = os.environ.get("EXPLORER_URL")
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not URL, reason="set EXPLORER_URL to run"),
]


@pytest.fixture(scope="module")
def ex():
    with argex.Client(URL, timeout=60.0) as client:
        yield client


@pytest.fixture(scope="module")
def busiest(ex):
    table = ex.segments()
    return int(table.seg[np.argmax(table.n_frames)])


def test_meta(ex):
    meta = ex.meta()
    assert meta.n_segments == len(ex.segments()) > 0
    assert meta.epoch_min < meta.epoch_max and meta.tile * (1 << meta.zmax) == meta.side
    assert meta.provenance["archive_format"] == 44


def test_frames_nearest_and_cutout(ex, busiest):
    band = ex.bands[0].name if ex.bands else None
    listing = ex.frames(busiest, band=band)
    if not len(listing):
        pytest.skip("the busiest segment has no frames in this band")
    near = ex.nearest(busiest, listing.epochs[-1], band=band)
    assert near.epoch == listing.epochs[-1]
    assert near.frame_key == listing.frame_keys[-1]
    assert ex.frame(busiest, key=near.frame_key).epoch == near.epoch
    ra, dec = argex.sky.pixel_to_radec(busiest, 1024, 1024)
    cut = ex.cutout_at(ra, dec, 64, key=near.frame_key, seg=busiest)
    assert cut.shape == (64, 64) and cut.frame.frame_key == near.frame_key
    assert cut.header()["FRAMEKEY"] == near.frame_key


def test_small_mixed_stack(ex, busiest):
    last = ex.frames(busiest).epochs[-1]
    stack = ex.stack(busiest, 1000, 1000, 16, t1=last - 400, t2=last, max_frames=40)
    assert stack.data.shape[1:] == (16, 16)
    assert np.all(np.diff(stack.epochs) >= 0)
    assert len({f.frame_key for f in stack.frames}) == len(stack)


def test_errors(ex, busiest):
    with pytest.raises(argex.UnknownBand):
        ex.nearest(busiest, 0, band="no-such-band")
    with pytest.raises(argex.BadRequest, match="'z'"):
        ex.tiles(busiest, ex.meta().zmax + 1, [(0, 0)])
    with pytest.raises(argex.NotFound):
        ex.tile(1, 0, 0, 0)


def test_coverage_and_side_routes(ex, busiest):
    meta = ex.meta()
    cov = ex.coverage(meta.epoch_max - 86400, meta.epoch_max)
    assert all(n > 0 for n in cov.values())
    assert isinstance(ex.transients(), dict)
    assert busiest in ex.minipix_active([busiest], meta.epoch_max)
