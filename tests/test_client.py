"""Client against the in-process fake server (conftest.FakeServer)."""

import httpx
import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

import argex
from argex import sky
from conftest import (
    BLUE,
    RED,
    SEG_A,
    SEG_B,
    T0,
    TILE,
    ZMAX,
    FakeServer,
    frame_key,
    pixel_value,
)

RHO_IDX = {k: k for k in range(12)}  # frame index of the k-th rho frame
BETA_IDX = {k: 12 + k for k in range(12)}


def _ep(k):
    return T0 + 63.25 * k


def test_meta_cameras_provenance(client):
    meta = client.meta()
    assert meta.tile == TILE and meta.zmax == ZMAX and meta.tiles_per_side == 8
    assert meta.band_names == ("beta", "rho")
    assert client.band("rho").configs == (RED,)
    assert client.band("rho").cameras == ("AC461M-007313-26",)
    assert meta.cameras == {BLUE: "AC461M-007313-34", RED: "AC461M-007313-26"}
    prov = client.provenance()
    assert prov.server_procver == "srv1234" and prov.archive_procver is None
    assert prov.client_version == argex.__version__


def test_time_inputs_are_tt_epochs(client):
    """ISO strings and datetimes are UTC and convert to TT epochs through
    astropy: in 2026 the epoch exceeds the UTC clock difference from the
    origin by TT - UTC = 69.184 s."""
    first = client.epochs(SEG_A, "2026-10-18T08:00:00", band="rho", limit=1)
    assert first.tolist() == [_ep(0)]
    iso = argex.times.to_iso(_ep(0))
    assert argex.times.to_epoch(iso) == pytest.approx(_ep(0), abs=1e-3)
    utc_clock = (Time(iso, scale="utc") - Time("2025-01-01T00:00:00", scale="utc")).sec
    assert _ep(0) - utc_clock == pytest.approx(69.184, abs=1e-3)


def test_unknown_band_is_checked_before_any_query(client, server):
    client.meta()
    n = len(server.requests)
    with pytest.raises(argex.UnknownBand, match="beta"):
        client.epochs(SEG_A, band="gamma")
    assert len(server.requests) == n


def test_segments_table(client):
    table = client.segments()
    assert len(table) == 2 and SEG_A in table and 12345 not in table
    row = table.row(SEG_A)
    assert row["n_frames"] == 24 and row["epoch_min"] < row["epoch_max"]
    ra, dec = table.centers()
    assert np.allclose(
        (ra[table.index(SEG_A)], dec[table.index(SEG_A)]), sky.seg_center(SEG_A)
    )


def test_coverage_is_sparse_dict(client):
    cov = client.coverage(T0, T0 + 130.0)
    assert cov == {SEG_A: 6, SEG_B: 2}
    assert client.coverage(0, 1) == {}


def test_frames_follow_next_t1(client, server):
    listing = client.frames(SEG_A)  # 24 frames through a cap of 5
    assert len(listing) == 24 and np.all(np.diff(listing.epochs) >= 0)
    want = sorted((_ep(k), c) for k in range(12) for c in (BLUE, RED))
    assert listing.epochs.tolist() == [e for e, _ in want]
    assert listing.configs.tolist() == [c for _, c in want]
    assert listing.frame_keys.tolist() == [frame_key(e, c) for e, c in want]
    assert isinstance(listing.times, Time) and listing.times.scale == "tt"
    rho = client.epochs(SEG_A, band="rho")
    assert rho.tolist() == [_ep(k) for k in range(12)]
    assert client.epochs(SEG_A, band="rho", limit=3).tolist() == rho[:3].tolist()
    assert client.epochs(SEG_B, band="beta").size == 0
    assert all(
        r.url.params.get("detail") == "1"
        for r in server.requests
        if r.url.path == "/epochs"
    )


def test_frames_accept_datetimes(client):
    got = client.epochs(
        SEG_A,
        argex.times.to_time(_ep(0) - 1).utc.isot,
        argex.times.to_datetime(_ep(1)),
        band="rho",
    )
    assert len(got) == 2


def test_nearest_and_frame_identity(client):
    near = client.nearest(SEG_A, _ep(1) + 7, band="beta")
    assert isinstance(near, argex.FrameInfo)
    assert near.epoch == _ep(1) and near.telescope_config == BLUE
    assert near.camera == "AC461M-007313-34" and near.band == "beta"
    assert near.frame_key == frame_key(_ep(1), BLUE)
    assert near.offset_s == pytest.approx(-7.0)
    assert near.tt_int * 256 + near.tt_frac == near.epoch_256
    assert client.nearest(SEG_B, T0, band="beta") is None
    assert client.nearest(SEG_A, _ep(1) + 7, max_dt=1) is None
    by_key = client.frame(SEG_A, key=frame_key(_ep(2), RED))
    assert by_key.epoch == _ep(2) and by_key.band == "rho"
    assert client.frame(SEG_A, key=frame_key(_ep(2) + 1, RED)) is None
    with pytest.raises(TypeError):
        client.frame(SEG_A)


def test_tiles_carry_frame(client):
    tile = client.tile(SEG_A, ZMAX, 2, 1, t=_ep(0), band="rho")
    assert tile.frame.epoch == _ep(0) and tile.frame.telescope_config == RED
    assert tile.data[5, 7] == pixel_value(0, 2 * TILE + 7, TILE + 5)
    pyr = client.tile(SEG_A, 1, 0, 1)
    assert pyr.frame is None and np.all(pyr.data == np.float32(1.5))
    with pytest.raises(TypeError):
        client.tile(SEG_A, ZMAX, 0, 0)  # native needs t or key
    batch = client.tiles(SEG_A, ZMAX, [(0, 0), (7, 7)], t=0, band="beta")
    assert set(batch.tiles) == {(0, 0), (7, 7)}
    assert batch.frame.clamped and batch.frame.epoch == _ep(0)
    empty = client.tiles(SEG_B, ZMAX, [(0, 0)], t=0, band="beta")
    assert empty.frame is None and len(empty) == 0
    assert len(client.tiles(SEG_A, ZMAX, [(0, 0)], t=0, max_dt=60)) == 0


def test_key_selects_the_camera(client):
    """At a shared epoch only the key tells the two cameras' frames apart."""
    for config, idx in ((BLUE, BETA_IDX[3]), (RED, RHO_IDX[3])):
        batch = client.tiles(SEG_A, ZMAX, [(1, 1)], key=frame_key(_ep(3), config))
        assert batch.frame.telescope_config == config
        assert batch.tiles[(1, 1)][0, 0] == pixel_value(idx, TILE, TILE)


def test_cutout_spans_tiles_and_overhangs(client, server):
    cut = client.cutout(SEG_A, -10, 250, 30, 20, t=_ep(4), band="rho")
    assert cut.shape == (20, 30) and cut.epoch == _ep(4)
    assert np.all(np.isnan(cut.data[:, :10]))  # off the segment's left edge
    py, px = np.mgrid[250:270, 0:20]
    assert np.array_equal(cut.data[:, 10:], pixel_value(4, px, py).astype(np.float32))
    xy = server.requests[-1].url.params["xy"]
    assert sorted(xy.split(",")) == ["0_0", "0_1"]


def test_cutout_errors(client):
    with pytest.raises(argex.NoData):
        client.cutout(SEG_B, 0, 0, 16, t=0, band="beta")
    with pytest.raises(ValueError):
        client.cutout(SEG_A, 5000, 0, 16, t=0)


def test_cutout_header_wcs_and_provenance(client):
    ra, dec = sky.pixel_to_radec(SEG_A, 600.2, 700.4)
    cut = client.cutout_at(ra, dec, 33, t=_ep(5) + 2, band="rho")
    assert (cut.x0, cut.y0) == (600 - 16, 700 - 16)
    assert isinstance(cut.wcs, WCS)
    world = cut.wcs.pixel_to_world(16.2, 16.4)
    assert (world.ra.deg, world.dec.deg) == pytest.approx((ra, dec), abs=1e-9)
    h = cut.header()
    assert h["FRAMEKEY"] == frame_key(_ep(5), RED) and h["TELCONF"] == RED
    assert h["CAMID"] == "AC461M-007313-26" and h["BAND"] == "rho"
    assert h["ARGEPOCH"] == _ep(5) and h["ARGE256"] == round(_ep(5) * 256)
    assert h["ARGTTINT"] == int(_ep(5)) and h["ARGFMT"] == 44
    assert h["ARGFILE"].endswith(".arg") and h["ARGDT"] == pytest.approx(-2.0)
    assert h["TIMESYS"] == "TT" and h["MJDREF"] == 60676.0
    assert h["MJD-AVG"] == pytest.approx(60676.0 + _ep(5) / 86400.0)
    assert h["DATE-AVG"] == argex.times.to_time(_ep(5)).isot
    assert "DATE-OBS" not in h  # exposure start is not archived
    assert h["SRVPROC"] == "srv1234" and h["ARGEXVER"] == argex.__version__
    assert "PROCVER" not in h and "archive PROCVER not recorded" in str(h["COMMENT"])
    assert WCS(h).wcs.crpix[0] == pytest.approx(1024.5 - cut.x0)


def test_locate_prefers_segments_with_data(client):
    ra, dec = sky.seg_center(SEG_A)
    assert [h[0] for h in client.locate(ra, dec)] == [SEG_A]
    with pytest.raises(argex.NoData):
        client.cutout_at(10.0, -30.0, 16, t=0)


def test_stack_by_key_keeps_both_cameras(client):
    stack = client.stack(SEG_A, 100, 100, 8, t1=_ep(0), t2=_ep(2))
    assert len(stack) == 6  # three epochs x two cameras
    assert [f.telescope_config for f in stack.frames] == [RED, BLUE] * 3
    assert stack.epochs.tolist() == [_ep(0), _ep(0), _ep(1), _ep(1), _ep(2), _ep(2)]
    assert stack[1].data[0, 0] == pixel_value(BETA_IDX[0], 100, 100)
    rho = stack.select(band="rho")
    assert len(rho) == 3 and {f.band for f in rho.frames} == {"rho"}
    assert np.array_equal(
        rho.light_curve(0, 0), [pixel_value(k, 100, 100) for k in range(3)]
    )
    with pytest.raises(ValueError, match="max_frames"):
        client.stack(SEG_A, 0, 0, 8, band="rho", max_frames=3)


def test_stack_fits_frames_table(client, tmp_path):
    stack = client.stack(SEG_A, 100, 100, 8, t1=_ep(0), t2=_ep(1))
    path = tmp_path / "stack.fits"
    stack.writeto(path)
    with fits.open(path) as hdul:
        assert hdul[0].data.shape == (4, 8, 8)
        h = hdul[0].header
        assert h["NFRAMES"] == 4 and "TELCONF" not in h  # mixed cameras
        assert "TELCONF varies by plane" in str(h["COMMENT"])
        rows = hdul["FRAMES"].data
        assert rows["TELCONF"].tolist() == [RED, BLUE, RED, BLUE]
        assert rows["BAND"].tolist() == ["rho", "beta", "rho", "beta"]
        assert rows["FRAMEKEY"].tolist() == [f.frame_key for f in stack.frames]
        assert rows["CAMID"][1] == "AC461M-007313-34"
        assert hdul["FRAMES"].header["TIMESYS"] == "TT"
        assert rows["UTC_AVG"][0] == argex.times.to_iso(_ep(0))
    one_band = client.stack(SEG_A, 100, 100, 8, t1=_ep(0), t2=_ep(1), band="rho")
    assert one_band.header()["TELCONF"] == RED


def test_minipix_cutout_and_group(client):
    cut = client.minipix_cutout(SEG_A, [64 * 3 + 5, 64 * 4 + 6], t=_ep(0), band="rho")
    assert (cut.x0, cut.y0, cut.shape) == (5 * 32, 3 * 32, (64, 64))
    assert cut.mask.sum() == 2 * 32 * 32
    assert np.isnan(cut.data[0, 40]) and not np.isnan(cut.data[0, 0])
    assert cut.wcs.pixel_shape == (64, 64)
    stack = client.minipix_stack(SEG_A, 7, t1=_ep(0), t2=_ep(1), band="beta")
    assert stack.data.shape == (2, 32, 32) and stack.mask.all()


def test_catalog_and_minipix_routes(client):
    assert client.transients() == {"events": [], "trails": []}
    bogus = client.bogus(0, 10, [SEG_A])
    assert bogus["ra"].tolist() == [50.0] and bogus["seg"].dtype.kind == "i"
    assert client.minipix_active([SEG_A, SEG_B], 5.0) == {
        SEG_A: [3, 70],
        SEG_B: [3, 70],
    }
    assert client.minipix_schedule(SEG_A, 3) == [(10.0, 20.0), (30.0, 40.0)]
    with pytest.raises(argex.BadRequest, match="'idx' must be in 0..4095"):
        client.minipix_schedule(SEG_A, 4096)


def test_http_errors_map_to_exceptions(client):
    with pytest.raises(argex.NotFound, match="no such segment"):
        client.tiles(4242, 0)
    with pytest.raises(argex.BadRequest, match="'z' must be in"):
        client._get("tiles/383117/9")


def test_server_and_transport_errors():
    def boom(request):
        return httpx.Response(502, text="Bad Gateway")

    with argex.Client("http://x/", transport=httpx.MockTransport(boom)) as c:
        with pytest.raises(argex.ServerError) as info:
            c.meta()
    assert info.value.status == 502 and isinstance(info.value, argex.HTTPError)

    def down(request):
        raise httpx.ConnectError("refused", request=request)

    with argex.Client("http://x/", transport=httpx.MockTransport(down)) as c:
        with pytest.raises(argex.TransportError):
            c.meta()


@pytest.mark.parametrize("base", ["http://edge/explorer", "http://edge/explorer/"])
def test_prefixed_base_url(base):
    server = FakeServer(prefix="/explorer")
    with argex.Client(base, transport=httpx.MockTransport(server)) as c:
        assert c.meta().version == "abcd1234"
        assert c.tiles(SEG_A, 0).tiles
    assert all(r.url.path.startswith("/explorer/") for r in server.requests)
