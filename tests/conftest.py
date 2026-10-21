"""An in-process fake of the explorer server's read API, served to the client
through httpx.MockTransport. It mirrors the server's wire contract: /meta's
epoch origin, cameras and provenance; /epochs sorted with frame keys and
next_t1 paging under a small cap; frame identity on /nearest, /frame and
the X-Frame-* headers of native tiles; key= and max_dt=; bare-ish 400s.
With stack=True it also advertises and serves the /stack stream, paged
under stack_cap; drop_stack_after=n cuts the first stream after n frames."""

import math

import httpx
import numpy as np
import pytest

import argex
from argex import wire

TILE, SIDE, ZMAX = 256, 2048, 3
SEG_A, SEG_B = 383117, 383116  # SEG_B holds no frames in band "beta"
BLUE, RED = 39835, 35048
CAMERAS = {BLUE: "AC461M-007313-34", RED: "AC461M-007313-26"}
BAND_OF = {BLUE: "beta", RED: "rho"}
T0 = 56_623_014.71484375  # on the 1/256 s grid

META = {
    "mode": "sky",
    "tile": TILE,
    "side": SIDE,
    "zmax": ZMAX,
    "n_segments": 2,
    "n_frames_total": 0,  # filled below
    "bytes_total": 0,
    "epoch_min": 0.0,
    "epoch_max": 0.0,
    "version": "abcd1234",
    "vmin": 0.3,
    "vmax": 10.0,
    "bg_level": 0.55,
    "bands": [
        {
            "name": "beta",
            "label": "β",
            "passband": "365–520 nm",
            "configs": [BLUE],
            "cameras": [CAMERAS[BLUE]],
        },
        {
            "name": "rho",
            "label": "ρ",
            "passband": "565–825 nm",
            "configs": [RED],
            "cameras": [CAMERAS[RED]],
        },
    ],
    "minipix_grid": 64,
    "scheme": {},
    "writable": False,
    "epoch_scale": "tt",
    "epoch_origin": "2025-01-01T00:00:00",
    "cameras": {str(k): v for k, v in CAMERAS.items()},
    "provenance": {
        "server_procver": "srv1234",
        "archive_procver": None,
        "archive_format": 44,
        "frame_key": "(telescope_config << 40) | epoch_256 (ADR-0024)",
        "frame_key_epoch_bits": 40,
    },
}


def pixel_value(frame_idx, px, py):
    """Synthetic pixel value (scalar or array): exact in fp16, and distinct
    per frame and across neighbouring positions."""
    return np.asarray((px * 7 + py * 13) % 997 + frame_idx, dtype=np.float64)


def _frames():
    """{seg: [(epoch, config), ...]} in on-disk order (per camera): the two
    cameras expose together, so every epoch is shared."""
    rho = [(T0 + 63.25 * k, RED) for k in range(12)]
    beta = [(T0 + 63.25 * k, BLUE) for k in range(12)]
    return {SEG_A: rho + beta, SEG_B: [(T0 + 120.0 * k, RED) for k in range(5)]}


def frame_key(epoch, config):
    return (config << 40) | int(round(epoch * 256))


class FakeServer:
    def __init__(
        self,
        epochs_cap=5,
        prefix="",
        meta_overrides=None,
        stack=False,
        stack_cap=4,
        drop_stack_after=None,
    ):
        self.frames = _frames()
        self.stack_cap = stack_cap
        self.drop_stack_after = drop_stack_after
        self.epochs_cap = epochs_cap
        self.prefix = prefix
        self.requests = []
        all_eps = [e for fr in self.frames.values() for e, _ in fr]
        self.meta = dict(
            META,
            n_frames_total=len(all_eps),
            epoch_min=min(all_eps),
            epoch_max=max(all_eps),
        )
        if stack:
            self.meta["features"] = ["stack"]
        self.meta.update(meta_overrides or {})
        self.bands = {b["name"]: set(b["configs"]) for b in META["bands"]}

    # helpers ------------------------------------------------------------
    def _band(self, q):
        name = q.get("band", "")
        if not name:
            return None
        if name not in self.bands:
            raise _Status(400, "bad request: 'band' must be one of beta, rho")
        return self.bands[name]

    def _frames_of(self, seg, configs):
        return [
            (i, e, c)
            for i, (e, c) in enumerate(self.frames.get(seg, []))
            if configs is None or c in configs
        ]

    def identity(self, seg, i, t=None, clamped=False):
        e, c = self.frames[seg][i]
        out = {
            "seg": seg,
            "epoch": e,
            "epoch_256": int(round(e * 256)),
            "telescope_config": c,
            "frame_key": str(frame_key(e, c)),
            "format": 44,
            "checksum": i,
            "data_file": f"2026_3/266/seg_{seg:09d}/segment_42.arg",
            "byte_offset": 1000 * i,
            "camera": CAMERAS.get(c),
            "band": BAND_OF.get(c),
            "clamped": clamped,
            "fill": False,
        }
        if t is not None:
            out.update(requested_t=t, offset_s=e - t)
        return out

    def _nearest(self, seg, t, configs, max_dt=None):
        cands = self._frames_of(seg, configs)
        if not cands:
            return None, False
        eps = [e for _, e, _ in cands]
        tc = min(max(t, min(eps)), max(eps))
        i, e, _c = min(cands, key=lambda iec: (abs(iec[1] - tc), iec[2]))
        if max_dt is not None and abs(e - t) > max_dt:
            return None, False
        return i, tc != t

    def _by_key(self, seg, key):
        key = int(key)
        for i, (e, c) in enumerate(self.frames.get(seg, [])):
            if frame_key(e, c) == key:
                return i
        return None

    @staticmethod
    def _float(q, name, default=0.0):
        value = float(q.get(name, default))
        if not math.isfinite(value):
            raise _Status(400, f"bad request: {name!r} must be a finite number")
        return value

    # routing ------------------------------------------------------------
    def __call__(self, request):
        self.requests.append(request)
        path = request.url.path
        if not path.startswith(self.prefix + "/"):
            return httpx.Response(404, text="Not Found")
        path = path[len(self.prefix) :]
        q = dict(request.url.params)
        try:
            return self.route(path, q)
        except _Status as exc:
            return httpx.Response(exc.status, text=exc.body)

    def route(self, path, q):
        parts = path.strip("/").split("/")
        if path == "/meta":
            return httpx.Response(200, json=self.meta)
        if path == "/segments.bin":
            segs = sorted(self.frames)
            eps = [[e for e, _ in self.frames[s]] for s in segs]
            blob = (
                np.array(segs, "<u4").tobytes()
                + np.array([len(e) for e in eps], "<u4").tobytes()
                + np.array([min(e) for e in eps], "<f8").tobytes()
                + np.array([max(e) for e in eps], "<f8").tobytes()
            )
            return httpx.Response(200, content=blob)
        if path == "/coverage":
            t1, t2 = self._float(q, "t1"), self._float(q, "t2")
            counts = [
                sum(t1 <= e <= t2 for e, _ in self.frames[s])
                for s in sorted(self.frames)
            ]
            return httpx.Response(200, content=np.array(counts, "<u4").tobytes())
        if path == "/epochs":
            return self.epochs(q)
        if path == "/nearest":
            configs = self._band(q)
            seg, t = int(q.get("seg", -1)), self._float(q, "t")
            max_dt = float(q["max_dt"]) if "max_dt" in q else None
            i, clamped = self._nearest(seg, t, configs, max_dt)
            if i is None:
                return httpx.Response(200, json={"epoch": None})
            return httpx.Response(200, json=self.identity(seg, i, t, clamped))
        if path == "/frame":
            seg = int(q.get("seg", -1))
            if "key" in q:
                i, t = self._by_key(seg, q["key"]), None
            else:
                t = self._float(q, "t")
                i, _ = self._nearest(seg, t, self._band(q))
            if i is None:
                raise _Status(404, "no such frame")
            return httpx.Response(200, json=self.identity(seg, i, t))
        if parts[0] in ("tile", "tiles"):
            return self.tiles(parts, q)
        if parts[0] == "stack" and "stack" in self.meta.get("features", ()):
            return self.stack(parts, q)
        if path == "/transients":
            return httpx.Response(200, json={"events": [], "trails": []})
        if path == "/bogus":
            return httpx.Response(
                200,
                json={
                    "seg": [SEG_A],
                    "t": [1.5],
                    "ra": [50.0],
                    "dec": [41.0],
                    "rb": [0.2],
                    "kind": [1],
                },
            )
        if path == "/minipix/active":
            segs = [s for s in q.get("segs", "").split(",") if s]
            return httpx.Response(200, json={s: [3, 70] for s in segs})
        if path == "/minipix/schedule":
            idx = int(q["idx"])
            if not 0 <= idx < 4096:
                raise _Status(400, "bad request: 'idx' must be in 0..4095")
            return httpx.Response(
                200,
                json={
                    "seg": int(q["seg"]),
                    "idx": idx,
                    "intervals": [[10.0, 20.0], [30.0, 40.0]],
                },
            )
        raise _Status(404, "Not Found")

    def epochs(self, q):
        configs = self._band(q)
        seg = int(q.get("seg", -1))
        t1, t2 = self._float(q, "t1"), self._float(q, "t2")
        t1q = math.floor(t1 * 256) / 256
        rows = sorted(
            (e, c) for _i, e, c in self._frames_of(seg, configs) if t1q <= e <= t2
        )
        out = {"epochs": [], "truncated": False, "next_t1": None}
        if len(rows) > self.epochs_cap:
            cut = self.epochs_cap
            while cut > 0 and rows[cut][0] == rows[cut - 1][0]:
                cut -= 1
            out["truncated"], out["next_t1"] = True, rows[cut][0]
            rows = rows[:cut]
        out["epochs"] = [e for e, _ in rows]
        if q.get("detail") == "1":
            out["frame_keys"] = [str(frame_key(e, c)) for e, c in rows]
            out["configs"] = [c for _, c in rows]
        return httpx.Response(200, json=out)

    def tiles(self, parts, q):
        configs = self._band(q)
        seg, z = int(parts[1]), int(parts[2])
        if seg not in self.frames:
            raise _Status(404, "no such segment")
        if not 0 <= z <= ZMAX:
            raise _Status(400, "bad request: 'z' must be in 0..3")
        if parts[0] == "tile":
            xy = [(int(parts[3]), int(parts[4]))]
        else:
            xy = [
                tuple(int(v) for v in p.split("_"))
                for p in q.get("xy", "").split(",")
                if p
            ]
        n = 1 << z
        xy = [(x, y) for x, y in xy if 0 <= x < n and 0 <= y < n]
        headers, tiles, epoch = {}, {}, None
        if z < ZMAX:
            tiles = {(x, y): np.full((TILE, TILE), z + 0.5, np.float16) for x, y in xy}
        else:
            if "key" in q:
                i, clamped, t = self._by_key(seg, q["key"]), False, None
            else:
                t = self._float(q, "t")
                max_dt = float(q["max_dt"]) if "max_dt" in q else None
                i, clamped = self._nearest(seg, t, configs, max_dt)
            if i is not None:
                ident = self.identity(seg, i, t, clamped)
                epoch = ident["epoch"]
                headers = {
                    "X-Epoch": repr(epoch),
                    "X-Epoch-Tt": repr(epoch),
                    "X-Frame-Key": ident["frame_key"],
                    "X-Telescope-Config": str(ident["telescope_config"]),
                    "X-Camera": ident["camera"],
                    "X-Band": ident["band"],
                    "X-Archive-Format": "44",
                    "X-Frame-Checksum": str(ident["checksum"]),
                    "X-Data-File": ident["data_file"],
                    "X-Byte-Offset": str(ident["byte_offset"]),
                    "X-Clamped": "1" if clamped else "0",
                    "X-Fill": "1" if q.get("fill") == "1" else "0",
                }
                if t is not None:
                    headers["X-Requested-T"] = repr(t)
                    headers["X-Epoch-Offset"] = repr(epoch - t)
                py, px = np.mgrid[0:TILE, 0:TILE]
                for x, y in xy:
                    tiles[(x, y)] = pixel_value(i, px + x * TILE, py + y * TILE).astype(
                        np.float16
                    )
        if parts[0] == "tiles":
            return httpx.Response(
                200, content=wire.encode_tiles(epoch, tiles), headers=headers
            )
        if not tiles:
            raise _Status(404, "no such tile")
        (arr,) = tiles.values()
        return httpx.Response(
            200, content=np.ascontiguousarray(arr, "<f2").tobytes(), headers=headers
        )


def _stack_body(server, seg, rows, x0, y0, nx, ny, end):
    """The /stack records of `rows` [(i, epoch, config)], cut short (a
    dropped connection) after server.drop_stack_after frames, once."""
    py, px = np.mgrid[y0 : y0 + ny, x0 : x0 + nx]
    outside = (px < 0) | (px >= SIDE) | (py < 0) | (py >= SIDE)
    frames = []
    for i, _e, _c in rows:
        pixels = pixel_value(i, px, py)
        pixels[outside] = np.nan
        frames.append((server.identity(seg, i), pixels))
    body = wire.encode_stack(seg, x0, y0, nx, ny, frames, end)
    drop, server.drop_stack_after = server.drop_stack_after, None
    if drop is None:
        yield body
        return
    # the stack record, `drop` frame records, then the connection fails
    pos = 0
    for _ in range(1 + drop):
        (n,) = np.frombuffer(body, "<u4", 1, pos)
        pos += 4 + int(n) + (0 if pos == 0 else nx * ny * 2)
    yield body[:pos]
    raise httpx.ReadError("connection reset")


def _stack(self, parts, q):
    seg = int(parts[1])
    if seg not in self.frames:
        raise _Status(404, "no such segment")
    configs = self._band(q)
    x0, y0, nx = int(q["x0"]), int(q["y0"]), int(q["nx"])
    ny = int(q.get("ny", nx))
    t1, t2 = self._float(q, "t1", -1e18), self._float(q, "t2", 1e18)
    t1q = math.floor(t1 * 256) / 256
    rows = sorted(
        ((i, e, c) for i, e, c in self._frames_of(seg, configs) if t1q <= e <= t2),
        key=lambda r: (r[1], r[2]),
    )
    end = {"frames": len(rows), "truncated": False, "next_t1": None}
    if len(rows) > self.stack_cap:
        cut = self.stack_cap
        while cut > 0 and rows[cut][1] == rows[cut - 1][1]:
            cut -= 1
        end = {"frames": cut, "truncated": True, "next_t1": rows[cut][1]}
        rows = rows[:cut]
    return httpx.Response(
        200, content=_stack_body(self, seg, rows, x0, y0, nx, ny, end)
    )


FakeServer.stack = _stack


class _Status(Exception):
    def __init__(self, status, body):
        self.status, self.body = status, body


@pytest.fixture
def server():
    return FakeServer()


@pytest.fixture
def client(server):
    with argex.Client(
        "http://fake/", transport=httpx.MockTransport(server), max_workers=2
    ) as c:
        yield c
