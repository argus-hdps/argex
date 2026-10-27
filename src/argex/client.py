"""HTTP client for the Argus data explorer tile server."""

import math
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np

from . import __version__, sky, times, wire
from .errors import (
    BadRequest,
    DecodeError,
    ExplorerError,
    HTTPError,
    NoData,
    NotFound,
    ServerError,
    TransportError,
    UnknownBand,
)
from .models import (
    Band,
    Cutout,
    CutoutStack,
    FrameInfo,
    FrameList,
    Meta,
    MinipixSeries,
    Provenance,
    SegmentTable,
    Tile,
    TileBatch,
)

DEFAULT_URL = "http://localhost:8765/"


class Client:
    """Read-only client for one explorer server.

    ``base_url`` is the server root or a web-edge prefix that proxies it
    (e.g. ``https://host/explorer/``). Times are taken as Argus epochs (TT
    seconds since 2025-01-01T00:00:00 TT) or as anything
    :func:`argex.times.as_time` accepts (datetimes, ISO strings, astropy
    Time), and returned as epochs; :func:`argex.times.to_time` converts them
    to astropy Time. ``band`` is a band name from :attr:`bands` (e.g. ``"beta"``,
    ``"rho"``), a :class:`Band`, or None for every camera.

    The client holds one connection pool; use it as a context manager or
    call :meth:`close`. It is safe to share between threads. A request that
    fails to connect, drops mid-response or meets a 502/503/504 from a web
    edge is retried up to ``retries`` times, waiting ``backoff`` seconds and
    doubling; every request is a read, so a retry is always safe.
    """

    def __init__(
        self,
        base_url=DEFAULT_URL,
        *,
        timeout=30.0,
        headers=None,
        max_workers=4,
        transport=None,
        verify=True,
        retries=3,
        backoff=0.5,
    ):
        hdrs = {"User-Agent": f"argex/{__version__}"}
        hdrs.update(headers or {})
        self._http = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers=hdrs,
            transport=transport,
            verify=verify,
            follow_redirects=True,
        )
        self.max_workers = max_workers
        self.retries = max(0, int(retries))
        self.backoff = float(backoff)
        self._meta = None
        self._segments = None

    # ── session ────────────────────────────────────────────────────────

    @property
    def base_url(self):
        return str(self._http.base_url)

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return f"Client({self.base_url!r})"

    _RETRY_STATUS = frozenset({502, 503, 504})

    def _attempts(self):
        """Delays before each attempt: 0, then backoff doubling."""
        yield 0.0
        for k in range(self.retries):
            yield self.backoff * (2**k)

    def _get(self, path, **params):
        params = {k: v for k, v in params.items() if v is not None}
        failure = None
        for delay in self._attempts():
            time.sleep(delay)
            try:
                response = self._http.get(path, params=params)
            except httpx.TransportError as exc:
                failure = TransportError(f"{type(exc).__name__} for {path}: {exc}")
                continue
            if response.status_code in self._RETRY_STATUS:
                failure = self._http_error(response)
                continue
            if response.is_success:
                return response
            raise self._http_error(response)
        raise failure

    @staticmethod
    def _http_error(response):
        status, body = response.status_code, response.text.strip()[:300]
        cls = (
            BadRequest
            if status == 400
            else NotFound
            if status == 404
            else ServerError
            if status >= 500
            else HTTPError
        )
        return cls(status, response.url, body)

    def _json(self, path, **params):
        return self._get(path, **params).json()

    # ── archive summary and provenance ─────────────────────────────────

    def meta(self, refresh=False):
        """Archive summary and serving constants (cached; ``refresh`` refetches)."""
        if self._meta is None or refresh:
            self._meta = Meta.from_json(self._json("meta"))
        return self._meta

    def provenance(self):
        """Processing versions stamped on products from this server."""
        prov = self.meta().provenance
        return Provenance(
            archive_procver=prov.get("archive_procver"),
            server_procver=prov.get("server_procver"),
            client_version=__version__,
        )

    @property
    def bands(self):
        """The bands this server serves, blue to red."""
        return self.meta().bands

    def band(self, name):
        """The :class:`Band` named ``name``; :class:`UnknownBand` otherwise."""
        for band in self.bands:
            if band.name == name:
                return band
        raise UnknownBand(
            f"unknown band {name!r}; this server serves {list(self.meta().band_names)}"
        )

    def _band(self, band):
        if band is None or band == "":
            return None
        name = band.name if isinstance(band, Band) else str(band)
        return self.band(name).name

    def segments(self, refresh=False):
        """Every segment holding data (cached; ``refresh`` refetches)."""
        if self._segments is None or refresh:
            blob = self._get("segments.bin").content
            self._segments = SegmentTable(
                **wire.decode_segments(blob), version=self.meta(refresh).version
            )
        return self._segments

    def coverage(self, t1, t2):
        """{seg: n_epochs} for every segment with frames in [t1, t2] (all
        bands). Counts are interpolated from weekly buckets, so they are
        estimates; use :meth:`frames` for exact lists."""
        blob = self._get(
            "coverage", t1=times.to_epoch(t1), t2=times.to_epoch(t2)
        ).content
        table = self.segments()
        if len(blob) != 4 * len(table):
            table = self.segments(refresh=True)
        counts = wire.decode_coverage(blob, len(table))
        hit = np.flatnonzero(counts)
        return dict(zip(table.seg[hit].tolist(), counts[hit].tolist()))

    # ── frames ─────────────────────────────────────────────────────────

    def _window(self, t1, t2):
        meta = self.meta()
        lo = times.to_epoch(t1) if t1 is not None else meta.epoch_min - 1.0
        hi = times.to_epoch(t2) if t2 is not None else meta.epoch_max + 1.0
        return lo, hi

    def frames(self, seg, t1=None, t2=None, *, band=None, limit=None):
        """A segment's frames in [t1, t2] (default: the whole archive),
        ascending by epoch, with their frame keys and telescope_configs.
        With ``band=None`` co-pointed cameras that expose together each
        contribute a frame at the same epoch. ``limit`` keeps the earliest
        that many. Pages of the server's 4000-frame replies are followed
        through their ``next_t1``."""
        band = self._band(band)
        lo, hi = self._window(t1, t2)
        eps, keys, cfgs = [], [], []
        while True:
            reply = self._json(
                "epochs", seg=int(seg), t1=lo, t2=hi, band=band, detail=1
            )
            eps += reply.get("epochs", [])
            keys += [int(k) for k in reply.get("frame_keys", [])]
            cfgs += reply.get("configs", [])
            if limit is not None and len(eps) >= limit:
                break
            next_t1 = reply.get("next_t1")
            if not reply.get("truncated") or next_t1 is None:
                break
            lo = float(next_t1)
        n = len(eps) if limit is None else min(limit, len(eps))
        return FrameList(
            int(seg),
            np.asarray(eps[:n], np.float64),
            np.asarray(keys[:n], np.int64),
            np.asarray(cfgs[:n], np.int64),
        )

    def epochs(self, seg, t1=None, t2=None, *, band=None, limit=None):
        """Sorted exact epochs of a segment's frames (:meth:`frames`)."""
        return self.frames(seg, t1, t2, band=band, limit=limit).epochs

    def nearest(self, seg, t, *, band=None, max_dt=None):
        """:class:`FrameInfo` of the segment's frame nearest ``t``, or None
        (no frames, or none within ``max_dt`` seconds)."""
        reply = self._json(
            "nearest",
            seg=int(seg),
            t=times.to_epoch(t),
            band=self._band(band),
            max_dt=max_dt,
        )
        if reply.get("epoch") is None:
            return None
        return FrameInfo.from_json(reply)

    def frame(self, seg, *, t=None, key=None, band=None, max_dt=None):
        """:class:`FrameInfo` of the frame with frame key ``key``, else the
        frame nearest ``t``; None when there is no such frame."""
        if key is None and t is None:
            raise TypeError("give t or key")
        try:
            reply = self._json(
                "frame",
                seg=int(seg),
                key=str(int(key)) if key is not None else None,
                t=times.to_epoch(t) if key is None else None,
                band=self._band(band),
                max_dt=max_dt,
            )
        except NotFound as exc:
            if "no such frame" in exc.body:
                return None
            raise
        return FrameInfo.from_json(reply)

    # ── sky positions ──────────────────────────────────────────────────

    def locate(self, ra, dec):
        """[(seg, px, py), ...] of the segments holding data that contain
        (ra, dec), deepest inside first. Empty if none do."""
        table = self.segments()
        return [h for h in sky.segments_containing(ra, dec) if h[0] in table]

    def _locate1(self, ra, dec):
        hits = self.locate(ra, dec)
        if not hits:
            raise NoData(f"no segment with data contains ra={ra}, dec={dec}")
        return hits[0]

    def frames_at(self, ra, dec, t1=None, t2=None, *, band=None, limit=None):
        """:meth:`frames` of the best segment containing (ra, dec)."""
        seg = self._locate1(ra, dec)[0]
        return self.frames(seg, t1, t2, band=band, limit=limit)

    def nearest_at(self, ra, dec, t, *, band=None, max_dt=None):
        """:meth:`nearest` in the best segment containing (ra, dec)."""
        seg = self._locate1(ra, dec)[0]
        return self.nearest(seg, t, band=band, max_dt=max_dt)

    # ── tiles ──────────────────────────────────────────────────────────

    def _native_params(self, z, t, key, max_dt):
        if z != self.meta().zmax:
            return {}
        if key is not None:
            return {"key": str(int(key))}
        if t is None:
            raise TypeError("native tiles need a time t or a frame key")
        epoch = times.to_epoch(t)
        if not math.isfinite(epoch):
            raise ValueError(f"non-finite time {t!r}")
        return {"t": epoch, "max_dt": max_dt}

    def tile(
        self, seg, z, x, y, *, t=None, key=None, band=None, fill=False, max_dt=None
    ):
        """One tile. At the native level (z = meta().zmax) it comes from the
        frame with frame key ``key``, else the frame nearest ``t``; below it
        from the epoch-averaged pyramid. Raises :class:`NotFound` if the
        segment or tile is absent (or no frame lies within ``max_dt``)."""
        response = self._get(
            f"tile/{int(seg)}/{int(z)}/{int(x)}/{int(y)}",
            band=self._band(band),
            fill=1 if fill else None,
            **self._native_params(z, t, key, max_dt),
        )
        return Tile(
            int(seg),
            int(z),
            int(x),
            int(y),
            wire.decode_tile(response.content, self.meta().tile),
            FrameInfo.from_headers(seg, response.headers),
        )

    def tiles(
        self, seg, z, xy=None, *, t=None, key=None, band=None, fill=False, max_dt=None
    ):
        """Several tiles of one segment and level in one request; ``xy`` is
        an iterable of (x, y), default every tile of the level. Native tiles
        all come from one frame (``key``, else nearest ``t``); the batch is
        empty when there is none (or none within ``max_dt``). ``fill``
        patches no-data pixels from other frames within 20 min."""
        if xy is None:
            n = 1 << int(z)
            xy = [(x, y) for y in range(n) for x in range(n)]
        response = self._get(
            f"tiles/{int(seg)}/{int(z)}",
            xy=",".join(f"{int(x)}_{int(y)}" for x, y in xy),
            band=self._band(band),
            fill=1 if fill else None,
            **self._native_params(z, t, key, max_dt),
        )
        _epoch, tiles = wire.decode_tiles(response.content, self.meta().tile)
        frame = FrameInfo.from_headers(seg, response.headers)
        return TileBatch(int(seg), int(z), frame if tiles else None, tiles)

    def overview(self, seg, z=0, *, band=None):
        """The epoch-averaged pyramid image of a segment at level z < zmax,
        (tile << z) px square; NaN where the pyramid has no tile."""
        meta = self.meta()
        if not 0 <= z < meta.zmax:
            raise ValueError(f"pyramid levels are 0..{meta.zmax - 1}")
        batch = self.tiles(seg, z, band=band)
        n = meta.tile << z
        image = np.full((n, n), np.nan, np.float32)
        t = meta.tile
        for (x, y), arr in batch.tiles.items():
            image[y * t : (y + 1) * t, x * t : (x + 1) * t] = arr
        return image

    # ── cutouts ────────────────────────────────────────────────────────

    def _window_tiles(self, x0, y0, nx, ny):
        tile, side = self.meta().tile, self.meta().side
        xa, xb = max(0, x0), min(side, x0 + nx)
        ya, yb = max(0, y0), min(side, y0 + ny)
        if xa >= xb or ya >= yb:
            raise ValueError(f"window ({x0}, {y0}, {nx}x{ny}) misses the segment")
        return [
            (tx, ty)
            for ty in range(ya // tile, (yb - 1) // tile + 1)
            for tx in range(xa // tile, (xb - 1) // tile + 1)
        ]

    def _assemble(self, batch, x0, y0, nx, ny, mask=None):
        tile = self.meta().tile
        out = np.full((ny, nx), np.nan, np.float32)
        for (tx, ty), arr in batch.tiles.items():
            # overlap of tile [tx*tile, +tile) with window [x0, x0+nx), in segment px
            ax, bx = max(tx * tile, x0), min((tx + 1) * tile, x0 + nx)
            ay, by = max(ty * tile, y0), min((ty + 1) * tile, y0 + ny)
            if ax < bx and ay < by:
                out[ay - y0 : by - y0, ax - x0 : bx - x0] = arr[
                    ay - ty * tile : by - ty * tile, ax - tx * tile : bx - tx * tile
                ]
        if mask is not None:
            out[~mask] = np.nan
        return out

    def cutout(
        self,
        seg,
        x0,
        y0,
        nx,
        ny=None,
        *,
        t=None,
        key=None,
        band=None,
        fill=False,
        max_dt=None,
        mask=None,
    ):
        """Native pixels [y0, y0+ny) x [x0, x0+nx) of one frame (``key``, else
        nearest ``t``). The window may overhang the segment (NaN there);
        ``mask`` (ny, nx) bool NaNs the pixels outside it. Raises
        :class:`NoData` if there is no such frame."""
        ny = nx if ny is None else ny
        x0, y0, nx, ny = int(x0), int(y0), int(nx), int(ny)
        xy = self._window_tiles(x0, y0, nx, ny)
        name = self._band(band)
        batch = self.tiles(
            seg, self.meta().zmax, xy, t=t, key=key, band=name, fill=fill, max_dt=max_dt
        )
        if batch.frame is None:
            raise NoData(f"segment {seg}: no such frame{_in_band(name)}")
        return Cutout(
            int(seg),
            x0,
            y0,
            self._assemble(batch, x0, y0, nx, ny, mask),
            batch.frame,
            self.provenance(),
            mask,
        )

    def segment_image(self, seg, *, t=None, key=None, band=None, fill=False):
        """The whole 2048 x 2048 native frame (8 MB on the wire)."""
        side = self.meta().side
        return self.cutout(seg, 0, 0, side, t=t, key=key, band=band, fill=fill)

    def _centered(self, ra, dec, size, seg):
        if seg is None:
            seg, px, py = self._locate1(ra, dec)
        else:
            px, py = sky.radec_to_pixel(seg, ra, dec)
        return seg, int(round(px)) - size // 2, int(round(py)) - size // 2

    def cutout_at(
        self,
        ra,
        dec,
        size=64,
        *,
        t=None,
        key=None,
        band=None,
        fill=False,
        max_dt=None,
        seg=None,
    ):
        """A size x size native cutout centered on (ra, dec), in the segment
        that holds the position deepest inside (or ``seg``). The target is at
        pixel (size // 2, size // 2)."""
        seg, x0, y0 = self._centered(ra, dec, int(size), seg)
        return self.cutout(
            seg, x0, y0, int(size), t=t, key=key, band=band, fill=fill, max_dt=max_dt
        )

    def minipix_cutout(
        self,
        seg,
        idxs,
        *,
        t=None,
        key=None,
        band=None,
        fill=False,
        max_dt=None,
        masked=True,
    ):
        """The window bounding one minipix or a group of them (segment-local
        indices, ``iy * grid + ix``). With ``masked``, pixels outside the
        named minipixes are NaN; the cutout's ``mask`` marks the selection."""
        grid = self.meta().minipix_grid
        x0, y0, nx, ny = sky.minipix_window(idxs, grid)
        mask = sky.minipix_mask(idxs, grid) if masked else None
        return self.cutout(
            seg,
            x0,
            y0,
            nx,
            ny,
            t=t,
            key=key,
            band=band,
            fill=fill,
            max_dt=max_dt,
            mask=mask,
        )

    # ── stacks ─────────────────────────────────────────────────────────

    def stack(
        self,
        seg,
        x0,
        y0,
        nx,
        ny=None,
        *,
        t1=None,
        t2=None,
        band=None,
        fill=False,
        max_frames=500,
        mask=None,
    ):
        """One plane per frame of the segment in [t1, t2]. From a server
        offering ``/stack`` the planes arrive cropped, in one streamed
        response per few thousand frames; otherwise each is fetched by its
        frame key as tiles, ``max_workers`` requests in flight. With ``band=None``
        every camera's frames are included, interleaved by epoch; each
        plane's :class:`FrameInfo` names its camera and band. Raises
        ValueError when the window holds more than ``max_frames`` frames.
        ``fill`` fetches each plane by its epoch within its own band instead
        (gap-filled from other frames), so frames sharing an epoch collapse
        to one."""
        ny = nx if ny is None else ny
        x0, y0, nx, ny = int(x0), int(y0), int(nx), int(ny)
        name = self._band(band)
        listing = self.frames(seg, t1, t2, band=name)
        if len(listing) > max_frames:
            raise ValueError(
                f"{len(listing)} frames in the window, over max_frames={max_frames}"
            )
        if not fill and "stack" in self.meta().raw.get("features", ()):
            self._window_tiles(x0, y0, nx, ny)  # the window must meet the segment
            lo, hi = self._window(t1, t2)
            frames, planes = self._stream_stack(seg, x0, y0, nx, ny, lo, hi, name)
            if mask is not None:
                for plane in planes:
                    plane[~mask] = np.nan
            data = np.stack(planes) if planes else np.empty((0, ny, nx), np.float32)
            return CutoutStack(
                int(seg), x0, y0, data, tuple(frames), self.provenance(), mask
            )
        xy = self._window_tiles(x0, y0, nx, ny)
        zmax = self.meta().zmax

        def fetch(item):
            epoch, key, _config = item
            if fill:
                return self.tiles(seg, zmax, xy, t=epoch, band=name, fill=True)
            return self.tiles(seg, zmax, xy, key=key)

        with ThreadPoolExecutor(max(1, self.max_workers)) as pool:
            batches = list(pool.map(fetch, listing))
        planes, frames, seen = [], [], set()
        for batch in batches:
            frame = batch.frame
            if frame is None or (frame.frame_key, frame.epoch) in seen:
                continue
            seen.add((frame.frame_key, frame.epoch))
            frames.append(frame)
            planes.append(self._assemble(batch, x0, y0, nx, ny, mask))
        order = sorted(
            range(len(frames)),
            key=lambda k: (frames[k].epoch, frames[k].telescope_config),
        )
        data = (
            np.stack([planes[k] for k in order])
            if planes
            else np.empty((0, ny, nx), np.float32)
        )
        return CutoutStack(
            int(seg),
            x0,
            y0,
            data,
            tuple(frames[k] for k in order),
            self.provenance(),
            mask,
        )

    def _stream_stack(self, seg, x0, y0, nx, ny, lo, hi, band):
        """(frames, planes) of every frame in [lo, hi] from the server's
        /stack stream (see :meth:`_stream`)."""
        params = {"x0": x0, "y0": y0, "nx": nx, "ny": ny}
        _head, heads, planes = self._stream(f"stack/{int(seg)}", params, lo, hi, band)
        return [FrameInfo.from_json(h) for h in heads], planes

    def _stream(self, path, params, lo, hi, band):
        """(head, frame records, planes) of every frame in [lo, hi] from a
        framed stream (/stack, /minipix/series), following its pages. A
        stream that drops is resumed from the last frame received (frames
        already in hand are not repeated), within the client's retry
        budget."""
        head, heads, planes, seen = None, [], [], set()
        t1, attempts = lo, self._attempts()
        while True:
            end = None
            try:
                for rec, pixels in self._records(path, params, t1, hi, band):
                    kind = rec["kind"]
                    if kind == "frame":
                        ident = (rec.get("frame_key"), float(rec["epoch"]))
                        if ident not in seen:
                            seen.add(ident)
                            heads.append(rec)
                            planes.append(pixels)
                    elif kind == "end":
                        end = rec
                    elif head is None:
                        head = rec
                if end is None:
                    raise DecodeError(f"{path}: stream ended without its end record")
            except (TransportError, DecodeError, ServerError) as exc:
                if (
                    isinstance(exc, ServerError)
                    and exc.status not in self._RETRY_STATUS
                ):
                    raise
                delay = next(attempts, None)
                if delay is None:
                    raise exc from None
                time.sleep(delay)
                # resume at the last epoch received: frames sharing it may
                # repeat, and `seen` drops them
                t1 = float(heads[-1]["epoch"]) if heads else t1
                continue
            attempts = self._attempts()
            next(attempts)
            if not end.get("truncated") or end.get("next_t1") is None:
                return head, heads, planes
            t1 = float(end["next_t1"])

    def _records(self, path, params, t1, t2, band):
        """The decoded records of one framed-stream request, as they
        arrive."""
        params = dict(params, t1=t1, t2=t2)
        if band:
            params["band"] = band
        try:
            with self._http.stream("GET", path, params=params) as response:
                if not response.is_success:
                    response.read()
                    raise self._http_error(response)
                yield from wire.iter_stack_records(response.iter_bytes())
        except httpx.TransportError as exc:
            raise TransportError(f"{type(exc).__name__} for {path}: {exc}") from None

    def stack_at(
        self,
        ra,
        dec,
        size=32,
        *,
        t1=None,
        t2=None,
        band=None,
        fill=False,
        max_frames=500,
        seg=None,
    ):
        """:meth:`stack` of a size x size window centered on (ra, dec)."""
        seg, x0, y0 = self._centered(ra, dec, int(size), seg)
        return self.stack(
            seg,
            x0,
            y0,
            int(size),
            t1=t1,
            t2=t2,
            band=band,
            fill=fill,
            max_frames=max_frames,
        )

    def minipix_stack(
        self,
        seg,
        idxs,
        *,
        t1=None,
        t2=None,
        band=None,
        fill=False,
        max_frames=500,
        masked=True,
    ):
        """:meth:`stack` over the window bounding one minipix or a group."""
        grid = self.meta().minipix_grid
        x0, y0, nx, ny = sky.minipix_window(idxs, grid)
        mask = sky.minipix_mask(idxs, grid) if masked else None
        return self.stack(
            seg,
            x0,
            y0,
            nx,
            ny,
            t1=t1,
            t2=t2,
            band=band,
            fill=fill,
            max_frames=max_frames,
            mask=mask,
        )

    # ── catalogs and minipixes ─────────────────────────────────────────

    def minipix_series(self, seg, idx, t1=None, t2=None, *, band=None):
        """One minipix's archived cutouts in [t1, t2] (default: all), with
        each frame's data-quality and photometric-calibration records, as a
        :class:`MinipixSeries`. These are the cutouts the pipeline kept for
        an active minipix, from the minipix store; :meth:`minipix_stack`
        instead cuts the same window out of the archived segment frames.
        Raises NotFound when the minipix has no cutouts in the window."""
        if "minipix_series" not in self.meta().raw.get("features", ()):
            raise ExplorerError("this server does not offer /minipix/series")
        lo, hi = self._window(t1, t2)
        head, heads, planes = self._stream(
            f"minipix/series/{int(seg)}/{int(idx)}", {}, lo, hi, self._band(band)
        )
        if head is None:
            raise DecodeError(f"minipix/series/{seg}/{idx}: no head record")
        return MinipixSeries.from_records(head, heads, planes, self.provenance())

    def minipix_series_at(self, ra, dec, t1=None, t2=None, *, band=None):
        """:meth:`minipix_series` of the minipix holding (ra, dec). Where
        segments overlap, the cutouts live in one of them, so each segment
        containing the position is tried in turn. Returns
        ``(series, x, y)``: the series and the position's cutout pixel."""
        grid = self.meta().minipix_grid
        for seg, px, py in sky.segments_containing(ra, dec):
            idx = sky.minipix_index(px, py, grid)
            try:
                series = self.minipix_series(seg, idx, t1, t2, band=band)
            except NotFound:
                continue
            x, y = series.radec_to_pixel(ra, dec)
            return series, float(x), float(y)
        raise NoData(f"no minipix cutouts at ra={ra}, dec={dec}")

    def transients(self):
        """{"events": [...], "trails": [...]} (served empty until the
        transients database backs the route)."""
        return self._json("transients")

    def bogus(self, t1, t2, segs):
        """Bogus detections in [t1, t2] for the given segments, as columns
        {seg, t, ra, dec, rb, kind} of numpy arrays (served empty until the
        transients database backs the route)."""
        reply = self._json(
            "bogus",
            t1=times.to_epoch(t1),
            t2=times.to_epoch(t2),
            segs=",".join(str(int(s)) for s in _iter_ints(segs)),
        )
        return {k: np.asarray(v) for k, v in reply.items()}

    def minipix_active(self, segs, t, window=60.0):
        """{seg: [local minipix index, ...]} active within t +/- window
        seconds. Index = iy * grid + ix (see :func:`argex.sky.minipix_index`).
        The server answers at most 256 segments per call."""
        reply = self._json(
            "minipix/active",
            segs=",".join(str(int(s)) for s in _iter_ints(segs)),
            t=times.to_epoch(t),
            w=float(window),
        )
        return {int(k): list(v) for k, v in reply.items()}

    def minipix_schedule(self, seg, idx):
        """[(start, end), ...] activation intervals (epochs) of one minipix."""
        reply = self._json("minipix/schedule", seg=int(seg), idx=int(idx))
        return [(float(s), float(e)) for s, e in reply.get("intervals", [])]


def _iter_ints(values):
    if isinstance(values, (int, np.integer)):
        return [int(values)]
    return [int(v) for v in values]


def _in_band(name):
    return f" in band {name!r}" if name else ""
