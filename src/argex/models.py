"""Typed results returned by :class:`argex.Client`."""

import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
from astropy.io import fits

from . import dq as dqflags
from . import sky, times


@dataclass(frozen=True)
class Band:
    """A served photometric band: the ``band=`` value, the telescope_config
    codes imaging it and the cameras behind them (where the server knows)."""

    name: str
    label: str
    passband: str
    configs: tuple[int, ...]
    cameras: tuple[str, ...] = ()


@dataclass(frozen=True)
class Meta:
    """/meta: archive summary, serving constants, cameras, provenance."""

    tile: int  # tile side, px
    side: int  # segment side, px
    zmax: int  # native zoom level; levels below it are the static pyramid
    n_segments: int
    n_frames_total: int
    bytes_total: int
    epoch_min: float
    epoch_max: float
    version: str  # changes whenever the coverage table does
    vmin: float  # suggested display cuts
    vmax: float
    bg_level: float
    bands: tuple[Band, ...]
    minipix_grid: int
    writable: bool
    cameras: dict  # {telescope_config: camera id}
    provenance: dict  # server_procver, archive_procver, archive_format, ...
    raw: dict = field(repr=False, compare=False)

    @classmethod
    def from_json(cls, d):
        return cls(
            tile=int(d["tile"]),
            side=int(d["side"]),
            zmax=int(d["zmax"]),
            n_segments=int(d["n_segments"]),
            n_frames_total=int(d["n_frames_total"]),
            bytes_total=int(d.get("bytes_total", 0)),
            epoch_min=float(d["epoch_min"]),
            epoch_max=float(d["epoch_max"]),
            version=str(d["version"]),
            vmin=float(d["vmin"]),
            vmax=float(d["vmax"]),
            bg_level=float(d.get("bg_level", 0.0)),
            bands=tuple(
                Band(
                    b["name"],
                    b["label"],
                    b["passband"],
                    tuple(b["configs"]),
                    tuple(b.get("cameras", ())),
                )
                for b in d.get("bands", ())
            ),
            minipix_grid=int(d.get("minipix_grid", 64)),
            writable=bool(d.get("writable", False)),
            cameras={int(k): v for k, v in d.get("cameras", {}).items()},
            provenance=dict(d.get("provenance", {})),
            raw=d,
        )

    @property
    def band_names(self):
        return tuple(b.name for b in self.bands)

    @property
    def tiles_per_side(self):
        """Native tiles across one segment (8 at the defaults)."""
        return self.side // self.tile


@dataclass(frozen=True)
class FrameInfo:
    """Identity and archive provenance of one archived frame, as the server
    reports it (/frame, /nearest, or the X-Frame-* headers of a tile).

    ``epoch_256`` and ``telescope_config`` are the archive frame record
    header (format ``format``: [magic][tt_int][tt_frac][telescope_config],
    preceded by the ``checksum`` byte); ``frame_key`` packs them per
    ADR-0024. ``camera`` and ``band`` are the server's mapping of the code,
    None when it has none. ``clamped`` means the requested time lay outside
    the segment's epoch range; ``fill`` that no-data pixels were filled from
    other frames (this is the primary frame)."""

    seg: int
    epoch: float
    epoch_256: int
    telescope_config: int
    frame_key: int | None
    camera: str | None
    band: str | None
    format: int
    checksum: int
    data_file: str
    byte_offset: int
    clamped: bool = False
    fill: bool = False
    requested_t: float | None = None
    offset_s: float | None = None

    @property
    def tt_int(self):
        """Integer seconds of the header epoch (the record's tt_int field)."""
        return self.epoch_256 // 256

    @property
    def tt_frac(self):
        """1/256 s fraction of the header epoch (the record's tt_frac byte)."""
        return self.epoch_256 % 256

    @property
    def time(self):
        """The frame epoch (exposure midpoint) as an astropy Time (TT)."""
        return times.to_time(self.epoch)

    @classmethod
    def from_json(cls, d):
        key = d.get("frame_key")
        return cls(
            seg=int(d["seg"]),
            epoch=float(d["epoch"]),
            epoch_256=int(d["epoch_256"]),
            telescope_config=int(d["telescope_config"]),
            frame_key=int(key) if key is not None else None,
            camera=d.get("camera"),
            band=d.get("band"),
            format=int(d["format"]),
            checksum=int(d["checksum"]),
            data_file=str(d["data_file"]),
            byte_offset=int(d["byte_offset"]),
            clamped=bool(d.get("clamped", False)),
            fill=bool(d.get("fill", False)),
            requested_t=_opt_float(d.get("requested_t")),
            offset_s=_opt_float(d.get("offset_s")),
        )

    @classmethod
    def from_headers(cls, seg, headers):
        """From a native tile response's X-Frame-* headers; None when the
        response carries no frame (pyramid tiles, empty batches)."""
        h = {k.lower(): v for k, v in headers.items()}
        if "x-telescope-config" not in h:
            return None
        epoch = float(h["x-epoch"])
        key = h.get("x-frame-key")
        return cls(
            seg=int(seg),
            epoch=epoch,
            epoch_256=int(round(epoch * 256.0)),
            telescope_config=int(h["x-telescope-config"]),
            frame_key=int(key) if key else None,
            camera=h.get("x-camera"),
            band=h.get("x-band"),
            format=int(h["x-archive-format"]),
            checksum=int(h["x-frame-checksum"]),
            data_file=h["x-data-file"],
            byte_offset=int(h["x-byte-offset"]),
            clamped=h.get("x-clamped") == "1",
            fill=h.get("x-fill") == "1",
            requested_t=_opt_float(h.get("x-requested-t")),
            offset_s=_opt_float(h.get("x-epoch-offset")),
        )

    def cards(self):
        """FITS cards [(keyword, value, comment)] of this frame's identity,
        archive provenance and time (TIMESYS = 'TT'; DATE-AVG and
        MJD-AVG are the exposure midpoint)."""
        out = [
            ("TANSEGID", self.seg, "sky segment id"),
            ("FRAMEKEY", self.frame_key, "(TELCONF << 40) | epoch_256, ADR-0024"),
            ("TELCONF", self.telescope_config, "telescope_config code, ADR-0026"),
            ("CAMID", self.camera, "camera behind TELCONF (server mapping)"),
            ("BAND", self.band, "explorer band of TELCONF"),
            ("ARGEPOCH", self.epoch, "frame midpoint, TT s since MJDREF"),
            ("ARGE256", self.epoch_256, "epoch in 1/256 s (frame header)"),
            ("ARGTTINT", self.tt_int, "frame header tt_int"),
            ("ARGTTFRC", self.tt_frac, "frame header tt_frac byte"),
            ("ARGFMT", self.format, "archive frame format (header magic)"),
            ("ARGCKSUM", self.checksum, "archive frame checksum byte"),
            ("ARGFILE", self.data_file, "archive data file"),
            ("ARGOFFS", self.byte_offset, "frame record offset in ARGFILE"),
            ("ARGCLAMP", self.clamped, "requested t was outside the segment"),
            ("ARGFILL", self.fill, "no-data pixels filled from other frames"),
            ("ARGREQT", self.requested_t, "requested time, s since origin"),
            ("ARGDT", self.offset_s, "ARGEPOCH minus ARGREQT, s"),
        ]
        out += time_cards(self.epoch)
        return [c for c in out if c[1] is not None]


def time_cards(epoch):
    """TIMESYS / MJDREF / MJD-AVG / DATE-AVG cards (TT) for a midpoint epoch."""
    return [
        ("TIMESYS", "TT", "time scale of the Argus epoch"),
        ("MJDREF", times.MJDREF, "2025-01-01T00:00:00 TT"),
        ("DATEREF", times.ORIGIN, "Argus epoch origin (TT)"),
        ("TIMEUNIT", "s", "unit of ARGEPOCH"),
        ("MJD-AVG", float(times.to_mjd(epoch)), "exposure midpoint (TT)"),
        ("DATE-AVG", times.to_iso(epoch, "tt"), "exposure midpoint (TT)"),
    ]


@dataclass(frozen=True)
class Provenance:
    """Processing versions behind a product: the archive writer's PROCVER
    (None when unrecorded), the explorer server's, and this client's."""

    archive_procver: str | None
    server_procver: str | None
    client_version: str

    def cards(self):
        out = []
        if self.archive_procver:
            out.append(("PROCVER", self.archive_procver, "pipeline that wrote archive"))
        else:
            out.append(("COMMENT", None, "archive PROCVER not recorded"))
        if self.server_procver:
            out.append(("SRVPROC", self.server_procver, "explorer server PROCVER"))
        out.append(("ARGEXVER", self.client_version, "argex client version"))
        out.append(
            (
                "DATE",
                datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
                "file creation (UTC)",
            )
        )
        return out


@dataclass(frozen=True, eq=False)
class SegmentTable:
    """/segments.bin: every segment holding data, with its frame count and
    epoch range. Columns are aligned numpy arrays."""

    seg: np.ndarray
    n_frames: np.ndarray
    epoch_min: np.ndarray
    epoch_max: np.ndarray
    version: str = ""

    def __post_init__(self):
        object.__setattr__(
            self, "_pos", {int(s): i for i, s in enumerate(self.seg.tolist())}
        )

    def __len__(self):
        return len(self.seg)

    def __contains__(self, seg):
        return int(seg) in self._pos

    def index(self, seg):
        """Row of a segment; KeyError if it holds no data."""
        return self._pos[int(seg)]

    def row(self, seg):
        """{seg, n_frames, epoch_min, epoch_max} for one segment, or None."""
        i = self._pos.get(int(seg))
        if i is None:
            return None
        return {
            "seg": int(self.seg[i]),
            "n_frames": int(self.n_frames[i]),
            "epoch_min": float(self.epoch_min[i]),
            "epoch_max": float(self.epoch_max[i]),
        }

    def centers(self):
        """(ra, dec) arrays of every segment center."""
        return sky.seg_center(self.seg)


@dataclass(frozen=True, eq=False)
class FrameList:
    """A segment's frames in a time window, ascending by epoch (co-pointed
    cameras exposing together share an epoch, ordered by telescope_config)."""

    seg: int
    epochs: np.ndarray  # float64
    frame_keys: np.ndarray  # int64, ADR-0024
    configs: np.ndarray  # int64 telescope_config

    def __len__(self):
        return len(self.epochs)

    def __iter__(self):
        return iter(
            zip(self.epochs.tolist(), self.frame_keys.tolist(), self.configs.tolist())
        )

    @property
    def times(self):
        return times.to_time(self.epochs)


@dataclass(frozen=True, eq=False)
class Tile:
    """One tile; ``frame`` is the native frame it came from, None for the
    epoch-averaged pyramid levels."""

    seg: int
    z: int
    x: int
    y: int
    data: np.ndarray  # (tile, tile) float32, NaN where there is no data
    frame: FrameInfo | None

    @property
    def epoch(self):
        return self.frame.epoch if self.frame is not None else None


@dataclass(frozen=True, eq=False)
class TileBatch:
    """A /tiles response: tiles of one segment and zoom level, all from the
    same frame (native) or the pyramid (``frame`` None)."""

    seg: int
    z: int
    frame: FrameInfo | None
    tiles: dict  # {(x, y): (tile, tile) float32}

    @property
    def epoch(self):
        return self.frame.epoch if self.frame is not None else None

    def __len__(self):
        return len(self.tiles)


def _header(cards):
    header = fits.Header()
    for key, value, comment in cards:
        if key == "COMMENT":
            header.add_comment(comment)
        elif isinstance(value, str) and len(value) > 68:
            header[f"HIERARCH {key}"] = value
        else:
            header[key] = (value, comment)
    return header


@dataclass(frozen=True, eq=False)
class Cutout:
    """A native-resolution image from one frame. The pixel grid is the
    segment's TAN grid shifted to (x0, y0); pixels off the segment or
    without data are NaN. ``mask`` (minipix groups) marks the pixels that
    belong to the selection; the rest are NaN."""

    seg: int
    x0: int  # 0-indexed origin of data[0, 0] in the segment
    y0: int
    data: np.ndarray  # (ny, nx) float32
    frame: FrameInfo
    provenance: Provenance
    mask: np.ndarray | None = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def epoch(self):
        return self.frame.epoch

    @property
    def band(self):
        return self.frame.band

    @property
    def time(self):
        """Frame midpoint as an astropy Time (TT)."""
        return self.frame.time

    @property
    def wcs(self):
        """astropy WCS of the cutout (0-indexed pixels, data[y, x])."""
        return sky.wcs(self.seg, self.x0, self.y0, self.data.shape)

    def pixel_to_radec(self, x, y):
        """Cutout pixel (0-indexed, data[y, x]) -> (ra, dec)."""
        return sky.pixel_to_radec(
            self.seg, np.asarray(x) + self.x0, np.asarray(y) + self.y0
        )

    def radec_to_pixel(self, ra, dec):
        """(ra, dec) -> float cutout pixel (x, y)."""
        px, py = sky.radec_to_pixel(self.seg, ra, dec)
        return px - self.x0, py - self.y0

    def header(self):
        """astropy fits.Header: WCS, frame identity and archive provenance,
        time (TIMESYS = 'TT') and processing versions."""
        return _product_header(
            self.wcs,
            [
                ("CUTORGX", self.x0, "cutout origin x in segment (0-indexed)"),
                ("CUTORGY", self.y0, "cutout origin y in segment (0-indexed)"),
            ]
            + self.frame.cards()
            + self.provenance.cards(),
        )

    def to_hdu(self):
        return fits.PrimaryHDU(self.data, header=self.header())

    def writeto(self, path, overwrite=False):
        """Write a FITS file."""
        self.to_hdu().writeto(path, overwrite=overwrite)


@dataclass(frozen=True, eq=False)
class CutoutStack:
    """The same pixel window over many frames: ``data[k]`` comes from
    ``frames[k]``, ascending by epoch, one plane per distinct frame. With
    ``band=None`` frames of several cameras interleave; each plane's frame
    says which."""

    seg: int
    x0: int
    y0: int
    data: np.ndarray  # (n_frames, ny, nx) float32
    frames: tuple[FrameInfo, ...]
    provenance: Provenance
    mask: np.ndarray | None = None

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, k):
        return Cutout(
            self.seg,
            self.x0,
            self.y0,
            self.data[k],
            self.frames[k],
            self.provenance,
            self.mask,
        )

    def __iter__(self):
        return (self[k] for k in range(len(self)))

    @property
    def epochs(self):
        return np.array([f.epoch for f in self.frames], np.float64)

    @property
    def wcs(self):
        return sky.wcs(self.seg, self.x0, self.y0, self.data.shape[1:])

    def pixel_to_radec(self, x, y):
        """Window pixel (0-indexed, data[k, y, x]) -> (ra, dec)."""
        return sky.pixel_to_radec(
            self.seg, np.asarray(x) + self.x0, np.asarray(y) + self.y0
        )

    def radec_to_pixel(self, ra, dec):
        """(ra, dec) -> float window pixel (x, y)."""
        px, py = sky.radec_to_pixel(self.seg, ra, dec)
        return px - self.x0, py - self.y0

    def select(self, band=None, telescope_config=None):
        """The planes of one band or camera code, as a new stack."""
        keep = [
            k
            for k, f in enumerate(self.frames)
            if (band is None or f.band == band)
            and (telescope_config is None or f.telescope_config == telescope_config)
        ]
        return CutoutStack(
            self.seg,
            self.x0,
            self.y0,
            self.data[keep],
            tuple(self.frames[k] for k in keep),
            self.provenance,
            self.mask,
        )

    def light_curve(self, x, y, radius=0):
        """Per-plane mean over a (2r+1)^2 box at cutout pixel (x, y),
        ignoring NaNs: a quick-look light curve, not photometry."""
        x, y = int(round(x)), int(round(y))
        box = self.data[
            :,
            max(0, y - radius) : y + radius + 1,
            max(0, x - radius) : x + radius + 1,
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN planes
            return np.nanmean(box.reshape(len(box), -1), axis=1)

    def header(self):
        """Primary header: WCS, window, the frame fields shared by every
        plane, the epoch span and processing versions. Per-plane identity
        lives in the FRAMES extension."""
        cards = [
            ("CUTORGX", self.x0, "cutout origin x in segment (0-indexed)"),
            ("CUTORGY", self.y0, "cutout origin y in segment (0-indexed)"),
            ("TANSEGID", self.seg, "sky segment id"),
            ("NFRAMES", len(self), "planes; identities in FRAMES extension"),
        ]
        for key, attr in (
            ("TELCONF", "telescope_config"),
            ("CAMID", "camera"),
            ("BAND", "band"),
        ):
            values = {getattr(f, attr) for f in self.frames}
            if len(values) == 1 and None not in values:
                cards.append((key, values.pop(), "same for every plane"))
            else:
                cards.append(("COMMENT", None, f"{key} varies by plane: see FRAMES"))
        if self.frames:
            cards += [
                ("TIMESYS", "TT", "time scale of the Argus epoch"),
                ("MJDREF", times.MJDREF, "2025-01-01T00:00:00 TT"),
                ("TIMEUNIT", "s", ""),
                ("ARGEPMIN", float(self.epochs.min()), "first frame midpoint, s"),
                ("ARGEPMAX", float(self.epochs.max()), "last frame midpoint, s"),
            ]
        return _product_header(self.wcs, cards + self.provenance.cards())

    def frames_table(self):
        """BinTableHDU "FRAMES": one row per plane with its identity, archive
        provenance and time."""
        fr = self.frames
        epochs = self.epochs
        cols = [
            fits.Column("PLANE", "J", array=np.arange(len(fr))),
            fits.Column("EPOCH", "D", unit="s", array=epochs),
            fits.Column("EPOCH256", "K", array=[f.epoch_256 for f in fr]),
            fits.Column(
                "FRAMEKEY",
                "K",
                array=[f.frame_key if f.frame_key is not None else -1 for f in fr],
            ),
            fits.Column("TELCONF", "J", array=[f.telescope_config for f in fr]),
            fits.Column("CAMID", "24A", array=[f.camera or "" for f in fr]),
            fits.Column("BAND", "8A", array=[f.band or "" for f in fr]),
            fits.Column("MJD_AVG", "D", unit="d", array=times.to_mjd(epochs)),
            fits.Column(
                "DATE_AVG",
                "23A",
                array=np.atleast_1d(times.to_iso(epochs, "tt")) if len(fr) else [],
            ),
            fits.Column(
                "UTC_AVG",
                "23A",
                array=np.atleast_1d(times.to_iso(epochs, "utc")) if len(fr) else [],
            ),
            fits.Column("ARGFMT", "I", array=[f.format for f in fr]),
            fits.Column("ARGCKSUM", "I", array=[f.checksum for f in fr]),
            fits.Column("ARGFILE", "64A", array=[f.data_file for f in fr]),
            fits.Column("ARGOFFS", "K", array=[f.byte_offset for f in fr]),
            fits.Column("FILLED", "L", array=[f.fill for f in fr]),
        ]
        table = fits.BinTableHDU.from_columns(cols, name="FRAMES")
        table.header["TIMESYS"] = ("TT", "scale of EPOCH, MJD_AVG, DATE_AVG")
        table.header["MJDREF"] = times.MJDREF
        return table

    def to_hdulist(self):
        """HDUList: the cube as the primary HDU and the FRAMES table."""
        primary = fits.PrimaryHDU(self.data, header=self.header())
        return fits.HDUList([primary, self.frames_table()])

    def writeto(self, path, overwrite=False):
        self.to_hdulist().writeto(path, overwrite=overwrite)


@dataclass(frozen=True, eq=False)
class MinipixSeries(CutoutStack):
    """One archived minipix over time: the 32 x 32 cutouts the pipeline kept
    for it, one plane per frame, with each frame's data-quality and
    photometric-calibration records.

    Pixels are e-/s. Per frame ``k``:

    - ``dq_or[k, q]``, ``n_flagged[k, q]``: the OR of the data-quality flags
      (:class:`argex.dq.Flags`) and the flagged-pixel count in quadrant
      ``q`` (``[SW, SE, NW, NE]``); ``has_dq[k]`` is False when the frame
      has no record.
    - ``zp[k, a]``: the zeropoint of aperture ``a`` at the minipix centre,
      for fluxes in e-/s at g - r = ``gr_pivot``; ``grad_x[k]``,
      ``grad_y[k]``: its change across ``gradient_px`` pixels along x and
      y; ``color[k]``: its change per mag of g - r. NaN where the record is
      null or the frame has none (``has_photocal[k]`` False): the frame was
      not calibrated, or the minipix centre was off it.

    A calibrated magnitude is ``zeropoint(x, y, gr) - 2.5 log10(flux)``;
    :meth:`photometry` does the whole measurement."""

    idx: int = -1
    dq_or: np.ndarray | None = None  # (n, 4) uint32
    n_flagged: np.ndarray | None = None  # (n, 4) uint16
    has_dq: np.ndarray | None = None  # (n,) bool
    zp: np.ndarray | None = None  # (n, 3) float64
    grad_x: np.ndarray | None = None  # (n,) float64
    grad_y: np.ndarray | None = None
    color: np.ndarray | None = None
    has_photocal: np.ndarray | None = None  # (n,) bool
    gr_pivot: float = 0.6
    gradient_px: int = 32

    @classmethod
    def from_records(cls, head, frames, planes, provenance):
        """From the decoded /minipix/series records: the head record, the
        frame records and their pixel planes."""
        n = len(frames)
        ny, nx = int(head["ny"]), int(head["nx"])
        dq_or = np.zeros((n, 4), np.uint32)
        n_flagged = np.zeros((n, 4), np.uint16)
        has_dq = np.zeros(n, bool)
        zp = np.full((n, 3), np.nan)
        grads = np.full((3, n), np.nan)
        has_cal = np.zeros(n, bool)
        for k, f in enumerate(frames):
            if f.get("dq") is not None:
                has_dq[k] = True
                dq_or[k] = f["dq"]["dq_or"]
                n_flagged[k] = f["dq"]["n_flagged"]
            cal = f.get("photocal")
            if cal is not None:
                has_cal[k] = True
                zp[k] = [np.nan if z is None else z for z in cal["zp"]]
                for row, name in enumerate(("grad_x", "grad_y", "color")):
                    if cal[name] is not None:
                        grads[row, k] = cal[name]
        return cls(
            seg=int(head["seg"]),
            x0=int(head["x0"]),
            y0=int(head["y0"]),
            data=(np.stack(planes) if planes else np.empty((0, ny, nx), np.float32)),
            frames=tuple(FrameInfo.from_json(f) for f in frames),
            provenance=provenance,
            idx=int(head["idx"]),
            dq_or=dq_or,
            n_flagged=n_flagged,
            has_dq=has_dq,
            zp=zp,
            grad_x=grads[0],
            grad_y=grads[1],
            color=grads[2],
            has_photocal=has_cal,
            gr_pivot=float(head.get("gr_pivot", 0.6)),
            gradient_px=int(head.get("gradient_px", nx)),
        )

    def _take(self, keep):
        keep = np.asarray(keep, dtype=int)
        return MinipixSeries(
            self.seg,
            self.x0,
            self.y0,
            self.data[keep],
            tuple(self.frames[k] for k in keep),
            self.provenance,
            self.mask,
            idx=self.idx,
            dq_or=self.dq_or[keep],
            n_flagged=self.n_flagged[keep],
            has_dq=self.has_dq[keep],
            zp=self.zp[keep],
            grad_x=self.grad_x[keep],
            grad_y=self.grad_y[keep],
            color=self.color[keep],
            has_photocal=self.has_photocal[keep],
            gr_pivot=self.gr_pivot,
            gradient_px=self.gradient_px,
        )

    def select(self, band=None, telescope_config=None):
        """The frames of one band or camera code, as a new series."""
        return self._take(
            [
                k
                for k, f in enumerate(self.frames)
                if (band is None or f.band == band)
                and (telescope_config is None or f.telescope_config == telescope_config)
            ]
        )

    @property
    def center(self):
        """The cutout pixel (x, y) the zeropoint is quoted at."""
        ny, nx = self.data.shape[1:]
        return (nx - 1) / 2.0, (ny - 1) / 2.0

    def zeropoint(self, x=None, y=None, gr=None, aperture=0):
        """Each frame's zeropoint (n,) at cutout pixel (x, y) (default: the
        centre) for a source of color ``gr`` (g - r, mag; default: the
        pivot), from the records' first-order model. NaN for frames without
        a calibration."""
        cx, cy = self.center
        x = cx if x is None else x
        y = cy if y is None else y
        out = (
            self.zp[:, aperture]
            + self.grad_x * (x - cx) / self.gradient_px
            + self.grad_y * (y - cy) / self.gradient_px
        )
        if gr is not None:
            out = out + self.color * (gr - self.gr_pivot)
        return out

    def dq(self, x=None, y=None, radius=None):
        """Each frame's DQ flags (n,) uint32: the OR over the quadrants a
        circle of ``radius`` at cutout pixel (x, y) touches, or over the
        whole cutout without one. Frames without a record give 0; see
        ``has_dq``."""
        qs = range(4) if radius is None else dqflags.quadrants(x, y, radius)
        out = np.zeros(len(self), np.uint32)
        for q in qs:
            out |= self.dq_or[:, q]
        return out

    def records(self, x=None, y=None, gr=None):
        """The per-frame DQ and photocal records as an astropy Table, one
        row per frame (in the series' order):

        ``epoch`` (s), ``mjd`` (TT), ``utc``, ``band``, ``telescope_config``,
        ``frame_key``; ``has_dq``, ``dq`` (the OR over all quadrants),
        ``dq_names``, ``dq_or`` and ``n_flagged`` (4 quadrants each,
        ``[SW, SE, NW, NE]``); ``has_photocal``, ``zp`` (aperture 1 at the
        centre, for g - r = ``gr_pivot``), ``grad_x``, ``grad_y``, ``color``.
        Given a cutout pixel (x, y), and optionally a color ``gr``, it adds
        ``zp_at``: the zeropoint evaluated there (:meth:`zeropoint`)."""
        from astropy.table import Table

        epochs = self.epochs
        flags = self.dq()
        cols = {
            "epoch": epochs,
            "mjd": times.to_mjd(epochs) if len(self) else np.empty(0),
            "utc": (np.atleast_1d(times.to_iso(epochs, "utc")) if len(self) else []),
            "band": [f.band or "" for f in self.frames],
            "telescope_config": [f.telescope_config for f in self.frames],
            "frame_key": [
                f.frame_key if f.frame_key is not None else -1 for f in self.frames
            ],
            "has_dq": self.has_dq,
            "dq": flags,
            "dq_names": [",".join(dqflags.names(v)) for v in flags],
            "dq_or": self.dq_or,
            "n_flagged": self.n_flagged,
            "has_photocal": self.has_photocal,
            "zp": self.zp[:, 0],
            "grad_x": self.grad_x,
            "grad_y": self.grad_y,
            "color": self.color,
        }
        if x is not None and y is not None:
            cols["zp_at"] = self.zeropoint(x, y, gr)
        return Table(
            cols,
            meta={
                "seg": self.seg,
                "idx": self.idx,
                "gr_pivot": self.gr_pivot,
                "gradient_px": self.gradient_px,
                "x": x,
                "y": y,
                "gr": gr,
            },
        )

    def photometry(
        self,
        x,
        y,
        radius=4.0,
        annulus=(8.0, 13.0),
        *,
        gr=None,
        exptime=None,
        bad=dqflags.BAD,
    ):
        """Circular-aperture photometry of the source at cutout pixel (x, y)
        in every frame, calibrated with its photocal record. Returns an
        astropy Table, one row per frame:

        ``epoch`` (s), ``mjd`` (TT), ``band``, ``telescope_config``,
        ``frame_key``; ``flux``, ``flux_err``, ``bkg`` (e-/s, the background
        per pixel); ``zp`` (at (x, y) and ``gr``), ``mag``, ``mag_err`` (AB
        in the frame's band); ``dq`` (flags of the quadrants the aperture
        touches) and ``good``: calibrated, a finite positive flux, and none
        of the ``bad`` flags under the aperture.

        The background is the annulus median, its noise 1.4826 x the
        annulus MAD; pixels of the annulus off the cutout are not used.
        ``flux_err`` combines the background noise in the aperture and in
        the background estimate, plus the source's photon noise when the
        frames' ``exptime`` (s) is given. ``mag_err`` is the flux error
        alone: the records carry no zeropoint uncertainty."""
        from astropy.table import Table

        ny, nx = self.data.shape[1:]
        weights = _aperture_weights((ny, nx), x, y, radius)
        yy, xx = np.mgrid[:ny, :nx]
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        ring = (d2 >= annulus[0] ** 2) & (d2 < annulus[1] ** 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN annuli
            sky_px = self.data[:, ring]
            bkg = np.nanmedian(sky_px, axis=1)
            sigma = 1.4826 * np.nanmedian(np.abs(sky_px - bkg[:, None]), axis=1)
            n_sky = np.isfinite(sky_px).sum(axis=1)
            flux = ((self.data - bkg[:, None, None]) * weights).sum(axis=(1, 2))
            area, w2 = weights.sum(), (weights**2).sum()
            var = sigma**2 * (w2 + area**2 * (np.pi / 2) / np.maximum(n_sky, 1))
            if exptime is not None:
                var = var + np.clip(flux, 0, None) / float(exptime)
            flux_err = np.sqrt(var)
            zp = self.zeropoint(x, y, gr)
            mag = zp - 2.5 * np.log10(flux)
            mag_err = 2.5 / np.log(10) * flux_err / flux
        flags = self.dq(x, y, radius + 0.5)
        good = (
            np.isfinite(zp) & np.isfinite(flux) & (flux > 0) & ((flags & int(bad)) == 0)
        )
        epochs = self.epochs
        return Table(
            {
                "epoch": epochs,
                "mjd": times.to_mjd(epochs) if len(self) else np.empty(0),
                "band": [f.band or "" for f in self.frames],
                "telescope_config": [f.telescope_config for f in self.frames],
                "frame_key": [
                    f.frame_key if f.frame_key is not None else -1 for f in self.frames
                ],
                "flux": flux,
                "flux_err": flux_err,
                "bkg": bkg,
                "zp": zp,
                "mag": mag,
                "mag_err": mag_err,
                "dq": flags,
                "good": good,
            },
            meta={
                "seg": self.seg,
                "idx": self.idx,
                "x": float(x),
                "y": float(y),
                "radius": float(radius),
                "annulus": tuple(map(float, annulus)),
                "gr": gr,
            },
        )

    def frames_table(self):
        """The FRAMES table with each frame's DQ and photocal records."""
        table = super().frames_table()
        cols = list(table.columns) + [
            fits.Column("HAS_DQ", "L", array=self.has_dq),
            fits.Column("DQ_OR", "4J", array=self.dq_or.astype(np.int64)),
            fits.Column("DQ_NFLAG", "4I", array=self.n_flagged.astype(np.int16)),
            fits.Column("HAS_CAL", "L", array=self.has_photocal),
            fits.Column("ZP", "3D", unit="mag", array=self.zp),
            fits.Column("ZP_GRADX", "D", unit="mag", array=self.grad_x),
            fits.Column("ZP_GRADY", "D", unit="mag", array=self.grad_y),
            fits.Column("ZP_COLOR", "D", unit="mag/mag", array=self.color),
        ]
        out = fits.BinTableHDU.from_columns(cols, name="FRAMES", header=table.header)
        out.header["MINIPIX"] = (self.idx, "minipix index in the segment")
        out.header["GRPIVOT"] = (self.gr_pivot, "g - r at which ZP is quoted")
        out.header["GRADPX"] = (self.gradient_px, "pixels per ZP_GRADX/Y step")
        return out


def _aperture_weights(shape, x, y, radius, supersample=5):
    """Fraction of each pixel inside a circle, by supersampling."""
    ny, nx = shape
    sub = (np.arange(supersample) + 0.5) / supersample - 0.5
    yy = np.arange(ny)[:, None, None, None] + sub[None, None, :, None]
    xx = np.arange(nx)[None, :, None, None] + sub[None, None, None, :]
    inside = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
    return inside.mean(axis=(2, 3))


def _product_header(wcs, cards):
    """The WCS header (less its placeholder MJDREF) extended with cards."""
    header = wcs.to_header()
    header.remove("MJDREF", ignore_missing=True)
    header.extend(_header(cards), update=True)
    return header


def _opt_float(value):
    return None if value is None else float(value)
