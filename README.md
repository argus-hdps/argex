# argex: Argus data explorer client

A small Python client for the read API of the Argus data explorer tile
server (`applications/data_explorer` in
[argus-hdps/hdps](https://github.com/argus-hdps/hdps)). It returns numpy arrays, astropy
`Time` and `WCS` objects, and FITS products stamped with each frame's
identity and provenance. Dependencies: `httpx`, `numpy`, `astropy`.

```bash
pip install git+https://github.com/argus-hdps/argex     # or, in a checkout: pip install -e .
```

## Quick start

```python
import argex

with argex.Client("http://localhost:8765/") as ex:      # or https://host/explorer/
    print(ex.meta().band_names)                          # ('beta', 'rho')

    ra, dec = 50.77, 41.31
    frames = ex.frames_at(ra, dec, "2026-10-18T09:00", "2026-10-18T10:00")
    print(len(frames), frames.times[0].isot, frames.configs[:2])

    cut = ex.cutout_at(ra, dec, 64, t="2026-10-18T09:00", band="rho")
    print(cut.frame.frame_key, cut.frame.camera, cut.time.utc.isot)
    cut.writeto("cut.fits")

    # every camera's frames, one plane each; FRAMES table names each plane
    stack = ex.stack_at(ra, dec, 16, t1="2026-10-18T09:00", t2="2026-10-18T09:30")
    rho = stack.select(band="rho")
    print(rho.data.shape, rho.light_curve(8, 8))
    stack.writeto("cube.fits")
```

## Conventions

**Times.** Epochs are Argus epochs: TT seconds since 2025-01-01T00:00:00 TT
(ADR-0012), the instant 2024-12-31T23:58:50.816 UTC. Frame epochs are
exposure midpoints. Time arguments take an epoch (a number) or any datetime
(naive means UTC), ISO-8601 string (UTC unless it has an offset),
`numpy.datetime64` (UTC) or `astropy.time.Time` (any scale), converted
through astropy Time objects on the TT scale, with astropy's leap-second
tables supplying TT - UTC (69.184 s since 2017). Results are epochs;
`argex.times.to_time` and `FrameInfo.time` give astropy `Time` (TT), and
`argex.times.to_iso` / `to_datetime` give UTC strings and datetimes.

**Bands and cameras.** `band=` takes a name from `Client.bands` (`"beta"`
365-520 nm, `"rho"` 565-825 nm on the ADDEM cameras) or a `Band`. `None`
means every camera: frame lists interleave the cameras (two cameras
exposing together share an epoch, ordered by telescope_config) and stacks
hold one plane per frame. Unknown names raise `UnknownBand` before any
query. `/coverage`, `/bogus` and `/transients` take no band.

**Frames.** Every native read reports the frame it came from as a
`FrameInfo`: epoch, `epoch_256`, `telescope_config`, frame key (ADR-0024),
camera and band (the server's mapping), archive format (record magic),
checksum byte, data file and byte offset, and whether the requested time was
outside the segment's range (`clamped`) or pixels were gap-filled (`fill`).
A frame key (`key=`) selects one frame exactly; `max_dt=` refuses a nearest
frame further than that many seconds from `t`.

**Coordinates and WCS.** Segments are 0.5 deg north-up TAN images, 2048 px
square at 0.879 arcsec/px, with ids `dec_band * 1440 + ra_slot`. Pixels are
0-indexed, `image[py, px]`, center at (1023.5, 1023.5); RA grows toward
-px. `argex.sky` holds the geometry (`seg_center`, `segment_of`,
`segments_containing`, `radec_to_pixel`, `pixel_to_radec`, `wcs`,
`minipix_window`, `minipix_mask`, `minipix_wcs`, `minipix_index`). Every
cutout and stack has an astropy `WCS` (`.wcs`). Neighbouring segments overlap
by 28 arcsec; the `*_at` helpers use the segment holding data that contains
the position deepest inside, or the one passed as `seg=`.

**Tiles.** Segments are served as 256 px fp16 tiles: the native level
`z = meta().zmax` (8 x 8 tiles) comes from one frame; levels below it are an
epoch-averaged pyramid with no frame. Pixels without data are NaN. The
client returns float32.

## API

| Method | Returns |
| --- | --- |
| `Client(base_url, *, timeout=30, headers=None, max_workers=4, transport=None, verify=True, retries=3, backoff=0.5)` | context manager, thread-safe |
| `meta(refresh=False)`, `provenance()` | `Meta`, `Provenance` |
| `bands`, `band(name)` | `tuple[Band]`, `Band` |
| `segments(refresh=False)` | `SegmentTable` |
| `coverage(t1, t2)` | `{seg: n_epochs}` (estimated from weekly buckets) |
| `frames(seg, t1=None, t2=None, *, band, limit)` | `FrameList` (epochs, frame_keys, configs) |
| `epochs(seg, t1, t2, *, band, limit)` | sorted float64 epochs |
| `frames_at(ra, dec, t1, t2, *, band, limit)` | `FrameList` of the best segment |
| `nearest(seg, t, *, band, max_dt)` / `nearest_at(ra, dec, t, ...)` | `FrameInfo` or None |
| `frame(seg, *, t=None, key=None, band, max_dt)` | `FrameInfo` or None |
| `locate(ra, dec)` | `[(seg, px, py), ...]` segments with data, deepest first |
| `tile(seg, z, x, y, *, t, key, band, fill, max_dt)` | `Tile` (`.frame`) |
| `tiles(seg, z, xy=None, *, t, key, band, fill, max_dt)` | `TileBatch` (one frame) |
| `overview(seg, z=0, *, band)` | pyramid mosaic, ndarray |
| `cutout(seg, x0, y0, nx, ny=None, *, t, key, band, fill, max_dt, mask)` | `Cutout` |
| `cutout_at(ra, dec, size=64, *, t, key, band, fill, max_dt, seg)` | `Cutout` centered on the target |
| `minipix_cutout(seg, idxs, *, t, key, band, fill, max_dt, masked=True)` | `Cutout` of a minipix or group |
| `segment_image(seg, *, t, key, band, fill)` | full 2048 x 2048 `Cutout` |
| `stack(seg, x0, y0, nx, ny=None, *, t1, t2, band, fill, max_frames=500, mask)` | `CutoutStack` |
| `stack_at(ra, dec, size=32, ...)`, `minipix_stack(seg, idxs, ...)` | `CutoutStack` |
| `transients()`, `bogus(t1, t2, segs)` | catalog dicts (empty until the transients DB backs them) |
| `minipix_active(segs, t, window=60)`, `minipix_schedule(seg, idx)` | `{seg: [idx]}`, `[(start, end)]` |
| `minipix_series(seg, idx, t1=None, t2=None, *, band)` | `MinipixSeries` |
| `minipix_series_at(ra, dec, t1=None, t2=None, *, band)` | `(MinipixSeries, x, y)` |

`Cutout` carries `seg`, `x0`, `y0`, `data`, `frame`, `provenance`, `mask`
and offers `wcs`, `time`, `header()`, `to_hdu()`, `writeto()`,
`pixel_to_radec`, `radec_to_pixel`. `CutoutStack` adds `frames`, `epochs`,
iteration over `Cutout`s, `select(band=, telescope_config=)`,
`light_curve(x, y, radius)` and a FITS writer: the cube plus a `FRAMES` table
with one row per plane.

**Minipix series.** An active minipix keeps a 32 x 32 cutout of every frame
that covers it, in its own store beside the segments, with two per-frame
records: data quality and photometric calibration. `minipix_series` reads
them from the server's `/minipix/series` stream as a `MinipixSeries`, a
`CutoutStack` (pixels in e-/s) with, per frame:

- `dq_or[k, q]`, `n_flagged[k, q]`, `has_dq[k]`: the OR of the
  data-quality flags (`argex.dq.Flags`) and the flagged-pixel count in each
  16 x 16 quadrant `q` (`[SW, SE, NW, NE]`);
- `zp[k, a]`, `grad_x[k]`, `grad_y[k]`, `color[k]`, `has_photocal[k]`: the
  zeropoint at the cutout centre for fluxes in e-/s at g - r = `gr_pivot`
  (0.6), its change across `gradient_px` (32) pixels in x and y, and its
  change per mag of g - r; NaN where the frame was not calibrated.

`zeropoint(x, y, gr)` evaluates that model at a cutout pixel for a source
of color g - r; `dq(x, y, radius)` ORs the flags of the quadrants an
aperture touches; `photometry(x, y, radius, annulus, gr=, exptime=, bad=)`
measures every frame and returns an astropy Table of fluxes, calibrated AB
magnitudes in each frame's band, DQ flags and a `good` mask. The FITS
writer adds the records to the `FRAMES` table. `minipix_stack` is different:
it cuts the same window out of the archived segment frames.

`stack` reads the server's `/stack` stream: every frame in the window,
cropped server-side, in one response per few thousand frames, decoded as
it arrives. Against a server without `/stack`, and with `fill=True`, it
fetches each frame's tiles instead, one request per frame (`max_workers`
at a time). It refuses windows holding more than `max_frames` frames.

**Retries.** A request that fails to connect, drops mid-response, or gets
a 502, 503 or 504 from a web edge is retried up to `retries` times,
waiting `backoff` seconds and doubling. A dropped `/stack` or
`/minipix/series` stream resumes from the last frame received. Every request is a read, so retrying is
safe.

## FITS keywords

| Keyword | Meaning | Source |
| --- | --- | --- |
| WCS (`CTYPEi`, `CRPIXi`, `CRVALi`, `PCi_j`, `CDELTi`, `RADESYS`) | TAN grid of the window | segment geometry |
| `TANSEGID`, `CUTORGX`, `CUTORGY` | segment and window origin | request |
| `FRAMEKEY` | `(TELCONF << 40) \| epoch_256` (ADR-0024) | archive frame header |
| `TELCONF` | telescope_config code (ADR-0026) | archive frame header |
| `ARGE256`, `ARGTTINT`, `ARGTTFRC`, `ARGEPOCH` | frame epoch (midpoint) | archive frame header |
| `ARGFMT`, `ARGCKSUM` | record magic (format version), checksum byte | archive frame record |
| `ARGFILE`, `ARGOFFS` | archive data file and record offset | archive index |
| `CAMID`, `BAND` | camera and band of `TELCONF` | server mapping |
| `ARGCLAMP`, `ARGFILL`, `ARGREQT`, `ARGDT` | how the frame was chosen | server |
| `TIMESYS`, `MJDREF`, `DATEREF`, `TIMEUNIT`, `MJD-AVG`, `DATE-AVG` | midpoint time (TT) | archive frame header |
| `PROCVER` | processing version that wrote the archive | `/meta` (written only when the server knows it) |
| `SRVPROC`, `ARGEXVER`, `DATE` | server PROCVER, client version, file date | server, client |

There is no `DATE-OBS`, `DATE-BEG` or `EXPTIME`: the archive does not record
exposure start or duration, only the midpoint.

## Errors

All exceptions derive from `argex.ExplorerError`: `TransportError` (no
connection), `HTTPError` with subclasses `BadRequest` (400; the message names
the parameter), `NotFound` (404: unknown segment or tile) and `ServerError`
(5xx), plus `UnknownBand`, `NoData` (no frame or no segment with data where
one was needed) and `DecodeError`.

## Tests

```bash
pytest                                                   # unit, fake server
HDPS_REPO=~/hdps pytest tests/test_wire_times_sky.py     # sky scheme against the server's
EXPLORER_URL=http://localhost:8765/ pytest tests/test_live.py
```
