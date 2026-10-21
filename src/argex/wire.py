"""Decoders for the server's binary responses (all little-endian).

/segments.bin   four columns of N values each: u32 seg, u32 n_frames,
                f8 epoch_min, f8 epoch_max
/coverage       u32 count per segment, aligned to /segments.bin
/tile/...       one tile x tile fp16 tile, row-major; the frame epoch (if
                native) in the X-Epoch-Tt header
/tiles/...      f8 epoch, u32 count, then count x (u8 x, u8 y,
                tile x tile fp16); epoch is NaN for pyramid levels and for
                an empty batch
/stack/...      a sequence of records [u32 n][n B UTF-8 JSON][payload]:
                a "stack" record (seg, window, dtype, frames listed), one
                "frame" record per frame (its identity, then ny x nx fp16
                pixels, row-major) and an "end" record (frames sent,
                truncated, next_t1)
"""

import json
import struct

import numpy as np

from .errors import DecodeError

_TILES_HEADER = struct.Struct("<dI")
_XY = struct.Struct("<BB")


def decode_segments(blob):
    """/segments.bin -> dict of equal-length arrays (seg, n_frames,
    epoch_min, epoch_max)."""
    if len(blob) % 24:
        raise DecodeError(f"segments.bin: {len(blob)} B is not a multiple of 24")
    n = len(blob) // 24
    return {
        "seg": np.frombuffer(blob, "<u4", n, 0).astype(np.int64),
        "n_frames": np.frombuffer(blob, "<u4", n, 4 * n).astype(np.int64),
        "epoch_min": np.frombuffer(blob, "<f8", n, 8 * n).copy(),
        "epoch_max": np.frombuffer(blob, "<f8", n, 16 * n).copy(),
    }


def decode_coverage(blob, n_segments=None):
    """/coverage -> int64 count per segment."""
    if len(blob) % 4:
        raise DecodeError(f"coverage: {len(blob)} B is not a multiple of 4")
    counts = np.frombuffer(blob, "<u4").astype(np.int64)
    if n_segments is not None and len(counts) != n_segments:
        raise DecodeError(
            f"coverage: {len(counts)} counts for {n_segments} segments "
            "(the archive was rescanned; refetch segments)"
        )
    return counts


def decode_tile(blob, tile):
    """One fp16 tile -> (tile, tile) float32 array."""
    if len(blob) != tile * tile * 2:
        raise DecodeError(f"tile: {len(blob)} B, expected {tile * tile * 2}")
    return np.frombuffer(blob, "<f2").reshape(tile, tile).astype(np.float32)


def decode_tiles(blob, tile):
    """/tiles batch -> (epoch or None, {(x, y): (tile, tile) float32})."""
    if len(blob) < _TILES_HEADER.size:
        raise DecodeError(f"tiles: {len(blob)} B is shorter than the header")
    epoch, count = _TILES_HEADER.unpack_from(blob, 0)
    tile_bytes = tile * tile * 2
    expected = _TILES_HEADER.size + count * (_XY.size + tile_bytes)
    if len(blob) != expected:
        raise DecodeError(
            f"tiles: {len(blob)} B for {count} tiles, expected {expected}"
        )
    tiles = {}
    off = _TILES_HEADER.size
    for _ in range(count):
        x, y = _XY.unpack_from(blob, off)
        off += _XY.size
        tiles[(x, y)] = (
            np.frombuffer(blob, "<f2", tile * tile, off)
            .reshape(tile, tile)
            .astype(np.float32)
        )
        off += tile_bytes
    return (None if np.isnan(epoch) else float(epoch)), tiles


def encode_tiles(epoch, tiles):
    """Inverse of :func:`decode_tiles` ({(x, y): array} -> bytes); used to
    build test fixtures."""
    parts = [_TILES_HEADER.pack(float("nan") if epoch is None else epoch, len(tiles))]
    for (x, y), arr in tiles.items():
        parts.append(_XY.pack(x, y))
        parts.append(np.ascontiguousarray(arr, dtype="<f2").tobytes())
    return b"".join(parts)


_U32 = struct.Struct("<I")


def iter_stack_records(chunks):
    """Decode a /stack body as it arrives: ``chunks`` is an iterable of
    bytes, and each complete record is yielded as (head, pixels) with
    pixels an (ny, nx) float32 array for a "frame" record, else None.
    Raises DecodeError on a malformed record or a record cut short."""
    buf = bytearray()
    shape = None
    for chunk in chunks:
        buf += chunk
        pos = 0
        while True:
            if len(buf) - pos < 4:
                break
            (n,) = _U32.unpack_from(buf, pos)
            if len(buf) - pos - 4 < n:
                break
            try:
                head = json.loads(bytes(buf[pos + 4 : pos + 4 + n]))
                kind = head["kind"]
            except (ValueError, KeyError, TypeError) as exc:
                raise DecodeError(f"stack: malformed record ({exc})") from None
            size = 0
            if kind == "stack":
                if head.get("dtype") != "<f2":
                    raise DecodeError(f"stack: unsupported dtype {head.get('dtype')!r}")
                shape = (int(head["ny"]), int(head["nx"]))
            elif kind == "frame":
                if shape is None:
                    raise DecodeError("stack: frame record before the stack record")
                size = shape[0] * shape[1] * 2
            if len(buf) - pos - 4 - n < size:
                break
            start = pos + 4 + n
            pixels = None
            if size:
                pixels = (
                    np.frombuffer(bytes(buf[start : start + size]), "<f2")
                    .reshape(shape)
                    .astype(np.float32)
                )
            pos = start + size
            yield head, pixels
        del buf[:pos]
    if buf:
        raise DecodeError(f"stack: stream ended inside a record ({len(buf)} B left)")


def encode_stack(seg, x0, y0, nx, ny, frames, end):
    """Inverse of :func:`iter_stack_records` for a list of (identity dict,
    pixels) and an end record; used to build test fixtures."""

    def record(head, payload=b""):
        body = json.dumps(head).encode()
        return _U32.pack(len(body)) + body + payload

    head = {"kind": "stack", "seg": seg, "x0": x0, "y0": y0, "nx": nx, "ny": ny}
    parts = [record({**head, "dtype": "<f2", "frames": len(frames)})]
    for ident, pixels in frames:
        parts.append(
            record({**ident, "kind": "frame"}, np.asarray(pixels, "<f2").tobytes())
        )
    parts.append(record({"kind": "end", **end}))
    return b"".join(parts)
