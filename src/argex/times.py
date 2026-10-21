"""Argus epochs, the server's time coordinate, and their conversion through
:class:`astropy.time.Time`.

An Argus epoch is TT seconds since 2025-01-01T00:00:00 TT (ADR-0012), the
instant 2024-12-31T23:58:50.816 UTC. Frame epochs are exposure midpoints
(ADR-0024). Conversions go through astropy Time objects on the TT scale, so
leap seconds, the TT - UTC offset (69.184 s since 2017) and timezone-aware
datetimes are handled by its tables.
"""

from datetime import UTC, datetime

import numpy as np
from astropy.time import Time, TimeDelta

ORIGIN = "2025-01-01T00:00:00"
#: Epoch 0 as an astropy Time (TT).
ORIGIN_TT = Time(ORIGIN, format="isot", scale="tt")
#: MJD (TT) of epoch 0: MJDREF for FITS with TIMESYS = 'TT'.
MJDREF = 60676.0


def as_time(value):
    """Coerce a time given as a datetime (naive means UTC), an ISO-8601
    string (UTC unless it carries a UTC offset), a numpy datetime64 (UTC),
    or an astropy Time (any scale), to an astropy Time."""
    if isinstance(value, Time):
        return value
    if isinstance(value, str):
        text = value.strip()
        try:
            value = datetime.fromisoformat(text)
        except ValueError:
            return Time(text, scale="utc")
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(UTC).replace(tzinfo=None)
        return Time(value, scale="utc")
    if isinstance(value, np.datetime64):
        return Time(value, scale="utc")
    raise TypeError(f"cannot interpret {value!r} as a time")


def to_epoch(value):
    """An Argus epoch (float TT seconds) for ``value``: a number is already
    an epoch and returned unchanged; anything :func:`as_time` accepts is
    converted (an astropy Time array gives an array)."""
    if value is None:
        raise TypeError("a time is required")
    if isinstance(value, bool):
        raise TypeError("bool is not a time")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    delta = (as_time(value).tt - ORIGIN_TT).sec
    return float(delta) if np.ndim(delta) == 0 else np.asarray(delta, np.float64)


def to_time(epoch):
    """Argus epoch(s) -> astropy Time on the TT scale."""
    return ORIGIN_TT + TimeDelta(np.asarray(epoch, np.float64), format="sec")


def to_datetime(epoch):
    """Argus epoch -> timezone-aware UTC datetime (microsecond precision)."""
    return to_time(epoch).utc.to_datetime(timezone=UTC)


def to_iso(epoch, scale="utc", precision=3):
    """Argus epoch(s) -> ISO-8601 string(s) on ``scale`` ("utc" or "tt"),
    without a zone suffix."""
    t = getattr(to_time(epoch), scale)
    t.precision = precision
    return t.isot


def to_mjd(epoch):
    """Argus epoch(s) -> MJD on the TT scale."""
    return MJDREF + np.asarray(epoch, dtype=np.float64) / 86400.0
