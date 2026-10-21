"""Python client for the Argus data explorer tile server.

>>> import argex
>>> with argex.Client("http://localhost:8765/") as ex:
...     cut = ex.cutout_at(50.77, 41.31, 64, t="2026-10-18T09:00", band="rho")
...     cut.writeto("cut.fits")
"""

__version__ = "0.3.0"

from . import sky, times, wire  # noqa: E402
from .client import Client  # noqa: E402
from .errors import (  # noqa: E402
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
from .models import (  # noqa: E402
    Band,
    Cutout,
    CutoutStack,
    FrameInfo,
    FrameList,
    Meta,
    Provenance,
    SegmentTable,
    Tile,
    TileBatch,
)

__all__ = [
    "BadRequest",
    "Band",
    "Client",
    "Cutout",
    "CutoutStack",
    "DecodeError",
    "ExplorerError",
    "FrameInfo",
    "FrameList",
    "HTTPError",
    "Meta",
    "NoData",
    "NotFound",
    "Provenance",
    "SegmentTable",
    "ServerError",
    "Tile",
    "TileBatch",
    "TransportError",
    "UnknownBand",
    "sky",
    "times",
    "wire",
]
