"""Exceptions raised by the client."""


class ExplorerError(Exception):
    """Base class for every error the client raises."""


class TransportError(ExplorerError):
    """The server could not be reached, or the connection failed."""


class HTTPError(ExplorerError):
    """The server answered with a non-2xx status."""

    def __init__(self, status, url, body=""):
        self.status = status
        self.url = str(url)
        self.body = body
        detail = f": {body}" if body else ""
        super().__init__(f"HTTP {status} for {self.url}{detail}")


class BadRequest(HTTPError):
    """400: a parameter the server rejected (malformed or non-finite value,
    unknown band, out-of-range zoom). The server's body is a bare
    ``bad request`` and does not say which parameter."""


class NotFound(HTTPError):
    """404: unknown segment, or no such tile."""


class ServerError(HTTPError):
    """5xx from the server or the web edge in front of it."""


class UnknownBand(ExplorerError, ValueError):
    """A band name the server does not serve (checked against /meta)."""


class NoData(ExplorerError, LookupError):
    """No frame or segment with data where one was required."""


class DecodeError(ExplorerError):
    """A binary response did not match its wire format."""
