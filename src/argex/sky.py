"""Sky segmentation geometry: segment ids, centers, and the per-segment TAN
pixel grid.

A numpy-only port of the server's ``sky_scheme`` at the all-sky defaults.
Segments are 0.5 deg north-up gnomonic (TAN) images, 2048 px square at
900/1024 arcsec/px, laid out in declination bands of height SPACING_DEG
with RA spacing widened by 1/cos(dec). Neighbouring segments overlap by
OVERLAP_ARCSEC, so a sky position near an edge lies in more than one
segment. Ids are ``dec_band * MAX_RA_SLOTS + ra_slot``.

Pixel convention: numpy 0-indexed, ``image[py, px]``. The segment center
(CRVAL) sits at px = py = 1023.5 (FITS CRPIX 1024.5); RA increases toward
-px, Dec toward +py.
"""

import numpy as np

TILE_SIZE_ARCSEC = 1800.0
OVERLAP_ARCSEC = 28.125
PIXEL_SCALE_ARCSEC = 900.0 / 1024.0
SPACING_DEG = (TILE_SIZE_ARCSEC - OVERLAP_ARCSEC) / 3600.0  # 0.4921875
TILE_SIZE_DEG = TILE_SIZE_ARCSEC / 3600.0
PIXEL_SCALE_DEG = PIXEL_SCALE_ARCSEC / 3600.0
NPIX = 2048
DEC_MIN = -90.0
POLE_COS_CUT = 0.013  # bands within ~0.7 deg of a pole have no segments
MAX_RA_SLOTS = 1440
CENTER_PX = NPIX / 2.0 - 0.5  # 1023.5


def band_center_dec(band):
    """Declination of a dec band's segment centers."""
    return DEC_MIN + (np.asarray(band) + 0.5) * SPACING_DEG


def _ra_spacing(dec):
    return SPACING_DEG / np.maximum(np.cos(np.radians(dec)), 1e-12)


def slots_in_band(band):
    """Number of RA slots in a dec band; 0 for the empty pole bands."""
    dec = band_center_dec(band)
    n = np.where(
        np.cos(np.radians(dec)) < POLE_COS_CUT,
        0,
        np.maximum(1, np.ceil(360.0 / _ra_spacing(dec))),
    )
    return n.astype(np.int64) if np.ndim(band) else int(n)


def seg_center(seg):
    """(ra, dec) in degrees of segment center(s). The final RA slot of a band
    is a runt clipped at 360 deg; its center stays on its own side of the
    seam. Scalar or array input."""
    seg = np.asarray(seg, dtype=np.int64)
    dec = band_center_dec(seg // MAX_RA_SLOTS)
    spacing = _ra_spacing(dec)
    lo = (seg % MAX_RA_SLOTS) * spacing
    ra = (0.5 * (lo + np.minimum(lo + spacing, 360.0))) % 360.0
    if seg.ndim == 0:
        return float(ra), float(dec)
    return ra, dec


def segment_of(ra, dec):
    """The segment whose RA/Dec slot owns (ra, dec), or None at the poles.

    This is the slot that tiles the position without overlap; the position
    may also lie in the overlap margin of a neighbour (see
    :func:`segments_containing`)."""
    band = int(np.floor((float(dec) - DEC_MIN) / SPACING_DEG))
    center_dec = float(band_center_dec(band))
    if np.cos(np.radians(center_dec)) < POLE_COS_CUT:
        return None
    spacing = float(_ra_spacing(center_dec))
    n_slots = int(np.ceil(360.0 / spacing))
    slot = int(np.floor((float(ra) % 360.0) / spacing)) % n_slots
    return band * MAX_RA_SLOTS + slot


def segments_in_box(ra0, ra1, dec0, dec1):
    """Sorted ids of segments whose footprint intersects an RA/Dec box (deg).
    ``ra0 > ra1`` means the box wraps through RA 0."""
    if ra0 > ra1:
        return sorted(_box_ids(ra0, 360.0, dec0, dec1) | _box_ids(0.0, ra1, dec0, dec1))
    return sorted(_box_ids(ra0, ra1, dec0, dec1))


def _box_ids(r0, r1, d0, d1):
    half = TILE_SIZE_DEG / 2.0
    sp = SPACING_DEG
    j_lo = max(0, int(np.floor((d0 - half - DEC_MIN) / sp - 0.5)))
    j_hi = min(
        int(np.ceil((d1 + half - DEC_MIN) / sp - 0.5)), int(np.ceil(180.0 / sp)) - 1
    )
    out = set()
    for j in range(j_lo, j_hi + 1):
        dec_c = float(band_center_dec(j))
        cos_dec = np.cos(np.radians(dec_c))
        if cos_dec < POLE_COS_CUT:
            continue
        n_ra = slots_in_band(j)
        ra_sp = sp / cos_dec
        half_ra = half / cos_dec
        i_lo = int(np.floor((r0 - half_ra) / ra_sp - 0.5))
        i_hi = int(np.ceil((r1 + half_ra) / ra_sp - 0.5))
        if i_hi - i_lo + 1 >= n_ra:
            slots = range(n_ra)
        else:
            slots = {i % n_ra for i in range(i_lo, i_hi + 1)}
        for slot in slots:
            ra_c, _ = seg_center(j * MAX_RA_SLOTS + slot)
            if (
                ra_c + half_ra < r0
                or ra_c - half_ra > r1
                or dec_c + half < d0
                or dec_c - half > d1
            ):
                continue
            out.add(j * MAX_RA_SLOTS + slot)
    return out


def _gnomonic_forward(ra_c, dec_c, ra, dec):
    dec_r = np.radians(np.asarray(dec, dtype=np.float64))
    c_r = np.radians(dec_c)
    dra = np.radians(np.asarray(ra, dtype=np.float64) - ra_c)
    denom = np.sin(c_r) * np.sin(dec_r) + np.cos(c_r) * np.cos(dec_r) * np.cos(dra)
    xi = np.cos(dec_r) * np.sin(dra) / denom
    eta = (
        np.cos(c_r) * np.sin(dec_r) - np.sin(c_r) * np.cos(dec_r) * np.cos(dra)
    ) / denom
    return np.degrees(xi), np.degrees(eta)


def _gnomonic_inverse(ra_c, dec_c, xi_deg, eta_deg):
    xi = np.radians(np.asarray(xi_deg, dtype=np.float64))
    eta = np.radians(np.asarray(eta_deg, dtype=np.float64))
    c_r = np.radians(dec_c)
    rho = np.hypot(xi, eta)
    c = np.arctan(rho)
    with np.errstate(divide="ignore", invalid="ignore"):
        dec = np.arcsin(np.cos(c) * np.sin(c_r) + eta * np.sin(c) * np.cos(c_r) / rho)
        ra = np.radians(ra_c) + np.arctan2(
            xi * np.sin(c),
            rho * np.cos(c_r) * np.cos(c) - eta * np.sin(c_r) * np.sin(c),
        )
    dec = np.where(rho < 1e-12, c_r, dec)
    ra = np.where(rho < 1e-12, np.radians(ra_c), ra)
    return np.degrees(ra) % 360.0, np.degrees(dec)


def radec_to_pixel(seg, ra, dec):
    """Sky -> float 0-indexed pixel (px, py) in a segment (``image[py, px]``).
    Values outside [-0.5, NPIX - 0.5) are off the segment."""
    ra_c, dec_c = seg_center(seg)
    xi, eta = _gnomonic_forward(ra_c, dec_c, ra, dec)
    px = -xi / PIXEL_SCALE_DEG + CENTER_PX
    py = eta / PIXEL_SCALE_DEG + CENTER_PX
    if np.ndim(px) == 0:
        return float(px), float(py)
    return px, py


def pixel_to_radec(seg, px, py):
    """Float 0-indexed segment pixel -> sky (ra, dec) in degrees."""
    ra_c, dec_c = seg_center(seg)
    xi = -(np.asarray(px, dtype=np.float64) - CENTER_PX) * PIXEL_SCALE_DEG
    eta = (np.asarray(py, dtype=np.float64) - CENTER_PX) * PIXEL_SCALE_DEG
    ra, dec = _gnomonic_inverse(ra_c, dec_c, xi, eta)
    if np.ndim(ra) == 0:
        return float(ra), float(dec)
    return ra, dec


def edge_margin(px, py):
    """Pixels from (px, py) to the nearest segment edge; negative when off
    the segment."""
    lo, hi = -0.5, NPIX - 0.5
    return min(px - lo, hi - px, py - lo, hi - py)


def segments_containing(ra, dec):
    """[(seg, px, py), ...] for every segment whose pixel grid contains
    (ra, dec), deepest inside first (largest :func:`edge_margin`)."""
    pad = 1e-6
    cos_dec = max(np.cos(np.radians(dec)), 1e-12)
    dra = pad / cos_dec
    hits = []
    for seg in segments_in_box(
        (ra - dra) % 360.0, (ra + dra) % 360.0, dec - pad, dec + pad
    ):
        px, py = radec_to_pixel(seg, ra, dec)
        if edge_margin(px, py) > 0:
            hits.append((seg, px, py))
    hits.sort(key=lambda h: -edge_margin(h[1], h[2]))
    return hits


def wcs_header(seg, x0=0, y0=0):
    """FITS WCS cards (a dict) for a segment image or a cutout whose
    0-indexed pixel origin in the segment is (x0, y0)."""
    ra_c, dec_c = seg_center(seg)
    return {
        "WCSAXES": 2,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CRPIX1": CENTER_PX + 1.0 - x0,
        "CRPIX2": CENTER_PX + 1.0 - y0,
        "CRVAL1": ra_c,
        "CRVAL2": dec_c,
        "CD1_1": -PIXEL_SCALE_DEG,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": PIXEL_SCALE_DEG,
        "RADESYS": "ICRS",
    }


def minipix_index(px, py, grid=64):
    """Segment-local minipix index of a pixel: ``iy * grid + ix`` on the
    grid x grid minipix layout (32 px cells at the default 64)."""
    cell = NPIX // grid
    return int(py // cell) * grid + int(px // cell)


def wcs(seg, x0=0, y0=0, shape=None):
    """astropy WCS of a segment image, or of a window whose 0-indexed pixel
    origin in the segment is (x0, y0); ``shape`` (ny, nx) sets its pixel
    shape. Pixel coordinates follow astropy's 0-indexed convention, so
    ``wcs(seg).pixel_to_world(px, py)`` agrees with :func:`pixel_to_radec`."""
    from astropy.wcs import WCS

    w = WCS(wcs_header(seg, x0, y0))
    if shape is not None:
        w.pixel_shape = (int(shape[1]), int(shape[0]))
    return w


def minipix_origin(idx, grid=64):
    """0-indexed pixel origin (x0, y0) of a minipix in its segment."""
    cell = NPIX // grid
    idx = int(idx)
    if not 0 <= idx < grid * grid:
        raise ValueError(f"minipix index must be in 0..{grid * grid - 1}")
    return (idx % grid) * cell, (idx // grid) * cell


def minipix_window(idxs, grid=64):
    """(x0, y0, nx, ny): the pixel window bounding one minipix index or a
    group of them in one segment."""
    idxs = [int(idxs)] if np.ndim(idxs) == 0 else [int(i) for i in idxs]
    if not idxs:
        raise ValueError("no minipix indices")
    cell = NPIX // grid
    origins = np.array([minipix_origin(i, grid) for i in idxs])
    x0, y0 = origins.min(axis=0)
    x1, y1 = origins.max(axis=0) + cell
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def minipix_mask(idxs, grid=64):
    """Boolean mask over :func:`minipix_window` that is True on the named
    minipixes (a group need not fill its bounding window)."""
    x0, y0, nx, ny = minipix_window(idxs, grid)
    cell = NPIX // grid
    mask = np.zeros((ny, nx), bool)
    for idx in [int(idxs)] if np.ndim(idxs) == 0 else idxs:
        mx, my = minipix_origin(idx, grid)
        mask[my - y0 : my - y0 + cell, mx - x0 : mx - x0 + cell] = True
    return mask


def minipix_wcs(seg, idxs, grid=64):
    """astropy WCS of the window bounding one minipix or a group of them."""
    x0, y0, nx, ny = minipix_window(idxs, grid)
    return wcs(seg, x0, y0, (ny, nx))
