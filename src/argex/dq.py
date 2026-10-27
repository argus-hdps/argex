"""Data-quality flags of the minipix DQ records.

Each archived minipix cutout has a DQ record: per 16 x 16 quadrant of the
32 x 32 cutout, the bitwise OR of the frame's data-quality flags and the
number of flagged pixels. Quadrants are ``[SW, SE, NW, NE]`` in cutout
pixels, index ``q = (y >= 16) * 2 + (x >= 16)``. The bits are the pipeline's
``MaskFlags`` (``hdps.core.uncertainty``), mirrored here.
"""

from enum import IntFlag

import numpy as np

#: Quadrant side, in cutout pixels.
QUADRANT = 16


class Flags(IntFlag):
    DEAD_PIXEL = 1 << 0  # no response
    WARM_PIXEL = 1 << 1  # elevated but usable dark current
    HOT_PIXEL = 1 << 2  # abnormally high dark current
    HOT_ADJACENT_PIXEL = 1 << 3  # next to a hot pixel
    DARK_NOISE_PIXEL_1 = 1 << 4  # dark-frame variance > 3x expected
    DARK_NOISE_PIXEL_2 = 1 << 5  # > 5x
    DARK_NOISE_PIXEL_3 = 1 << 6  # > 10x
    HIGH_BIAS_PIXEL = 1 << 7  # bias > 5 sigma from median
    READ_NOISE_PIXEL_1 = 1 << 8  # bias-frame variance > 3x expected
    READ_NOISE_PIXEL_2 = 1 << 9  # > 5x
    READ_NOISE_PIXEL_3 = 1 << 10  # > 10x
    HIGH_FLAT_VALUE = 1 << 11  # flat > 5 sigma above local median
    LOW_FLAT_VALUE = 1 << 12  # flat > 5 sigma below local median
    SATURATED = 1 << 13  # above the saturation threshold
    NONLINEAR = 1 << 14  # in the nonlinear regime
    COSMIC_RAY = 1 << 15
    SATELLITE = 1 << 16  # satellite trail
    NO_FLAT = 1 << 17
    NO_DARK = 1 << 18
    BAD_FLAT = 1 << 19  # flat value out of range
    EDGE = 1 << 20  # near the image edge
    VIGNETTED = 1 << 21
    INTERPOLATED = 1 << 22
    SUSPECT = 1 << 23  # flagged by QA, not excluded
    USER1 = 1 << 24
    USER2 = 1 << 25
    NO_REFERENCE = 1 << 26  # no difference-imaging reference here


#: Flags that make a pixel's value unusable for photometry.
BAD = (
    Flags.DEAD_PIXEL
    | Flags.HOT_PIXEL
    | Flags.SATURATED
    | Flags.NONLINEAR
    | Flags.COSMIC_RAY
    | Flags.SATELLITE
    | Flags.BAD_FLAT
)


def names(bits):
    """The flag names set in ``bits`` (an int), lowest bit first."""
    bits = int(bits)
    return [f.name for f in Flags if bits & f]


def quadrants(x, y, radius):
    """Indices of the quadrants a circle of ``radius`` cutout pixels at
    ``(x, y)`` touches, in record order."""
    out = []
    for q in range(4):
        qx0, qy0 = (q % 2) * QUADRANT, (q // 2) * QUADRANT
        # nearest point of the quadrant's pixel area to the centre
        nx = np.clip(x, qx0 - 0.5, qx0 + QUADRANT - 0.5)
        ny = np.clip(y, qy0 - 0.5, qy0 + QUADRANT - 0.5)
        if (nx - x) ** 2 + (ny - y) ** 2 <= radius**2:
            out.append(q)
    return out
