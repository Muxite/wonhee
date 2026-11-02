"""Shared ffmpeg expression helpers (geq / xfade / any expr). Return strings; quote with '...' at use site."""


def hash2(a: str, b: str) -> str:
    """Pseudo-random 0..1 from two expressions (sin hash). Pure, so safe to nest and repeat."""
    return f"abs(mod(sin(({a})*12.9898+({b})*78.233)*43758.5453,1))"


def rgb_geq(r: str, g: str, b: str) -> str:
    """geq on planar RGB (0..255 per channel); output is gbrp, add format=yuv420p/rgba as needed."""
    return f"format=gbrp,geq=r='{r}':g='{g}':b='{b}'"
