"""Looks: named colour grades / film textures as ffmpeg filter strings, for video and stills.

Each preset is one filter chain, so a future single-pass render can splice it into its graph instead of
re-encoding (docs/production/policies.md §2). `intensity` scales every deviation from neutral: 0 = untouched,
1 = the preset as designed, 2 = double.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

Preset = Literal["moody", "grain", "moody_grain", "digicam", "faded", "bw_grain", "warm_pink", "cool_night", "lofi_cam", "none"]

MONO_FONTS = ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")


def _f(x: float) -> str:
    return f"{x:.4f}".rstrip("0").rstrip(".") or "0"


def _moody(k: float) -> list[str]:
    # underexposed, low-saturation, cool shadows, warm-ish highlights, soft vignette
    return [f"eq=brightness={_f(-0.07 * k)}:contrast={_f(1 + 0.10 * k)}:saturation={_f(1 - 0.28 * k)}"
            f":gamma={_f(1 - 0.10 * k)}",
            f"colorbalance=rs={_f(-0.03 * k)}:bs={_f(0.05 * k)}:rh={_f(0.03 * k)}:bh={_f(-0.02 * k)}",
            "vignette=angle=PI/5"]


def _digicam(k: float) -> list[str]:
    # 2010s point-and-shoot: punchy contrast, slightly hot and cool whites, crunchy sharpening
    return [f"eq=contrast={_f(1 + 0.18 * k)}:brightness={_f(0.025 * k)}:saturation={_f(1 + 0.12 * k)}",
            f"colorbalance=bh={_f(0.04 * k)}:bm={_f(0.02 * k)}",
            f"unsharp=5:5:{_f(0.9 * k)}"]


def _faded(k: float) -> list[str]:
    # lifted blacks, rolled-off whites, muted colour
    return [f"curves=all='0/{_f(0.07 * k)} 1/{_f(1 - 0.05 * k)}'", f"eq=saturation={_f(1 - 0.15 * k)}"]


def _bw(k: float) -> list[str]:
    return ["hue=s=0", f"eq=contrast={_f(1 + 0.15 * k)}"]


def _warm_pink(k: float) -> list[str]:
    # the "Ottawa, Japan" grade: warm, pink highlights, a touch less green
    return [f"eq=saturation={_f(1 + 0.10 * k)}:gamma_r={_f(1 + 0.05 * k)}:gamma_b={_f(1 - 0.03 * k)}",
            f"colorbalance=rh={_f(0.06 * k)}:bh={_f(0.04 * k)}:gh={_f(-0.03 * k)}"]


def _cool_night(k: float) -> list[str]:
    # teal shadows, warm highlights, slightly darker: neon / night-city footage
    return [f"eq=brightness={_f(-0.03 * k)}:contrast={_f(1 + 0.08 * k)}:saturation={_f(1 + 0.05 * k)}",
            f"colorbalance=bs={_f(0.08 * k)}:gs={_f(0.02 * k)}:rh={_f(0.04 * k)}"]


def _lofi_cam(k: float) -> list[str]:
    # phone-cam lo-fi (reel study c48, docs/specs/2026-10-20-reel-styles-design.md §5): muted colour (~0.55 saturation),
    # a slight blur, a magenta-tinted lift in the shadows and a strong vignette (PI/4.2 at intensity 1: edge/centre luminance 0.46 on a flat frame, the c48 value); the preset's grain is added after
    return [f"eq=saturation={_f(1 - 0.45 * k)}:contrast={_f(1 - 0.04 * k)}",
            f"gblur=sigma={_f(0.8 * k)}:steps=1",
            f"curves=all='0/{_f(0.05 * k)} 1/1'",
            f"colorbalance=rs={_f(0.06 * k)}:gs={_f(-0.03 * k)}:bs={_f(0.04 * k)}",
            f"vignette=angle=PI/{_f(1 + 3.2 * k)}"]


# preset -> (grade, default grain strength at intensity 1)
PRESETS: dict[str, tuple[Callable[[float], list[str]], int]] = {
    "moody": (_moody, 0),
    "grain": (lambda k: [], 24),
    "moody_grain": (_moody, 20),
    "digicam": (_digicam, 6),
    "faded": (_faded, 8),
    "bw_grain": (_bw, 26),
    "warm_pink": (_warm_pink, 0),
    "cool_night": (_cool_night, 0),
    "lofi_cam": (_lofi_cam, 8),
    "none": (lambda k: [], 0),
}


def filter_chain(preset: str, intensity: float = 1.0, *, video: bool, grain: int | None = None,
                 datestamp_file: Path | None = None, height: int | None = None) -> str:
    """The ffmpeg -vf chain for a look. `grain` overrides the preset's grain (0 turns it off).

    Grain is luma-only noise; temporal (changes per frame) for video, fixed for stills.
    `datestamp_file` holds the stamp text (a file, so no filtergraph escaping); `height` sizes it.
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown look '{preset}'; known: {', '.join(PRESETS)}")
    grade, g0 = PRESETS[preset]
    parts = grade(intensity) if intensity > 0 else []
    g = round(g0 * intensity) if grain is None else grain
    if g > 0:
        parts.append(f"noise=c0s={min(g, 100)}" + (":c0f=t" if video else ""))
    if datestamp_file is not None:
        parts.append(datestamp_filter(datestamp_file, height or 1080))
    parts.append("format=yuv420p" if video else "format=yuvj444p")
    return ",".join(parts)


def datestamp_filter(text_file: Path, height: int) -> str:
    """Orange LCD-style date in the bottom-right corner, like a 2000s digicam."""
    font = next((f for f in MONO_FONTS if Path(f).exists()), None)
    size = max(12, round(height * 0.032))
    src = f"fontfile={font}" if font else "font=Monospace"
    return (f"drawtext={src}:textfile={text_file}:fontsize={size}:fontcolor=0xFF9A2E"
            f":borderw={max(1, size // 14)}:bordercolor=0xC23A00@0.55"
            f":x=w-tw-{round(height * 0.04)}:y=h-th-{round(height * 0.05)}")
