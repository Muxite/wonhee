from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from . import palette as pal
from .registry import Ctx, get


def _palette(spec: Any) -> pal.Palette:
    if isinstance(spec, pal.Palette):
        return spec
    if isinstance(spec, str):
        if spec.startswith("hue:"):   # hue:H[:polarity[:harmony]] e.g. hue:225:dark:analogous
            parts = (spec.split(":") + ["dark", "analogous"])[:4]
            return pal.make(float(parts[1]), polarity=parts[2], harmony=parts[3])
        return pal.named(spec)
    if isinstance(spec, dict):
        return pal.make(float(spec.get("hue", 40)), polarity=spec.get("polarity", "dark"), harmony=spec.get("harmony", "analogous"), three=bool(spec.get("three", False)))
    return pal.named("ember")


def render_effect(name: str, *, size=(1500, 1000), seed: int = 0, palette: Any = "ember", params: dict | None = None,
                  source: np.ndarray | None = None, data: Any = None, with_meta: bool = False):
    """Render one effect to float32 RGB (H, W, 3) in 0..1. An effect may
    return RGB or (RGB, alpha) where alpha is (H, W)."""
    eff = get(name)
    full = {k: v[0] for k, v in eff.params.items()}
    full.update(params or {})
    ctx = Ctx(size=tuple(size), seed=int(seed), palette=_palette(palette), params=full, source=source, data=data, explicit=frozenset(params or {}))
    out = eff.fn(ctx)
    rgb = out[0] if isinstance(out, tuple) else out
    img = np.clip(np.asarray(rgb, np.float32), 0, 1)
    if not with_meta:
        return img
    # Provenance travels with the render. A data-driven effect run without data invented its values:
    # the image is decoration, never a result. A source image is recorded so a grade/edit is traceable.
    meta = {"effect": name, "family": eff.family, "seed": int(seed)}
    if "data" in eff.tags or eff.family in ("data", "multichan"):
        meta["synthetic"] = data is None
        meta["note"] = "decorative synthetic data; not measurements or results" if data is None else "values come from the supplied data"
    meta["has_source"] = source is not None
    meta.update(ctx.meta)
    return img, meta


def _blend(base: np.ndarray, top: np.ndarray, mode: str, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    if mode == "screen":
        mixed = 1 - (1 - base) * (1 - top)
    elif mode == "multiply":
        mixed = base * top
    elif mode == "add":
        mixed = np.clip(base + top, 0, 1)
    else:
        mixed = top
    return base * (1 - a) + mixed * a


def post(img: np.ndarray, ops: list[dict], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    for op in ops or []:
        kind, amt = op.get("op"), float(op.get("amount", 0.3))
        if kind == "bloom":
            thr = float(op.get("threshold", 0.75))
            bright = np.clip(img - thr, 0, 1) / max(1e-6, 1 - thr)
            glow = sum(ndimage.gaussian_filter(bright, (s, s, 0)) for s in (6, 18, 40)) / 3
            img = 1 - (1 - img) * (1 - np.clip(glow * amt * 2.2, 0, 1))
        elif kind == "grain":
            g = ndimage.gaussian_filter(rng.normal(0, 1, img.shape[:2]), 0.6)
            img = img + (g * amt * 0.12)[..., None]
        elif kind == "blur":
            img = ndimage.gaussian_filter(img, (amt * 20, amt * 20, 0))
        elif kind == "vignette":
            H, W = img.shape[:2]
            yy, xx = np.mgrid[0:H, 0:W]
            r = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2))
            img = img * (1 - amt * np.clip(r - 0.6, 0, 1))[..., None]
    return np.clip(img, 0, 1).astype(np.float32)


def render_recipe(recipe: dict, *, source: np.ndarray | None = None, data: Any = None) -> np.ndarray:
    """recipe = {size:[W,H], seed, palette, background:{effect,params,seed},
    layers:[{effect, params, seed, blend, opacity}], post:[{op, amount}]}"""
    size = tuple(recipe.get("size", (1500, 1000)))
    seed = int(recipe.get("seed", 0))
    palette = _palette(recipe.get("palette", "ember"))
    bg = recipe.get("background")
    if bg:
        img = render_effect(bg["effect"], size=size, seed=bg.get("seed", seed), palette=palette, params=bg.get("params"), source=source, data=data)
    else:
        img = np.ones((size[1], size[0], 3), np.float32) * palette.canvas
    for i, layer in enumerate(recipe.get("layers", [])):
        top = render_effect(layer["effect"], size=size, seed=layer.get("seed", seed + i + 1), palette=palette, params=layer.get("params"), source=source, data=data)
        img = _blend(img, top, layer.get("blend", "over"), np.full(img.shape[:2], float(layer.get("opacity", 1.0)), np.float32))
    return post(img, recipe.get("post", []), seed)


def save(img: np.ndarray, path: str) -> None:
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)


def load(path: str, size: tuple[int, int] | None = None) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if size:
        im = im.resize(size, Image.LANCZOS)
    return np.asarray(im, np.float32) / 255
