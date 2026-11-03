"""python -m wonhee.art list | render | sheet | check | recipe

  list [--family F] [--tag T]            effects with params and tags
  render EFFECT --out a.png [--seed N] [--size WxH] [--palette NAME|hue:200:dark:analogous] [--params JSON] [--source img.png]
  sheet EFFECT --out sheet.png [--seeds 0,1,2,3] [--palettes a,b]   contact sheet with labels
  check IMAGE                            no-focal-point / fill / contrast report
  recipe recipe.json --out a.png         layered recipe (see compose.render_recipe)
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from PIL import Image, ImageDraw

from . import compose, metrics, palette as pal
from .registry import REGISTRY, discover, names


def _size(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def _palette(s: str):
    if s.startswith("hue:"):
        _, hue, pol, har = (s.split(":") + ["dark", "analogous"])[:4]
        return pal.make(float(hue), polarity=pol, harmony=har)
    return pal.named(s)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m wonhee.art")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list"); s.add_argument("--family"); s.add_argument("--tag")
    r = sub.add_parser("render"); r.add_argument("effect"); r.add_argument("--out", required=True)
    r.add_argument("--seed", type=int, default=0); r.add_argument("--size", default="1500x1000")
    r.add_argument("--palette", default="ember"); r.add_argument("--params", default="{}"); r.add_argument("--source")
    sh = sub.add_parser("sheet"); sh.add_argument("effect"); sh.add_argument("--out", required=True)
    sh.add_argument("--seeds", default="0,1,2,3"); sh.add_argument("--palettes", default="ember"); sh.add_argument("--params", default="{}")
    ck = sub.add_parser("check"); ck.add_argument("image")
    rc = sub.add_parser("recipe"); rc.add_argument("file"); rc.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    discover()
    if a.cmd == "list":
        rows = [{"name": n, "family": REGISTRY[n].family, "tags": REGISTRY[n].tags, "params": {k: v[1] for k, v in REGISTRY[n].params.items()}, "doc": REGISTRY[n].doc.split("\n")[0]}
                for n in names(a.family, a.tag)]
        print(json.dumps(rows, indent=1)); return 0
    if a.cmd == "render":
        src = compose.load(a.source, _size(a.size)) if a.source else None
        img = compose.render_effect(a.effect, size=_size(a.size), seed=a.seed, palette=_palette(a.palette), params=json.loads(a.params), source=src)
        compose.save(img, a.out); print(json.dumps(metrics.analyze(img).as_dict())); return 0
    if a.cmd == "sheet":
        tiles, labels = [], []
        for p in a.palettes.split(","):
            for sd in a.seeds.split(","):
                img = compose.render_effect(a.effect, size=(750, 500), seed=int(sd), palette=_palette(p), params=json.loads(a.params))
                rep = metrics.analyze(img)
                tiles.append(Image.fromarray((img * 255).astype(np.uint8)))
                labels.append(f"{a.effect} s{sd} {p}  {'ok' if rep.ok else 'FOCAL' if rep.focal_ratio > 3 or rep.center_bias > 1.6 else 'weak'}")
        cols = min(len(tiles), 4); rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 760 + 10, rows * 540 + 10), (24, 24, 26)); d = ImageDraw.Draw(sheet)
        for i, (t, lb) in enumerate(zip(tiles, labels)):
            x, y = 10 + (i % cols) * 760, 10 + (i // cols) * 540
            sheet.paste(t, (x, y)); d.text((x, y + 504), lb, fill=(230, 230, 230))
        sheet.save(a.out); print(a.out); return 0
    if a.cmd == "check":
        print(json.dumps(metrics.analyze(compose.load(a.image)).as_dict(), indent=1)); return 0
    if a.cmd == "recipe":
        img = compose.render_recipe(json.load(open(a.file))); compose.save(img, a.out); return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
