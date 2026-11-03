"""python -m wonhee.art.grade_cli  list | sheet | describe | nearest | export-web | export-cube | import-cube

  list                                   all looks with mood
  sheet SRC --out sheet.png [--tile 360x240] [--render 750x500] [--looks a,b] [--cols N] [--small]
                                         contact sheet, original first, label under each tile.
                                         --small = 288x192 tiles rendered AT 288x192 (small-size check)
  describe IMG                           vibe vector (JSON)
  nearest IMG [-k 6]                     looks whose effect on a neutral chart is closest to IMG's vibe
  export-web LOOK [--intensity 1]        CSS filter chain + SVG filter snippet and how close they are
  export-cube LOOK out.cube [--intensity 1]   33^3 .cube (video: ffmpeg -vf lut3d=out.cube)
  import-cube FILE SRC --out img.png     apply a .cube to an image
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import compose, grading as G


def _wh(s):
    w, h = s.lower().split("x"); return int(w), int(h)


def _font(n=13):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"):
        try:
            return ImageFont.truetype(p, n)
        except OSError:
            pass
    return ImageFont.load_default()


def contact_sheet(src: np.ndarray, looks: list[str], tile=(360, 240), render=None, cols=None, seed=0, params=None) -> Image.Image:
    """Original + every look on one source; label under each tile."""
    from . import features as F
    tw, th = tile
    render = render or tile
    names = ["original"] + looks
    cols = cols or max(1, min(len(names), 2400 // (tw + 8)))
    rows = (len(names) + cols - 1) // cols
    lab_h = 18
    sheet = Image.new("RGB", (cols * (tw + 8) + 8, rows * (th + lab_h + 8) + 8), (20, 20, 22))
    d = ImageDraw.Draw(sheet); font = _font(12 if tw < 300 else 13)
    for i, n in enumerate(names):
        if n == "original":
            img = F.fit_cover(src, render, "center")
        else:
            img = G.render_look(src, n, render, seed, params)
        im = Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
        if im.size != (tw, th):
            im = im.resize((tw, th), Image.LANCZOS)
        x, y = 8 + (i % cols) * (tw + 8), 8 + (i // cols) * (th + lab_h + 8)
        sheet.paste(im, (x, y)); d.text((x + 2, y + th + 3), n, fill=(225, 225, 225), font=font)
    return sheet


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m wonhee.art.grade_cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    s = sub.add_parser("sheet"); s.add_argument("src"); s.add_argument("--out", required=True); s.add_argument("--tile", default="360x240")
    s.add_argument("--render"); s.add_argument("--looks"); s.add_argument("--cols", type=int); s.add_argument("--small", action="store_true"); s.add_argument("--seed", type=int, default=0)
    d = sub.add_parser("describe"); d.add_argument("image")
    n = sub.add_parser("nearest"); n.add_argument("image"); n.add_argument("-k", type=int, default=6)
    w = sub.add_parser("export-web"); w.add_argument("look"); w.add_argument("--intensity", type=float, default=1.0)
    c = sub.add_parser("export-cube"); c.add_argument("look"); c.add_argument("out"); c.add_argument("--intensity", type=float, default=1.0); c.add_argument("--size", type=int, default=33, help="LUT size (33 standard, 65 for hue-heavy looks)")
    ic = sub.add_parser("import-cube"); ic.add_argument("file"); ic.add_argument("src"); ic.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    looks = G.ensure_loaded()
    if a.cmd == "list":
        for k in sorted(looks):
            print(f"{k:28s} {looks[k].mood}")
        return 0
    if a.cmd == "sheet":
        src = compose.load(a.src)
        names = a.looks.split(",") if a.looks else sorted(looks)
        tile = (288, 192) if a.small else _wh(a.tile)
        render = tile if a.small else (_wh(a.render) if a.render else None)
        contact_sheet(src, names, tile, render, a.cols, a.seed).save(a.out); print(a.out); return 0
    if a.cmd == "describe":
        print(json.dumps(G.describe(compose.load(a.image)), indent=1)); return 0
    if a.cmd == "nearest":
        for name, dist in G.nearest_looks(compose.load(a.image), a.k):
            print(f"{dist:6.2f}  {name:26s} {looks[name].mood}")
        return 0
    if a.cmd == "export-web":
        css, fc = G.css_filter(a.look, a.intensity); svg, fs = G.svg_filter(a.look, a.intensity)
        print(f"/* {a.look}: CSS chain fidelity {fc['grade']} (rms {fc['rms_8bit']}/255) */\n{css}\n")
        print(f"<!-- {a.look}: SVG filter fidelity {fs['grade']} (rms {fs['rms_8bit']}/255, max {fs['max_8bit']}); use: filter: url(#look-{a.look}) -->\n{svg}\n")
        ov = G.css_overlays(a.look)
        if ov:
            print("/* finishing that CSS can approximate */\n" + ov)
        return 0
    if a.cmd == "export-cube":
        G.write_cube(G.look_lut(a.look, None, a.intensity) if a.size == 33 else G.look_lut_n(a.look, a.size, a.intensity), a.out, a.look); print(a.out); return 0
    if a.cmd == "import-cube":
        lut = G.read_cube(a.file); img = G.apply_lut(compose.load(a.src), lut); compose.save(img, a.out); print(a.out); return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
