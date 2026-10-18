"""The spec-sheet poster: a Swiss-grid card with a big title, two small label boxes, a tilted image and a
monospaced paragraph. It's a design-studio format; as a joke format it presents anything mundane like an
exhibited study ("Founder-studies", label "Hardware:", body = an absurdly precise brief for a desk photo).

Our own layout (grid, type and spacing chosen here), drawn with Pillow; fonts are the bundled Noto (OFL).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from wonhee.edit.text import FONTS

SANS = FONTS / "NotoSans-Regular.ttf"
MONO = FONTS / "NotoSansMono-Regular.ttf"


@dataclass
class PosterSpec:
    title: str
    labels: list[tuple[str, str]] = field(default_factory=list)  # up to 2 (label, value) boxes, top right
    body_label: str = "Prompt"
    body: str = ""
    tilt: float = -3.0  # degrees, the image's rotation
    paper: str = "#f3f2ef"
    rule: str = "#efb3a3"  # grid line colour
    ink: str = "#111111"
    body_ink: str = "#8c8c8c"
    size: tuple[int, int] = (1080, 1350)


def _font(path, px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), px)


def _fit_title(d: ImageDraw.ImageDraw, text: str, max_w: int, px: int) -> ImageFont.FreeTypeFont:
    while px > 20:
        f = _font(SANS, px)
        if d.textlength(text, font=f) <= max_w:
            return f
        px -= 4
    return _font(SANS, 20)


def _wrap(d: ImageDraw.ImageDraw, text: str, f, width: int, max_lines: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        nxt = f"{cur} {w}".strip()
        if d.textlength(nxt, font=f) <= width:
            cur = nxt
            continue
        lines.append(cur)
        cur = w
        if len(lines) == max_lines:
            break
    if len(lines) < max_lines and cur:
        lines.append(cur)
    if len(lines) == max_lines and " ".join(lines) != " ".join(words):
        while lines[-1] and d.textlength(lines[-1] + "…", font=f) > width:
            lines[-1] = lines[-1][:-1]
        lines[-1] += "…"
    return lines


def _justify(d: ImageDraw.ImageDraw, xy: tuple[int, int], line: str, f, width: int, fill: str, last: bool) -> None:
    words = line.split()
    if last or len(words) < 2:
        d.text(xy, line, font=f, fill=fill)
        return
    gap = (width - sum(d.textlength(w, font=f) for w in words)) / (len(words) - 1)
    x = xy[0]
    for w in words:
        d.text((x, xy[1]), w, font=f, fill=fill)
        x += d.textlength(w, font=f) + gap


def render(image: Image.Image, spec: PosterSpec) -> Image.Image:
    W, H = spec.size
    u = W / 1080  # every measure below is for a 1080-wide card
    m, g = round(24 * u), round(24 * u)  # margin, gutter
    col = (W - 2 * m - 3 * g) / 4
    xs = [round(m + i * (col + g)) for i in range(4)]
    card = Image.new("RGB", (W, H), spec.paper)
    d = ImageDraw.Draw(card)
    lw = max(1, round(1.5 * u))
    top_h, foot_h = round(240 * u), round(250 * u)
    foot_y = H - m - foot_h
    rows_y0, rows_y1 = m + top_h + g, foot_y - g
    n_rows = max(1, round((rows_y1 - rows_y0 + g) / (240 * u + g)))
    row_h = (rows_y1 - rows_y0 - (n_rows - 1) * g) / n_rows

    def box(x0, y0, x1, y1):
        d.rectangle([x0, y0, x1, y1], outline=spec.rule, width=lw)

    box(xs[0], m, xs[1] + col, m + top_h)  # title
    for i in range(2):
        box(xs[2 + i], m, xs[2 + i] + col, m + top_h)
    for r in range(n_rows):
        y = rows_y0 + r * (row_h + g)
        for x in xs:
            box(x, y, x + col, y + row_h)
    box(xs[0], foot_y, xs[0] + col, H - m)
    # title, labels
    tf = _fit_title(d, spec.title, round(2 * col + g - 16 * u), round(92 * u))
    d.text((xs[0] + round(8 * u), m - round(8 * u)), spec.title, font=tf, fill=spec.ink)
    lf = _font(SANS, round(19 * u))
    for i, (label, value) in enumerate(spec.labels[:2]):
        d.text((xs[2 + i] + round(10 * u), m + round(14 * u)), f"{label}\n{value}", font=lf, fill=spec.ink,
               spacing=round(4 * u))
    d.text((xs[0] + round(10 * u), foot_y + round(4 * u)), spec.body_label, font=_font(SANS, round(36 * u)),
           fill=spec.ink)
    # body: justified mono paragraph across the three right columns
    bf = _font(MONO, round(21 * u))
    bw = round(xs[3] + col - xs[1])
    lines = _wrap(d, spec.body, bf, bw, max_lines=int((foot_h - 8 * u) // (31 * u)))
    for i, line in enumerate(lines):
        _justify(d, (xs[1], foot_y + round(i * 31 * u)), line, bf, bw, spec.body_ink, i == len(lines) - 1)
    # the image: fitted, tilted, softly shadowed, over the grid
    area_w, area_h = round(860 * u), rows_y1 - round(110 * u) - m
    im = image.convert("RGB")
    s = min(area_w / im.width, area_h / im.height)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    rot = im.convert("RGBA").rotate(spec.tilt, expand=True, resample=Image.BICUBIC)
    shadow = Image.new("RGBA", rot.size, (0, 0, 0, 0))
    shadow.putalpha(rot.getchannel("A").point(lambda a: a * 0.22))
    shadow = shadow.filter(ImageFilter.GaussianBlur(round(10 * u)))
    cx, cy = W // 2, round((m + top_h * 0.45 + rows_y1) / 2)
    pos = (cx - rot.width // 2, cy - rot.height // 2)
    card.paste(shadow, (pos[0] + round(6 * u), pos[1] + round(10 * u)), shadow)
    card.paste(rot, pos, rot)
    return card
