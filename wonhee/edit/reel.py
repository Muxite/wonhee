"""compose_video's engine: a layout over time, planned as a few PNG states and one ffmpeg graph.

Design: docs/specs/2026-10-20-reel-styles-design.md §2 (timed layers, `anim`, the word-driven kinds `captions`,
`stack` and `list`, insets) and §3 (`tighten`). The layout's canvas is the video's duration. Static layers are drawn
once; each timed layer becomes PNG *states* that change only at its own event times (a word, an entry, an animation
step), so a 60 s reel with 120 caption changes is about 120 small PNGs, not 1,800 frames. The states overlay the clip
with `enable='gte(t,a)*lt(t,b)'` in one filter graph, so there is one encode.

Pure functions: no DB, no subprocess. `plan_reel` makes the states, `graph_for` the ffmpeg arguments.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from PIL import Image, ImageDraw

from wonhee.edit import compose as C
from wonhee.edit import text as T
from wonhee.edit import timeline as TL
from wonhee.nodes.subtitles import Word

MAX_OVERLAYS = 1500  # measured: ~0.03 s a state; the encode dominates (spec §2)
HOLD_MS = 300  # a caption chunk lingers this long after its last word (unless the next chunk starts sooner)
DEFAULT_SIZE = {"captions": 0.045, "stack": 0.075, "list": 0.04}


@dataclass
class Overlay:
    """One PNG state: where it sits and when it shows on the OUTPUT timeline (end None = to the end)."""
    img: Image.Image
    x: int
    y: int
    start_ms: float = 0
    end_ms: float | None = None
    layer: int = -1  # the layout layer it came from (-1 = a static group)


@dataclass
class Plan:
    overlays: list[Overlay] = field(default_factory=list)
    window: tuple[int, int, int, int] | None = None  # the clip's window in px; None = the clip fills the canvas
    pad_colour: str = "black"  # what shows round the clip inside its window (a flat canvas bg)
    clip_fit: str = "contain"
    notes: C.Notes = field(default_factory=C.Notes)
    duration_ms: int = 0  # of the output


def _crop(img: Image.Image, x: int, y: int) -> tuple[Image.Image, int, int] | None:
    bb = img.getbbox()
    if bb is None:
        return None
    return img.crop(bb), x + bb[0], y + bb[1]


def _map_words(words: list[Word], tl: TL.Timeline) -> list[Word]:
    out = []
    for w in words:
        a, b = round(tl.map(w.start_ms)), round(tl.map(w.end_ms))
        out.append(Word(start_ms=a, end_ms=max(a, b), text=w.text))
    return out


def _mark_key(text: str) -> str:
    """A word as `marks` names it: lowercase, letters and digits only."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _case(text: str, case: str) -> str:
    return text.lower() if case == "lower" else text.upper() if case == "upper" else text


# ---- word images -------------------------------------------------------------------------------------------------

def word_image(text: str, spec: T.TextSpec, ref_h: int) -> Image.Image:
    """One word (or entry) as a single-line RGBA image, cropped to its ink horizontally. All words of a spec share the
    line height and baseline, so they align when set side by side."""
    lines, _ = T._render_lines(replace(spec, content=text, box="none", width_frac=1000.0), (10 ** 6, ref_h))
    img = lines[0] if len(lines) == 1 else _hstack(lines)
    bb = img.getchannel("A").getbbox()
    return img if bb is None else img.crop((bb[0], 0, bb[2], img.height))


def _hstack(parts: list[Image.Image]) -> Image.Image:
    out = Image.new("RGBA", (sum(p.width for p in parts), max(p.height for p in parts)), (0, 0, 0, 0))
    x = 0
    for p in parts:
        out.alpha_composite(p, (x, 0))
        x += p.width
    return out


def _rrect(im: Image.Image, rect, fill, radius) -> None:
    ImageDraw.Draw(im).rounded_rectangle(rect, radius=radius, fill=fill)


def _px(spec: T.TextSpec, ref_h: int, default: float) -> int:
    return max(10, round(ref_h * (spec.size_frac or default)))


# ---- captions ----------------------------------------------------------------------------------------------------

def caption_image(texts: list[str], shown: int, active: int | None, ly: C.Layer, c: C.Ctx, size: tuple[int, int],
                  frac: float) -> tuple[Image.Image, int, int]:
    """A chunk as (image, dx, dy): the words laid out as if all were shown (so none move as words appear), only the
    first `shown` drawn; the `active` word (karaoke) sits on a coloured pill. The backing (box, pill or text.py's ragged
    paper, from bg_box) wraps the row and may be taller or wider than the layer: the image grows, never clips. (dx, dy)
    is where the image's top-left sits relative to the layer's box origin."""
    bw, bh = size
    spec = replace(C.text_spec(ly, c.pal, "", size_frac=frac), box="none")
    px = max(10, round(c.H * frac))
    gap = round(px * 0.28)
    imgs = [word_image(t, spec, c.H) for t in texts]
    act_spec = replace(spec, colours=(("#%02x%02x%02x" % C.colour(ly.active_colour, c.pal)[:3],) + tuple(spec.colours or ())[1:])
                       ) if ly.active_colour else None
    total = sum(i.width for i in imgs) + gap * (len(imgs) - 1)
    lh = imgs[0].height if imgs else 0
    hp = round(px * 0.16)
    row = Image.new("RGBA", (total + 2 * hp, lh), (0, 0, 0, 0))
    x = hp
    for k, im in enumerate(imgs):
        if k < shown:
            if ly.mode == "karaoke" and k == active:
                rect = (x - hp, round(lh * 0.1), x + im.width + hp, lh - round(lh * 0.06))
                _rrect(row, rect, C.colour(ly.highlight or "pal:1", c.pal), (rect[3] - rect[1]) / 2)
                if act_spec is not None:
                    im = word_image(texts[k], act_spec, c.H)
            row.alpha_composite(im, (x, 0))
        x += im.width + gap
    colour = "#%02x%02x%02x%02x" % C.colour(ly.bg_colour or "#000000b0", c.pal)
    if ly.bg_box == "paper":
        fonts = T._Fonts(spec.preset, px, "x", spec.role, spec.tracking_em)
        row = T._boxed(row, replace(spec, box="paper", box_colour=colour, box_pad_em=ly.bg_pad_em, content=" ".join(texts)),
                       px, round(px * 1.02) + T._style(spec, px).pad, fonts.base)
    elif ly.bg_box != "none":
        px_, py_ = round(ly.bg_pad_em * px), round(ly.bg_pad_em * px * 0.6)
        boxed = Image.new("RGBA", (row.width + 2 * px_, lh + 2 * py_), (0, 0, 0, 0))
        _rrect(boxed, (0, round(lh * 0.12), boxed.width - 1, boxed.height - round(lh * 0.08)), C.colour(colour, c.pal),
               (boxed.height - lh * 0.2) / 2 if ly.bg_box == "pill" else (ly.bg_radius_em or 0.2) * px)
        boxed.alpha_composite(row, (px_, py_))
        row = boxed
    # centre the row on the layer's box; the canvas is as big as needed
    cw, ch = max(bw, row.width), max(bh, row.height)
    out = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    out.alpha_composite(row, ((cw - row.width) // 2, (ch - row.height) // 2))
    return out, -((cw - bw) // 2), -((ch - bh) // 2)


def _fit_frac(texts: list[str], ly: C.Layer, c: C.Ctx, width: int, frac: float) -> float:
    """Shrink the size until the chunk's words (plus box padding) fit the layer's width."""
    for _ in range(30):
        spec = replace(C.text_spec(ly, c.pal, "", size_frac=frac), box="none")
        px = max(10, round(c.H * frac))
        total = sum(word_image(t, spec, c.H).width for t in texts) + round(px * 0.28) * (len(texts) - 1)
        if total + 2 * round(ly.bg_pad_em * px) <= width or px <= 10:
            break
        frac *= 0.93
    return frac


def captions_states(ly: C.Layer, i: int, c: C.Ctx, words: list[Word], tl: TL.Timeline) -> list[Overlay]:
    """`words` are on the source timeline: chunks are decided there (a pause the cuts shorten still breaks a chunk) and
    each chunk is then mapped through the cuts."""
    box = C.layer_box(ly, c, i)
    bw, bh = box[2] - box[0], box[3] - box[1]
    chunks = [_map_words(ch, tl) for ch in TL.chunk_words(words, ly.max_words, ly.pause_ms)]
    out = []
    base = ly.size_frac or DEFAULT_SIZE["captions"]
    pending = {_mark_key(k): v for k, v in (ly.marks or {}).items()}  # footnote markers, each used on the word's first spoken use
    for ci, chunk in enumerate(chunks):
        texts = [_case(w.text, ly.case) for w in chunk]
        for k, w in enumerate(chunk):
            key = _mark_key(w.text)
            if key in pending:
                texts[k] += pending.pop(key)
        frac = _fit_frac(texts, ly, c, bw, base)
        nxt = chunks[ci + 1][0].start_ms if ci + 1 < len(chunks) else None
        end = chunk[-1].end_ms + HOLD_MS
        end = min(end, nxt) if nxt is not None else end
        end = max(end, chunk[-1].start_ms + 1)
        if ly.mode == "chunk":
            spans = [(chunk[0].start_ms, end, len(chunk), None)]
        else:
            starts = [w.start_ms for w in chunk]
            ends = starts[1:] + [end]
            spans = [(a, max(a, b), k + 1 if ly.mode == "word" else len(chunk), k)
                     for k, (a, b) in enumerate(zip(starts, ends))]
        for a, b, shown, active in spans:
            img, dx, dy = caption_image(texts, shown, active, ly, c, (bw, bh), frac)
            got = _crop(img, box[0] + dx, box[1] + dy)
            if got:
                out.append(Overlay(got[0], got[1], got[2], a, b, i))
    return out


# ---- stack -------------------------------------------------------------------------------------------------------

def stack_states(ly: C.Layer, i: int, c: C.Ctx, words: list[Word], tl: TL.Timeline) -> list[Overlay]:
    box = C.layer_box(ly, c, i)
    bw, bh = box[2] - box[0], box[3] - box[1]
    phrases = [_map_words(ph, tl) for ph in TL.stack_phrases(words, ly.max_lines, ly.pause_ms)]  # phrases on source time
    out = []
    for pi, phrase in enumerate(phrases):
        texts = [_case(w.text, ly.case) for w in phrase]
        frac = ly.size_frac or DEFAULT_SIZE["stack"]
        for _ in range(30):  # one size per phrase, so lines don't jump as the stack builds
            spec = replace(C.text_spec(ly, c.pal, "", size_frac=frac), box="none")
            px = max(10, round(c.H * frac))
            imgs = [word_image(t, spec, c.H) for t in texts]
            pitch = round(px * (ly.line_height or 1.0))
            if (max(i_.width for i_ in imgs) <= bw and pitch * (len(texts) - 1) + imgs[0].height <= bh) or px <= 10:
                break
            frac *= 0.93
        nxt = phrases[pi + 1][0].start_ms if pi + 1 < len(phrases) else None
        end = min(phrase[-1].end_ms + HOLD_MS, nxt) if nxt is not None else phrase[-1].end_ms + HOLD_MS
        starts = [w.start_ms for w in phrase]
        for k, (a, b) in enumerate(zip(starts, starts[1:] + [max(end, starts[-1] + 1)])):
            img = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
            for j in range(k + 1):
                im = imgs[j]
                x = {"left": 0, "right": bw - im.width}.get(ly.align, (bw - im.width) // 2)
                img.alpha_composite(im, (max(0, x), j * pitch))
            got = _crop(img, box[0], box[1])
            if got:
                out.append(Overlay(got[0], got[1], got[2], a, max(a, b), i))
    return out


# ---- list --------------------------------------------------------------------------------------------------------

def list_alpha(d: float) -> float:
    """Brightness by distance from the active position: 1 at 0, 25% at 1, 15% from 2 on (linear between)."""
    d = abs(d)
    if d <= 1:
        return 1.0 - 0.75 * d
    if d <= 2:
        return 0.25 - 0.10 * (d - 1)
    return 0.15


def list_states(ly: C.Layer, i: int, c: C.Ctx, words: list[Word], duration_ms: int, tl: TL.Timeline) -> list[Overlay]:
    box = C.layer_box(ly, c, i)
    bw, bh = box[2] - box[0], box[3] - box[1]
    words = _map_words(words, tl)
    if ly.items:
        texts = [_case(t, ly.case) for t in ly.items]
        if ly.at_ms and len(ly.at_ms) >= len(texts):
            starts = [round(tl.map(a)) for a in ly.at_ms[:len(texts)]]
        else:
            t0 = round(tl.map(ly.t.in_ms)) if ly.t else (words[0].start_ms if words else 0)
            end0 = round(tl.map(duration_ms))
            starts = [t0 + round(k * (end0 - t0) / len(texts)) for k in range(len(texts))]
        end_all = round(tl.map(ly.t.out_ms)) if ly.t and ly.t.out_ms else round(tl.map(duration_ms))
    else:
        if not words:
            return []
        texts = [_case(w.text, ly.case) for w in words]
        starts = [w.start_ms for w in words]
        end_all = words[-1].end_ms + 600
    n = ly.window
    frac = ly.size_frac or DEFAULT_SIZE["list"]
    for _ in range(30):
        spec = replace(C.text_spec(ly, c.pal, "", size_frac=frac), box="none")
        px = max(10, round(c.H * frac))
        pitch = round(px * (ly.line_height or 1.25))
        if pitch * n <= bh or px <= 10:
            break
        frac *= 0.93
    kw = {k.lower(): v for k, v in (ly.keyword_colours or {}).items()}
    base_cols = tuple(spec.colours or ())
    lines: dict[int, Image.Image] = {}

    def line(e: int) -> Image.Image:
        if e not in lines:
            sp = spec
            if texts[e].lower() in kw:
                sp = replace(spec, colours=("#%02x%02x%02x" % C.colour(kw[texts[e].lower()], c.pal)[:3],) + base_cols[1:])
            im = word_image(texts[e], sp, c.H)
            if im.width > bw:
                im = im.resize((bw, max(1, round(im.height * bw / im.width))), Image.LANCZOS)
            lines[e] = im
        return lines[e]

    def state(pos: float) -> Image.Image:
        """The list with the active position at `pos` (an entry index, fractional while scrolling)."""
        img = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        for e in range(max(0, int(pos) - n // 2 - 1), min(len(texts), int(pos) + n // 2 + 2)):
            if abs(e - pos) > n // 2 + 1e-6:
                continue
            im = line(e)
            alpha = list_alpha(e - pos)
            if alpha < 1:
                im = im.copy()
                im.putalpha(im.getchannel("A").point(lambda v: round(v * alpha)))
            x = {"left": 0, "right": bw - im.width}.get(ly.align, (bw - im.width) // 2)
            img.alpha_composite(im, (max(0, x), round(bh / 2 + (e - pos) * pitch - im.height / 2)))
        return img

    out = []

    def emit(img: Image.Image, a: float, b: float) -> None:
        got = _crop(img, box[0], box[1])
        if got and b > a:
            out.append(Overlay(got[0], got[1], got[2], a, b, i))

    ends = starts[1:] + [max(end_all, starts[-1] + 1)]
    for k, (a, b) in enumerate(zip(starts, ends)):
        b = max(a, b)
        steps = TL.scroll_offsets(ly.scroll_ms, b - a) if k else []
        for j, (off, prog) in enumerate(steps):
            nxt = steps[j + 1][0] if j + 1 < len(steps) else min(ly.scroll_ms, int((b - a) * 0.8))
            emit(state(k - 1 + prog), a + off, a + nxt)
        rest = a + (min(ly.scroll_ms, int((b - a) * 0.8)) if steps else 0)
        emit(state(float(k)), rest, b)
    return out


# ---- timed ordinary layers ---------------------------------------------------------------------------------------

def _variant(img: Image.Image, box, st: TL.AnimState, ly: C.Layer, c: C.Ctx) -> tuple[Image.Image, int, int] | None:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2 + st.dy * c.H
    if st.scale != 1.0:
        img = img.resize((max(1, round(img.width * st.scale)), max(1, round(img.height * st.scale))), Image.LANCZOS)
    w, h = img.size
    nb = (round(cx - w / 2), round(cy - h / 2), round(cx - w / 2) + w, round(cy - h / 2) + h)
    im, x, y = C.placed(img, nb, 0 if ly.type == "band" else ly.rotate, ly.opacity * st.alpha)
    return _crop(im, x, y)


def timed_states(ly: C.Layer, i: int, c: C.Ctx, tl: TL.Timeline, duration_ms: int) -> list[Overlay]:
    tin = round(tl.map(ly.t.in_ms)) if ly.t else 0
    tout = round(tl.map(ly.t.out_ms)) if ly.t and ly.t.out_ms is not None else None
    states = TL.ladder(ly.anim, ly.dur_ms)
    out = []
    base = C.layer_image(ly, i, c)
    if base is None:
        return out
    img, box = base
    for a, b, st in TL.windows_for(tin, tout, states):
        use, ubox = img, box
        if st.chars < 1.0 or (ly.anim == "type" and ly.cursor and st.ms):
            if ly.type not in ("text", "sticker") or not ly.content:
                raise ValueError(f"layer {i}: anim 'type' needs a text or sticker layer with content")
            shown = ly.content[:max(1, round(len(ly.content) * st.chars))] + ("|" if ly.cursor else "")
            got = C.layer_image(ly, i, c, content=shown)
            if got is None:
                continue
            use, ubox = got
        got = _variant(use, ubox, st, ly, c)
        if got:
            out.append(Overlay(got[0], got[1], got[2], a, b, i))
    return out


# ---- the plan ----------------------------------------------------------------------------------------------------

def is_timed(ly: C.Layer) -> bool:
    return ly.type in C.WORD_TYPES or ly.t is not None or ly.anim != "none"


def plan_reel(layout: C.Layout, words: list[Word], duration_ms: int, inputs: list[Image.Image] | None = None,
              seed: int | None = None, *, mask_fn: C.MaskFn | None = None,
              timeline: TL.Timeline | None = None) -> Plan:
    """The overlay states for `layout` over a clip of `duration_ms` (source time) with spoken `words` (source time).

    The clip is the bottom of the stack (spec §2): the canvas background shows round it (a flat colour pads the window; a
    pattern is an overlay with the window punched out), and every other layer, static or timed, draws over it in list
    order. With no clip layer the clip fills the canvas (cover) and canvas.bg is unused. With a `timeline` (tighten)
    layer windows move onto the cut timeline, and chunks/phrases are decided on source time before being mapped."""
    tl = timeline or TL.Timeline()
    W, H = layout.canvas.w, layout.canvas.h
    base_seed = layout.seed if seed is None else seed
    c = C.Ctx(W, H, C.palette_of(layout), base_seed, inputs or [], mask_fn, C.Notes())
    plan = Plan(notes=c.notes, duration_ms=round(tl.map(duration_ms)))
    clip_idx = next((k for k, ly in enumerate(layout.layers) if ly.type == "photo" and ly.src in ("clip", "video")), None)
    overlays: list[Overlay] = []
    if clip_idx is not None:
        ly = layout.layers[clip_idx]
        plan.clip_fit = ly.clip_fit
        plan.window = C.layer_box(ly, c, clip_idx, safe=False)
        c.notes.windows.append(plan.window)
        bg = layout.canvas.bg
        if isinstance(bg, str):
            r, g, b, _ = C.colour(bg, c.pal)
            plan.pad_colour = f"0x{r:02x}{g:02x}{b:02x}"
        else:
            canvas = C.new_canvas(layout, base_seed)
            C.clip_hole(canvas, plan.window, C.Notes())
            overlays.append(Overlay(canvas, 0, 0, 0, None, -1))
    pending: Image.Image | None = None

    def flush():
        nonlocal pending
        if pending is not None:
            got = _crop(pending, 0, 0)
            if got:
                overlays.append(Overlay(got[0], got[1], got[2], 0, None, -1))
        pending = None

    for k, ly in enumerate(layout.layers):
        if k == clip_idx:
            continue
        if not is_timed(ly):
            got = C.layer_image(ly, k, c)
            if got is not None:
                if pending is None:
                    pending = C.new_canvas(layout, base_seed, transparent=True)
                im, x, y = C.placed(got[0], got[1], 0 if ly.type == "band" else ly.rotate, ly.opacity)
                C._paste_clipped(pending, im, x, y)
            continue
        flush()
        if ly.type == "captions":
            states = captions_states(ly, k, c, words, tl)
        elif ly.type == "stack":
            states = stack_states(ly, k, c, words, tl)
        elif ly.type == "list":
            states = list_states(ly, k, c, words, duration_ms, tl)
        else:
            states = timed_states(ly, k, c, tl, plan.duration_ms)
        if ly.type in C.WORD_TYPES and ly.t is not None:  # clip the word-driven states to the layer's window
            a = round(tl.map(ly.t.in_ms))
            b = round(tl.map(ly.t.out_ms)) if ly.t.out_ms is not None else None
            states = [replace(s, start_ms=max(s.start_ms, a), end_ms=s.end_ms if b is None else min(s.end_ms, b))
                      for s in states if s.start_ms < (b if b is not None else 10 ** 9)
                      and (s.end_ms is None or s.end_ms > a)]
        overlays.extend(states)
    flush()
    plan.overlays = overlays
    if len(overlays) > MAX_OVERLAYS:
        raise ValueError(f"{len(overlays)} overlay states; limit {MAX_OVERLAYS} (fewer words or layers)")
    return plan


# ---- ffmpeg ------------------------------------------------------------------------------------------------------

def _s(ms: float) -> str:
    return f"{ms / 1000:.3f}"


def graph_for(plan: Plan, size: tuple[int, int], tmpd: Path, kept: list[tuple[int, int]] | None,
              has_audio: bool, look_vf: str | None = None) -> tuple[list[str], str, str | None]:
    """(extra input args after the clip, filter_complex, audio map label or None).

    The clip is trimmed and concatenated per `kept` (tighten), scaled into the window (contain pads black, cover crops),
    graded by the optional `look_vf`, then every overlay PNG goes on in layer order with a half-open `gte(t,a)*lt(t,b)` enable, so no frame shows two
    states of one layer."""
    W, H = size
    x0, y0, x1, y1 = plan.window or (0, 0, W, H)
    ww, wh = (x1 - x0) // 2 * 2, (y1 - y0) // 2 * 2
    x0, y0 = x0 // 2 * 2, y0 // 2 * 2
    fit_cover = plan.clip_fit == "cover" or plan.window is None
    parts = []
    vin, alabel = "0:v", None
    if kept is not None:  # video and audio are cut at the same instants and joined by one concat, so they can't drift
        k = len(kept)
        parts.append(f"[0:v]split={k}" + "".join(f"[s{j}]" for j in range(k)))
        if has_audio:
            parts.append(f"[0:a]asplit={k}" + "".join(f"[t{j}]" for j in range(k)))
        for j, (a, b) in enumerate(kept):
            parts.append(f"[s{j}]trim=start={_s(a)}:end={_s(b)},setpts=PTS-STARTPTS[v{j}]")
            if has_audio:
                parts.append(f"[t{j}]atrim=start={_s(a)}:end={_s(b)},asetpts=PTS-STARTPTS[a{j}]")
        if has_audio:
            parts.append("".join(f"[v{j}][a{j}]" for j in range(k)) + f"concat=n={k}:v=1:a=1[vt][ac]")
            alabel = "[ac]"
        else:
            parts.append("".join(f"[v{j}]" for j in range(k)) + f"concat=n={k}:v=1:a=0[vt]")
        vin = "[vt]"
    vin = vin if vin.startswith("[") else f"[{vin}]"
    if look_vf:  # the look grades the clip only (after the cuts, before scaling), never the text or bands: same encode
        parts.append(f"{vin}{look_vf}[lk]")
        vin = "[lk]"
    scale = (f"scale={ww}:{wh}:force_original_aspect_ratio=increase:force_divisible_by=2,crop={ww}:{wh}" if fit_cover else
             f"scale={ww}:{wh}:force_original_aspect_ratio=decrease:force_divisible_by=2")
    px = f"{x0}" if fit_cover else f"trunc(({x0}+({ww}-iw)/2)/2)*2"
    py = f"{y0}" if fit_cover else f"trunc(({y0}+({wh}-ih)/2)/2)*2"
    parts.append(f"{vin}{scale},pad={W}:{H}:{px}:{py}:color={plan.pad_colour},setsar=1[b0]")
    args: list[str] = []
    last = "b0"
    for n, ov in enumerate(plan.overlays, 1):
        ov.img.save(tmpd / f"ov{n}.png")
        args += ["-i", f"ov{n}.png"]
        en = ""
        if ov.end_ms is not None:
            en = f":enable='gte(t,{_s(ov.start_ms)})*lt(t,{_s(ov.end_ms)})'"
        elif ov.start_ms > 0:
            en = f":enable='gte(t,{_s(ov.start_ms)})'"
        parts.append(f"[{last}][{n}:v]overlay={ov.x}:{ov.y}:format=auto{en}[o{n}]")
        last = f"o{n}"
    parts.append(f"[{last}]format=yuv420p[o]")
    return args, ";".join(parts), alabel
