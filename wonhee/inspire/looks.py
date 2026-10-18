"""Measured looks across saved posts: which posts share a photographic style, in numbers, not a model's guess.

A teardown describes a post's look by eye, and models misread light (a fisheye night shoot as "soft even light").
Here each post's photos (or its video keyframes) are measured with wonhee.edit.style, posts with close numbers are
grouped, and each group gets its shared traits and the `wonhee vibe` command that learns that look.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from wonhee.edit import style
from wonhee.media import path_for

MAX_FRAMES = 6
SATURATION = ("saturated", "muted colour", "moderate colour")
GROUP_DISTANCE = 0.8  # units of _vector; at 1.0 ordinary phone footage chains into one 19-post group (2026-10-18)


def post_images(conn, media_dir: Path, content_id: int) -> list[Path]:
    """A post's photos, else up to MAX_FRAMES keyframes of its video, spread over the shots."""
    shas = [r[0] for r in conn.execute("SELECT media_sha256 FROM content_media WHERE content_id=? AND kind='image' "
                                       "ORDER BY ord", (content_id,))]
    if not shas:
        shas = [r[0] for r in conn.execute(
            """SELECT s.keyframe_sha256 FROM media_shot s JOIN content_item c ON c.media_sha256=s.media_sha256
               WHERE c.id=? AND s.keyframe_sha256 IS NOT NULL ORDER BY s.ord""", (content_id,))]
        shas = list(dict.fromkeys(shas))
        if len(shas) > MAX_FRAMES:
            shas = [shas[round(i * (len(shas) - 1) / (MAX_FRAMES - 1))] for i in range(MAX_FRAMES)]
    return [p for p in (path_for(media_dir, s) for s in shas[:12]) if p.exists()]


def _vector(s: style.Stats) -> np.ndarray:
    """Tone (dark / mid / light points), colour per band, colourfulness and lens traits, scaled so ~1 unit each
    is a visible difference."""
    return np.array([s.lum[1] / 12, s.lum[3] / 12, s.lum[5] / 12,
                     *[s.bands[b].a / 6 for b in style.BANDS], *[s.bands[b].b / 6 for b in style.BANDS],
                     s.chroma / 8, s.edge_sharpness * 2.5, s.vignette * 2.5])


def distance(a: style.Stats, b: style.Stats) -> float:
    return float(np.linalg.norm(_vector(a) - _vector(b)) / np.sqrt(len(_vector(a))))


def measure_posts(conn, media_dir: Path, content_ids: list[int]) -> dict[int, style.Profile]:
    out = {}
    for cid in content_ids:
        paths = post_images(conn, media_dir, cid)
        if paths:
            out[cid] = style.profile([style.load(p) for p in paths], f"c{cid}")
    return out


def group(profiles: dict[int, style.Profile], threshold: float = GROUP_DISTANCE) -> list[list[int]]:
    """Average-link grouping: merge the two closest groups while their mean distance is under `threshold`.
    Returns groups of 2+ posts, tightest first; singletons are left out."""
    groups = [[c] for c in profiles]
    d = {(a, b): distance(profiles[a].stats, profiles[b].stats) for a in profiles for b in profiles if a < b}
    dist = lambda g, h: float(np.mean([d[min(a, b), max(a, b)] for a in g for b in h]))  # noqa: E731
    while len(groups) > 1:
        i, j, best = min(((i, j, dist(groups[i], groups[j])) for i in range(len(groups))
                          for j in range(i + 1, len(groups))), key=lambda t: t[2])
        if best > threshold:
            break
        groups[i] = sorted(groups[i] + groups.pop(j))
    multi = [g for g in groups if len(g) > 1]
    spread = lambda g: float(np.mean([d[a, b] for a in g for b in g if a < b]))  # noqa: E731
    return sorted(multi, key=spread)


def section(conn, media_dir: Path, content_ids: list[int], summaries: dict[int, str]) -> str:
    """Markdown for the inspire report: shared looks, then each post's measured traits."""
    profs = measure_posts(conn, media_dir, content_ids)
    if not profs:
        return ""
    out = ["## Looks (measured from the pixels)", "",
           "Posts grouped by how close their tone, colour per band and lens traits are. Learn a look and get a "
           "shooting guide with the command under it.", ""]
    plain, n = [], 0
    for g in group(profs):
        allp = [p for c in g for p in post_images(conn, media_dir, c)]
        prof = style.profile([style.load(p) for p in allp], "look")
        if not [h for h in prof.hints if h not in SATURATION]:  # nothing to learn: ordinary phone footage
            plain += g
            continue
        n += 1
        out += [f"**Look {n}: {', '.join(f'c{c}' for c in g)}**: {'; '.join(prof.hints)}.",
                "- " + "; ".join(f"c{c} {summaries.get(c, '')[:60]}" for c in g),
                f"- `wonhee vibe <your photos> {' '.join(f'--like content:{c}' for c in g)} --name look{n}`", ""]
    if plain:
        out += [f"Ordinary phone-footage look, nothing distinctive to copy: {', '.join(f'c{c}' for c in sorted(plain))}.",
                ""]
    out += ["| post | traits |", "|---|---|"]
    out += [f"| c{c} | {'; '.join(p.hints)} |" for c, p in sorted(profs.items())]
    return "\n".join(out) + "\n"
