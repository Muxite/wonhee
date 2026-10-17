"""Aggregate teardowns into one actionable report: no model call, so it is free and repeatable."""
from __future__ import annotations

from collections import Counter

# capability -> the wonhee node that already provides it (missing = a build candidate)
PROVIDED_BY = {
    "text_overlay": "text_image / text_video", "colour_emoji": "text_image / text_video",
    "band_text": "text_* preset=band", "script_text": "text_* preset=script", "look_grade": "look / look_image",
    "film_grain": "look preset=grain", "datestamp": "look datestamp=", "carousel_layout": "carousel",
    "panorama_span": "carousel span=true", "translation": "translate engine=local",
    "subtitles_from_speech": "speech_subs + burn_subtitles", "music_in_app": "(done in the Instagram app)",
}


# capabilities that are the creator's job on the shoot, not tooling wonhee could build
SHOOT_SIDE = {"on_camera_performance", "location_shoot", "props_or_costume", "two_person_shoot",
              "illustration_or_drawing", "stock_or_archive_footage"}


def _plays(m: dict | None) -> int:
    return (m or {}).get("plays") or (m or {}).get("likes") or 0


def _count_line(c: Counter) -> str:
    return ", ".join(f"{k} {v}" for k, v in c.most_common()) or "-"


def build(teardowns: list[dict], metrics: dict[int, dict], failed: list[int] | None = None) -> str:
    n = len(teardowns)
    out = [f"# Inspiration report: {n} post(s)", ""]
    if failed:
        out += [f"Teardown failed for content {', '.join(map(str, failed))} (see the run's flags).", ""]
    fmt = Counter(t["format"] for t in teardowns)
    cpl = Counter(t["coupling"]["level"] for t in teardowns)
    style = Counter(f"{t['text_overlay']['style']}@{t['text_overlay']['position']}" for t in teardowns
                    if t["text_overlay"]["present"])
    looks = Counter(t["look"] or "none" for t in teardowns)
    audio = Counter(t["audio"]["kind"] for t in teardowns)
    out += ["## What the set looks like",
            f"- formats: {_count_line(fmt)}",
            f"- text and footage: {_count_line(cpl)} (decoupled/loose = film footage and write text separately)",
            f"- text overlays: {sum(t['text_overlay']['present'] for t in teardowns)} of {n}; style@position: "
            f"{_count_line(style)}",
            f"- looks: {_count_line(looks)}",
            f"- audio: {_count_line(audio)}", ""]

    caps = Counter(c for t in teardowns for c in t["capabilities"])
    out += ["## What to build (tooling the set needs)", "", "| capability | posts | wonhee today |", "|---|---|---|"]
    for c, k in caps.most_common():
        if c not in SHOOT_SIDE:
            out.append(f"| {c} | {k} | {PROVIDED_BY.get(c, '**missing**')} |")
    shoot = [(c, k) for c, k in caps.most_common() if c in SHOOT_SIDE]
    if shoot:
        out += ["", "Shoot side (your job, not tooling): " + ", ".join(f"{c} {k}" for c, k in shoot)]
    out.append("")

    ranked = sorted(teardowns, key=lambda t: (-t["replicate"]["score_1to5"], -_plays(metrics.get(t["content_id"]))))
    out += ["## Best to replicate", "", "| id | format | coupling | score | effort | plays/likes | summary |",
            "|---|---|---|---|---|---|---|"]
    for t in ranked[:12]:
        out.append(f"| {t['content_id']} | {t['format']} | {t['coupling']['level']} | {t['replicate']['score_1to5']} | "
                   f"{t['replicate']['effort_minutes']} min | {_plays(metrics.get(t['content_id'])):,} | "
                   f"{t['summary'][:140]} |")
    out.append("")

    shoot = [(t["content_id"], f) for t in ranked if t["coupling"]["level"] in ("decoupled", "loose")
             and t["replicate"]["score_1to5"] >= 3 for f in t["footage_to_record"]]
    out += ["## Shoot list (footage that pairs with many lines)", ""]
    for on_cam, title in ((True, "You on camera"), (False, "Places and things")):
        group = sorted((x for x in shoot if x[1]["person_on_camera"] == on_cam),
                       key=lambda x: (x[1]["difficulty_1to5"], x[1]["camera"]))
        if not group:
            continue
        out += [f"**{title}**", ""]
        for cid, f in group:
            out.append(f"- {f['subject']} | {f['setting']} | {f['light']} | {f['camera']}, {f['framing']} | "
                       f"{f['duration_s']:g}s | difficulty {f['difficulty_1to5']} | from {cid}")
        out.append("")

    lines = [t for t in ranked if t.get("text_template")]
    out += ["## Line bank (templates, with how to render them)", ""]
    for t in lines:
        o = t["text_overlay"]
        how = []
        style = {"plain": "outline", "other": "outline"}.get(o["style"], o["style"])  # IG's plain font ~ outline
        if style in ("outline", "band", "script"):
            how.append(f"preset={style}")
        if o["position"] != "none":
            how.append(f"position={o['position']}")
        if t["look"] and t["look"] != "none":
            how.append(f"look={t['look']}")
        render = f"`text_video {' '.join(how)}`" if how else "`text_video`"
        out.append(f"- {t['text_template']}  → {render}  (from {t['content_id']}, {t['format']})")
    out.append("")

    learn = list(dict.fromkeys(l for t in ranked for l in t["learn"]))
    out += ["## Learnings", "", *[f"- {l}" for l in learn[:20]], ""]
    low = [t["content_id"] for t in teardowns if t["confidence"] == "low"]
    if low:
        out += [f"Low confidence (thin material): {', '.join(map(str, low))}.", ""]
    return "\n".join(out)
