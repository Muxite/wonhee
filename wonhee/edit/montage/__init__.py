"""Montage engine: joins stills (and clips) into one video with xfade transitions and motion effects.

Ported from the owner's slopcut project (docs/specs/2026-10-28-slopcut-port-design.md). It lives apart from
`edit/slop.py` (GitHub repo cards): that is a card layout, this is a timeline. Modules here are plain functions on
files and dicts, with no DB access; nodes wrap them.

- transitions / transitions_custom: the xfade names, grouped into style pools
- effects: zoompan motion presets for a still
- plan: a seeded, hand-editable dict that fully describes a render
- render: plan -> mp4 in cached passes (clips, transitions, final)
"""
