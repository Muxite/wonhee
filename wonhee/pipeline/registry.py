from __future__ import annotations

from wonhee.pipeline.context import Stage
from wonhee.scout.stages import analyze, describe, discover, fetch, pick, resolve, shotlist

SCOUT_STAGES: list[Stage] = [
    Stage("resolve", (), resolve.run),
    Stage("discover", ("resolve",), discover.run),
    Stage("pick", ("discover",), pick.run),
    Stage("fetch", ("pick",), fetch.run),
    Stage("analyze", ("fetch",), analyze.run),
    Stage("describe", ("analyze",), describe.run),
    Stage("shotlist", ("describe",), shotlist.run),
]

from wonhee.library import grab  # noqa: E402

GRAB_STAGES: list[Stage] = [
    Stage("grab", (), grab.grab),
    Stage("analyze", ("grab",), grab.analyze),
]

STAGES_BY_KIND: dict[str, list[Stage]] = {"scout": SCOUT_STAGES, "library": GRAB_STAGES}


def stages_for(kind: str) -> list[Stage]:
    if kind not in STAGES_BY_KIND:
        raise ValueError(f"project kind {kind!r} has no pipeline")
    return STAGES_BY_KIND[kind]
