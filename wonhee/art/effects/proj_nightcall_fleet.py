"""Nightcall: a fleet of ballistic ships (thrust + drag, pursuit) leaving light trails."""
from __future__ import annotations

import cv2
import numpy as np

from ..registry import effect
from . import light_common as lc

_DEFAULT_WARP = [[0.375, 0.0, 0.0, 0.575, 0.975, 0.65, 0.925, 0.4],
                 [0.35, 0.0, 0.025, 0.55, 1.0, 0.875, 0.85, 0.375],
                 [0.05, 0.0, 0.0, 0.325, 1.0, 0.975, 1.0, 0.825]]


def _warp_table(ctx):
    d = ctx.data
    if isinstance(d, dict) and "warp" in d:
        d = d["warp"]
    try:
        a = np.asarray(d, float)
        if a.ndim == 2 and a.shape[0] >= 3:
            return np.clip(a[:3], 0, 1)
    except Exception:
        pass
    return np.asarray(_DEFAULT_WARP)


def _warm_ramp(hue, n=512):
    from ..palette import oklch_to_rgb
    t = np.linspace(0, 1, n)
    cols = np.array([oklch_to_rgb(L, C, hue) for L, C in ((0.2, 0.0), (0.55, 0.14), (0.78, 0.15), (0.95, 0.04))], np.float32)
    pos = np.array([0, 0.45, 0.78, 1.0])
    return np.stack([np.interp(t, pos, cols[:, i]) for i in range(3)], 1).astype(np.float32)


@effect("proj-nightcall-fleet", family="light", tags=["nightcall", "ballistic", "trails", "all-over", "physics"],
        params={"ships": (150, "ships (two teams) on a periodic arena"),
                "steps": (420, "simulation ticks"),
                "drag": (0.035, "linear drag per tick"),
                "thrust": (0.16, "max thrust (px/tick^2 at 1500 wide)"),
                "fire": (0.016, "weapon-vector probability per ship per tick when in range"),
                "warp_scale": (1.0, "scale of the warp fraction taken from ctx.data (3 levels x checkpoints)")})
def fleet(ctx):
    """Real game physics in miniature: each ship is a particle with thrust,
    linear drag and a turn-limited pursuit of the nearest enemy (self-play
    pilots); trails are the ship worldlines, drawn as light. Weapon vectors
    (orange) are the straight ship-to-target lines. ctx.data = warp rate table
    (3 eval levels x checkpoints, from the real collapsed PPO run): each ship is
    assigned a (level, checkpoint); with that warp rate it hops (a discontinuous
    jump, ring marker) instead of fighting. Periodic arena: even coverage."""
    W, H = ctx.size
    sc = lc.scale(ctx); rng = ctx.rng
    n = int(ctx.p("ships")); T = int(ctx.p("steps"))
    warp = _warp_table(ctx) * float(ctx.p("warp_scale"))
    pos = lc.blue_noise(rng, n, W, H)
    n = len(pos)
    team = (np.arange(n) % 2)
    rng.shuffle(team)
    vel = rng.normal(0, 1.2 * sc, (n, 2))
    lvl = rng.integers(0, 3, n); ck = rng.integers(0, warp.shape[1], n)
    pw = warp[lvl, ck]
    warper = rng.random(n) < pw
    mass = rng.uniform(0.6, 1.5, n)
    box = np.array([W, H], float)
    drag = float(ctx.p("drag")); th = float(ctx.p("thrust")) * sc
    cy = np.zeros((H, W), np.float32); ow = np.zeros((H, W), np.float32); ring = np.zeros((H, W), np.float32)
    ang = rng.uniform(0, 6.283, n); jit = rng.normal(0, 1, (n, 2))
    for t in range(T):
        d = pos[None, :, :] - pos[:, None, :]
        d -= box * np.round(d / box)
        dist = np.hypot(d[..., 0], d[..., 1]) + 1e-6
        dist[team[:, None] == team[None, :]] = 1e9
        tgt = dist.argmin(1); dm = dist[np.arange(n), tgt]
        dv = d[np.arange(n), tgt]
        rel = vel[tgt] - vel
        lead = dv + rel * 12
        want = lead / (np.hypot(lead[:, 0], lead[:, 1])[:, None] + 1e-6)
        # orbit a little instead of ramming: tangential component inside 160 px
        tang = np.stack([-want[:, 1], want[:, 0]], 1)
        near = np.clip(1.4 - dm / (230 * sc), 0, 1)[:, None]
        want = want * (1 - 0.8 * near) + tang * near * (np.where(team, 1, -1)[:, None])
        jit = 0.9 * jit + 0.4 * rng.normal(0, 1, (n, 2))
        acc = (th * (want + 0.25 * jit)) / mass[:, None]
        vel = vel * (1 - drag) + acc
        new = pos + vel
        # draw non-wrapping segments
        a = pos.copy(); b = new.copy()
        wrapped = (np.abs(b - a) > box * 0.5).any(1)
        age = t / T
        w = (0.12 + 0.9 * age ** 1.3) * 0.9
        sel = ~wrapped & ~(warper & (t % 140 == 139))
        wt = w * np.where(warper, 0.55, 1.0) * (0.6 + 0.4 * mass)
        lc.splat_segments(cy, a[sel], b[sel], wt[sel], step=0.9)
        pos = new
        # hop: discontinuous jump for warpers, with ring markers
        hop = warper & (t % 140 == 139)
        if hop.any():
            land = pos[hop] + rng.normal(0, 1, (hop.sum(), 2)) * 160 * sc
            for p0, p1 in zip(pos[hop], land):
                cv2.circle(ring, (int(p0[0]) % W, int(p0[1]) % H), int(5 * sc), 1.0, 1, cv2.LINE_AA)
                cv2.circle(ring, (int(p1[0]) % W, int(p1[1]) % H), int(9 * sc), 0.7, 1, cv2.LINE_AA)
            vel[hop] *= 0.2
            pos[hop] = land
        pos %= box
        # weapon vectors: fighters (non-warpers) fire at the nearest enemy in range
        if t % 3 == 0:
            fire = (~warper) & (dm < 330 * sc) & (rng.random(n) < float(ctx.p("fire")) * 3)
            for i in np.nonzero(fire)[0]:
                p0 = pos[i]; p1 = pos[i] + dv[i]
                lc.splat_segments(ow, p0[None], p1[None], np.array([2.4 * (1 - dm[i] / (330 * sc)) + 0.5]), step=1.0)
    # cyan trails via palette ramp, orange vectors via warm ramp (sparing)
    base = lc.develop(ctx, cy + 0.35 * ring, exposure=1.5, glow=0.9, glow_sigmas=(1.5, 5, 16), base=lc.ground(ctx))
    ocol = lc.develop(ctx, ow, exposure=0.9, glow=0.6, glow_sigmas=(1.2, 4, 10), grain=0, ramp=_warm_ramp(48))
    m = (1 - np.exp(-0.9 * ow))[..., None] ** 0.8
    out = base * (1 - 0.85 * m) + ocol * 0.85 * m
    return np.clip(out, 0, 1).astype(np.float32)
