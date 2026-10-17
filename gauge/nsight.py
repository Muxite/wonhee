"""Nsight Systems inside job containers, and a summary of what the GPU did for that process.

Scopes, kept separate on purpose:
- process: kernels and memory copies from CUPTI, for the traced process tree only. Exact even when the GPU is shared.
- device:  GPU hardware counters (SM/tensor/DRAM/PCIe activity, clocks) at 10 kHz. Whole GPU; describes this job
           only when nothing else ran (see the run's initial_state and corun_group).
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
from pathlib import Path

NSYS_DIR = Path.home() / "opt/nsys-root/opt/nvidia/nsight-systems/2026.3.2"
NSYS_HOST = NSYS_DIR / "target-linux-x64" / "nsys"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "nsys"
METRICS_HZ = 10000
# Hardware CUDA trace (default). Ollama SIGKILLs its runner, so its trace is partial; llama.cpp direct is complete.
TRACE = os.environ.get("GAUGE_NSYS_TRACE", "cuda")
GRAPH = os.environ.get("GAUGE_NSYS_GRAPH", "graph" if TRACE == "cuda-sw" else "node")


def wrap(cmd: list[str], run_id: int, mode: str) -> tuple[list[str], tuple[str, ...], tuple[str, ...]]:
    """-> (command, extra mounts, cap_add). GPU counters need SYS_ADMIN (the driver restricts them to admins)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nsys = ["/nsys/target-linux-x64/nsys", "profile", "-o", f"/nsys-out/run{run_id}", "--force-overwrite=true",
            f"--trace={TRACE},nvtx", f"--cuda-graph-trace={GRAPH}", "--sample=none", "--cpuctxsw=none",
            "--wait=all", "--kill=none",
            "--cuda-flush-interval=500"]  # flush kernel records often: runtimes like Ollama SIGKILL their workers
    if mode == "metrics":
        nsys += ["--gpu-metrics-devices=all", f"--gpu-metrics-frequency={METRICS_HZ}"]
    caps = ("SYS_ADMIN",) if mode == "metrics" else ()
    return nsys + cmd, (f"{NSYS_DIR}:/nsys:ro", f"{OUT_DIR}:/nsys-out"), caps


# Kernel name -> what kind of work it is. Order matters: first match wins.
CATEGORIES = [
    ("gemv (decode matmul)", r"mul_mat_vec|gemv|mmvq|dequantize_mul_mat_vec"),
    ("gemm (batched matmul)", r"mul_mat_q|mmq|gemm|Kernel2|cutlass|cublas|sm\d+_xmma|ampere_|hopper_|nvjet|matmul"),
    ("attention", r"flash_attn|attn|softmax|soft_max|fmha"),
    ("convolution", r"conv|winograd|implicit|cudnn|fft"),
    ("norm", r"norm|rms|layer_norm|batch_norm|bn_"),
    ("rope/positional", r"rope"),
    ("quantize/convert", r"quantize|dequant|convert|cpy|copy|cast|to_fp|f16|bf16"),
    ("reduction", r"reduce|sum|argmax|topk|sort|cumsum"),
    ("elementwise", r"elementwise|vectorized|add|mul|scale|silu|gelu|relu|act|unary|binary|glu|fill"),
]
_CAT_RE = [(name, re.compile(rx, re.I)) for name, rx in CATEGORIES]


def category(name: str) -> str:
    for cat, rx in _CAT_RE:
        if rx.search(name):
            return cat
    return "other"


def export(run_id: int) -> Path | None:
    rep = OUT_DIR / f"run{run_id}.nsys-rep"
    if not rep.exists():
        return None
    out = OUT_DIR / f"run{run_id}.sqlite"
    subprocess.run([str(NSYS_HOST), "export", "--type", "sqlite", "-o", str(out), "--force-overwrite", "true",
                    str(rep)], capture_output=True, timeout=900)
    return out if out.exists() else None


def export_name(name: str) -> Path | None:
    rep = OUT_DIR / f"{name}.nsys-rep"
    if not rep.exists():
        return None
    out = OUT_DIR / f"{name}.sqlite"
    subprocess.run([str(NSYS_HOST), "export", "--type", "sqlite", "-o", str(out), "--force-overwrite", "true",
                    str(rep)], capture_output=True, timeout=900)
    return out if out.exists() else None


def _union_ns(intervals: list[tuple[int, int]]) -> int:
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def _has(c, table: str) -> bool:
    return c.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone() is not None


def summarize(db: Path) -> dict:
    c = sqlite3.connect(db)
    out: dict = {"scope_note": "kernels/copies: this process tree only; gpu_metrics: whole device"}
    kernels = c.execute(
        "SELECT k.start, k.end, s.value, k.gridX*k.gridY*k.gridZ, k.blockX*k.blockY*k.blockZ FROM CUPTI_ACTIVITY_KIND_KERNEL k"
        " JOIN StringIds s ON s.id = k.shortName ORDER BY k.start").fetchall() if _has(c, "CUPTI_ACTIVITY_KIND_KERNEL") else []
    copies = c.execute("SELECT start, end, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY").fetchall() \
        if _has(c, "CUPTI_ACTIVITY_KIND_MEMCPY") else []
    if kernels:
        t0, t1 = kernels[0][0], max(k[1] for k in kernels)
        busy = _union_ns([(k[0], k[1]) for k in kernels])
        gaps = [b[0] - a[1] for a, b in zip(kernels, kernels[1:]) if b[0] > a[1]]
        durs = sorted(k[1] - k[0] for k in kernels)
        cats: dict[str, list] = {}
        names: dict[str, list] = {}
        for s, e, n, grid, block in kernels:
            cats.setdefault(category(n), []).append(e - s)
            names.setdefault(n, []).append(e - s)
        tot = sum(durs)
        out["process"] = {
            "kernel_count": len(kernels),
            "gpu_window_s": (t1 - t0) / 1e9,  # first kernel start to last kernel end
            "kernel_busy_s": busy / 1e9,  # union of kernel intervals: time this process had work on the GPU
            "kernel_busy_frac_of_window": busy / (t1 - t0) if t1 > t0 else None,
            "kernel_time_sum_s": tot / 1e9,
            "kernel_median_us": durs[len(durs) // 2] / 1e3,
            "kernel_p90_us": durs[int(len(durs) * .9)] / 1e3,
            "launches_per_s": len(kernels) / ((t1 - t0) / 1e9) if t1 > t0 else None,
            "gap_median_us": sorted(gaps)[len(gaps) // 2] / 1e3 if gaps else None,
            "gaps_over_1ms": sum(g > 1e6 for g in gaps),
            "idle_in_gaps_over_1ms_s": sum(g for g in gaps if g > 1e6) / 1e9,
            "by_category": {k: {"time_s": sum(v) / 1e9, "share": sum(v) / tot, "count": len(v)}
                            for k, v in sorted(cats.items(), key=lambda kv: -sum(kv[1]))},
            "top_kernels": [{"name": n[:120], "category": category(n), "time_s": sum(v) / 1e9, "share": sum(v) / tot,
                             "count": len(v), "median_us": sorted(v)[len(v) // 2] / 1e3}
                            for n, v in sorted(names.items(), key=lambda kv: -sum(kv[1]))[:12]],
            "first_kernel_ns": t0, "last_kernel_ns": t1,
        }
    if copies:
        kinds = {1: "h2d", 2: "d2h", 8: "d2d", 10: "p2p"}
        agg: dict[str, dict] = {}
        for s, e, b, k in copies:
            a = agg.setdefault(kinds.get(k, f"kind{k}"), {"bytes": 0, "time_s": 0.0, "count": 0})
            a["bytes"] += b
            a["time_s"] += (e - s) / 1e9
            a["count"] += 1
        for a in agg.values():
            a["gb_per_s"] = a["bytes"] / a["time_s"] / 1e9 if a["time_s"] else None
        out["copies"] = agg
    if _has(c, "GPU_METRICS"):
        names = dict(c.execute("SELECT metricId, metricName FROM TARGET_INFO_GPU_METRICS"))
        want = {k: v for k, v in names.items() if re.search(
            r"SMs Active|SM Issue|Tensor Active|DRAM (Read|Write) Bandwidth|PCIe (RX|TX) Throughput|GPC Clock|"
            r"Compute Warps in Flight \[Throughput|Copy Engine Active.*Throughput|GR Active", v)}
        k0, k1 = (out["process"]["first_kernel_ns"], out["process"]["last_kernel_ns"]) if "process" in out else (None, None)
        dev = {}
        for mid, name in want.items():
            rows = c.execute("SELECT timestamp, value FROM GPU_METRICS WHERE metricId=?", (mid,)).fetchall()
            if not rows:
                continue
            if "Clock" in name:  # exported in Hz despite the [MHz] label
                rows = [(t, v / 1e6) for t, v in rows]
            vals = [v for _, v in rows]
            inwin = [v for t, v in rows if k0 is not None and k0 <= t <= k1]
            dev[name] = {"mean_whole": sum(vals) / len(vals), "p95_whole": sorted(vals)[int(len(vals) * .95)],
                         "mean_gpu_window": sum(inwin) / len(inwin) if inwin else None, "samples": len(vals)}
        out["device_metrics"] = dev
        out["device_metrics_hz"] = METRICS_HZ
    return out


def timeline(db: Path, bin_ms: float = 10.0) -> list[dict]:
    """Per-bin kernel busy share by category (process scope) plus device counters (device scope)."""
    c = sqlite3.connect(db)
    if not _has(c, "CUPTI_ACTIVITY_KIND_KERNEL"):
        return []
    ks = c.execute("SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s"
                   " ON s.id=k.shortName ORDER BY k.start").fetchall()
    if not ks:
        return []
    t0, b = ks[0][0], int(bin_ms * 1e6)
    bins: dict[int, dict] = {}
    for s, e, n in ks:
        cat = category(n)
        t = s
        while t < e:
            i = (t - t0) // b
            edge = t0 + (i + 1) * b
            seg = min(e, edge) - t
            d = bins.setdefault(i, {})
            d[cat] = d.get(cat, 0) + seg
            t = min(e, edge)
    out = []
    for i in sorted(bins):
        row = {"t_ms": i * bin_ms}
        row.update({k: v / b for k, v in bins[i].items()})
        out.append(row)
    return out


def store_profile(conn, run_id: int, summary: dict) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS gpu_profile (run_id INTEGER PRIMARY KEY, json TEXT NOT NULL)")
    conn.execute("INSERT OR REPLACE INTO gpu_profile (run_id, json) VALUES (?,?)", (run_id, json.dumps(summary)))
