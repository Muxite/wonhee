"""gauge doctor | build | prepare | run | sweep | corun | preflight | report | export"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path

import yaml

from gauge import store

ROOT = Path(__file__).resolve().parents[1]


def _kv(pairs: list[str]) -> dict:
    out = {}
    for s in pairs:
        k, _, v = s.partition("=")
        out[k] = yaml.safe_load(v)
    return out


def _fmt(s: dict) -> str:
    g = lambda k, d=2**30, u="GiB": "-" if s.get(k) is None else f"{s[k] / d:.2f}{u}"  # noqa: E731
    return (f"{s.get('failure_class')}  runtime {s.get('runtime_s', 0):.2f}s  cpu {s.get('cpu_time_s') or 0:.1f}s"
            f"  ram {g('ram_peak_bytes')}  vram {g('vram_peak_bytes')}  samples {s.get('sample_count')}"
            f" (missed {s.get('missed_samples')})")


def cmd_doctor(a) -> int:
    from gauge.probe.gpu import Gpu

    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"{'ok ' if cond else 'BAD'} {name} {detail}")

    check("cgroup v2", Path("/sys/fs/cgroup/cgroup.controllers").exists())
    info = subprocess.run(["docker", "info", "--format", "{{json .}}"], capture_output=True, text=True)
    d = json.loads(info.stdout) if info.returncode == 0 else {}
    check("docker reachable", bool(d), d.get("ServerVersion", info.stderr.strip()[:80]))
    check("docker cgroup driver systemd", d.get("CgroupDriver") == "systemd", d.get("CgroupDriver", ""))
    check("nvidia runtime", "nvidia" in d.get("Runtimes", {}))
    g = Gpu()
    check("nvml", g.ok, json.dumps(g.info()))
    if g.ok:
        g.device(), g.energy_mj(), g.processes()
        for k, v in g.capabilities.items():
            print(f"     {k}: {v}")
    from gauge.runner import GPU_LOCK

    check("gpu-lock", Path(GPU_LOCK).exists(), GPU_LOCK)
    return 0 if ok else 1


def cmd_build(a) -> int:
    from gauge.workloads.base import REGISTRY, get

    get("stress.cpu")
    done = set()
    for w in REGISTRY.values():
        if w.build and w.build not in done:
            done.add(w.build)
            print(f"building {w.image} from images/{w.build}")
            ctx = w.build_context or str(ROOT / "images" / w.build)
            subprocess.run(["docker", "build", "-t", w.image, "-f", str(ROOT / "images" / w.build / "Dockerfile"), ctx],
                           check=True)
    return 0


def cmd_prepare(a) -> int:
    from gauge.workloads.media import prepare_media

    print(prepare_media())
    return 0


def _spec(workload: str, params: dict, a):
    from gauge.spec import JobSpec

    return JobSpec(workload=workload, params=params, warmness=getattr(a, "warmness", "unknown"),
                   timeout_s=getattr(a, "timeout", 1800), mem_limit=getattr(a, "mem", None))


def cmd_run(a) -> int:
    from gauge import runner

    conn = store.connect(a.db)
    rid, s = runner.run(conn, _spec(a.workload, _kv(a.param), a), interval_s=a.interval, lock=not a.no_lock)
    print(f"run {rid}: {_fmt(s)}")
    if s.get("failure_class") != "success" and s.get("log_tail"):
        print("\n".join(s["log_tail"]))
    if a.json:
        print(json.dumps(s, indent=2))
    return 0


def expand(sweep: dict) -> list[tuple[str, dict]]:
    """sweep yaml: {jobs: [{workload, fixed: {...}, grid: {param: [values]}, repeats: n}]}"""
    out = []
    for j in sweep["jobs"]:
        grid = j.get("grid", {})
        keys = list(grid)
        for combo in itertools.product(*(grid[k] for k in keys)) if keys else [()]:
            params = {**j.get("fixed", {}), **dict(zip(keys, combo))}
            out += [(j["workload"], params)] * j.get("repeats", 1)
    return out


def cmd_sweep(a) -> int:
    from gauge import runner

    conn = store.connect(a.db)
    jobs = expand(yaml.safe_load(Path(a.file).read_text()))
    print(f"{len(jobs)} jobs")
    for i, (wl, params) in enumerate(jobs, 1):
        try:
            rid, s = runner.run(conn, _spec(wl, params, a), interval_s=a.interval, lock=not a.no_lock)
            print(f"[{i}/{len(jobs)}] run {rid} {wl} {params}: {_fmt(s)}", flush=True)
        except Exception as e:  # one broken job must not sink the sweep
            print(f"[{i}/{len(jobs)}] {wl} {params}: ERROR {type(e).__name__}: {e}", flush=True)
    return 0


def _job_arg(item: str, a):
    """'workload:k=v,k=v' (a value may itself contain ':', e.g. model=qwen2.5:3b)."""
    wl, _, rest = item.partition(":")
    return _spec(wl, _kv(rest.split(",")) if rest else {}, a)


def cmd_corun(a) -> int:
    from gauge import runner

    conn = store.connect(a.db)
    groups = [a.job] if a.job else yaml.safe_load(Path(a.pairs).read_text())["pairs"]
    for g in groups:
        for _ in range(a.repeats):
            for rid, s in runner.corun(conn, [_job_arg(i, a) for i in g], interval_s=a.interval, lock=not a.no_lock):
                print(f"run {rid}: {_fmt(s)}", flush=True)
    return 0


def cmd_preflight(a) -> int:
    from gauge.preflight import preflight

    conn = store.connect(a.db)
    print(json.dumps(preflight(conn, _spec(a.workload, _kv(a.param), a)), indent=2))
    return 0


def cmd_report(a) -> int:
    from gauge.report import write_report

    print(write_report(store.connect(a.db), Path(a.out)))
    return 0


def cmd_export(a) -> int:
    conn = store.connect(a.db)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for table in ("run", "prediction", "summary", "evaluation", "sample"):
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        with open(out / f"{table}.csv", "w", newline="") as f:
            wr = csv.writer(f)
            if rows:
                wr.writerow(rows[0].keys())
                wr.writerows(tuple(r) for r in rows)
        print(f"{table}: {len(rows)} rows")
    return 0


def _term(signum, frame):
    raise KeyboardInterrupt  # unwind through finally blocks: release gpu-lock, remove container


def main(argv=None) -> int:
    import signal

    signal.signal(signal.SIGTERM, _term)
    ap = argparse.ArgumentParser(prog="gauge")
    ap.add_argument("--db", default=str(store.DEFAULT_DB))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)
    sub.add_parser("build").set_defaults(fn=cmd_build)
    sub.add_parser("prepare").set_defaults(fn=cmd_prepare)

    def runopts(p):
        p.add_argument("--interval", type=float, default=0.1)
        p.add_argument("--no-lock", action="store_true")
        p.add_argument("--timeout", type=float, default=1800)
        p.add_argument("--mem", default=None, help="docker --memory override, e.g. 4g")
        p.add_argument("--warmness", default="unknown", choices=["cold", "warm", "unknown"])

    p = sub.add_parser("run")
    p.add_argument("workload")
    p.add_argument("param", nargs="*", help="key=value")
    p.add_argument("--json", action="store_true")
    runopts(p)
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("sweep")
    p.add_argument("file")
    runopts(p)
    p.set_defaults(fn=cmd_sweep)
    p = sub.add_parser("corun", help="gauge corun 'train.small:batch=64' 'llm.generate:model=qwen2.5:3b'")
    p.add_argument("job", nargs="*")
    p.add_argument("--pairs", help="yaml with pairs: [[job, job], ...]")
    p.add_argument("--repeats", type=int, default=1)
    runopts(p)
    p.set_defaults(fn=cmd_corun)
    p = sub.add_parser("preflight")
    p.add_argument("workload")
    p.add_argument("param", nargs="*")
    p.set_defaults(fn=cmd_preflight)
    p = sub.add_parser("report")
    p.add_argument("--out", default="reports/report.html")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("export")
    p.add_argument("--out", default="data/export")
    p.set_defaults(fn=cmd_export)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
