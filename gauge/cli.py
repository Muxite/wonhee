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
from gauge import paths



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
    if getattr(a, "mps", False):
        from gauge import mps

        try:
            mps.start()
            check("mps daemon", True, mps.NAME)
        except Exception as e:  # noqa: BLE001
            check("mps daemon", False, str(e)[:200])
        finally:
            mps.stop()
    return 0 if ok else 1


def cmd_build(a) -> int:
    from gauge.workloads.base import REGISTRY, get

    get("stress.cpu")
    done = set()
    for w in REGISTRY.values():
        if w.build and w.build not in done:
            done.add(w.build)
            print(f"building {w.image} from profiling/images/{w.build}")
            ctx = w.build_context or str(paths.IMAGES / w.build)
            subprocess.run(["docker", "build", "-t", w.image, "-f", str(paths.IMAGES / w.build / "Dockerfile"), ctx],
                           check=True)
    return 0


def cmd_prepare(a) -> int:
    from gauge.workloads.media import prepare_media

    print(prepare_media())
    return 0


def _spec(workload: str, params: dict, a):
    from gauge.spec import JobSpec

    return JobSpec(workload=workload, params=params, warmness=getattr(a, "warmness", "unknown"),
                   timeout_s=getattr(a, "timeout", 1800), mem_limit=getattr(a, "mem", None),
                   profile=getattr(a, "profile", "none"), sharing=getattr(a, "sharing", "timeslice"))


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
    """sweep yaml: {experiment?, jobs: [{workload, fixed: {...}, grid: {param: [values]}, repeats: n,
    profile?, sharing?, experiment?}]}. `experiment` labels the runs in the dataset (gauge/dataset.py)."""
    out = []
    for j in sweep["jobs"]:
        grid = j.get("grid", {})
        keys = list(grid)
        for combo in itertools.product(*(grid[k] for k in keys)) if keys else [()]:
            params = {**j.get("fixed", {}), **dict(zip(keys, combo))}
            over = {k: j[k] for k in ("profile", "sharing", "interval") if j.get(k)}  # interval: sampler, s
            if j.get("experiment") or sweep.get("experiment"):  # dataset label; provenance is not hashed
                over["provenance"] = {"experiment": j.get("experiment") or sweep["experiment"]}
            out += [(j["workload"], params, over)] * j.get("repeats", 1)
    return out


def order(jobs: list, how: str = "interleave", seed: int | None = None) -> list:
    """file: as written (repeats back to back). interleave: repeat 1 of every condition, then repeat 2, ...; with a
    seed, each round is shuffled. Back-to-back repeats confound a condition with time, temperature and cache
    state (the first repeat of a model-loading job reads from disk, the next from page cache)."""
    if how == "file":
        return list(jobs)
    import random

    rounds: list[list] = []
    seen: dict = {}
    for j in jobs:
        k = json.dumps(j, sort_keys=True, default=str)
        n = seen.get(k, 0)
        seen[k] = n + 1
        if n == len(rounds):
            rounds.append([])
        rounds[n].append(j)
    out = []
    for i, r in enumerate(rounds):
        if seed is not None:
            random.Random(seed * 1000 + i).shuffle(r)
        out += r
    return out


def cmd_sweep(a) -> int:
    from gauge import runner

    conn = store.connect(a.db)
    jobs = order(expand(yaml.safe_load(Path(a.file).read_text())), a.order, a.seed)
    print(f"{len(jobs)} jobs")
    for i, (wl, params, over) in enumerate(jobs, 1):
        try:
            over = dict(over)
            interval = over.pop("interval", a.interval)
            spec = _spec(wl, params, a).model_copy(update=over)
            rid, s = runner.run(conn, spec, interval_s=interval, lock=not a.no_lock)
            print(f"[{i}/{len(jobs)}] run {rid} {wl} {params}: {_fmt(s)}", flush=True)
        except Exception as e:  # one broken job must not sink the sweep
            print(f"[{i}/{len(jobs)}] {wl} {params}: ERROR {type(e).__name__}: {e}", flush=True)
    return 0


def _job_arg(item: str, a):
    """'workload:k=v,k=v' (a value may itself contain ':', e.g. model=qwen2.5:3b)."""
    wl, _, rest = item.partition(":")
    return _spec(wl, _kv(rest.split(",")) if rest else {}, a)


def cmd_corun(a) -> int:
    conn = store.connect(a.db)
    doc = {"pairs": [a.job]} if a.job else yaml.safe_load(Path(a.pairs).read_text())
    groups = [g for g in doc["pairs"] for _ in range(doc.get("repeats", a.repeats))]
    for g in order(groups, a.order, a.seed):
        for rid, s in run_group(conn, g, doc, a):
            print(f"run {rid}: {_fmt(s)}", flush=True)
    return 0


def run_group(conn, g, doc: dict, a) -> list:
    """One entry of a corun file: a list of job strings, or {jobs: [...], profile?, sharing?} to override the
    file's settings for that group. A one-job group runs as a plain solo run, so a file can hold a contrast's
    solo references next to its co-runs and interleave them (no block of solos run first)."""
    from gauge import runner

    jobs, over = (g, {}) if isinstance(g, list) else (g["jobs"], g)
    specs = [_job_arg(i, a) for i in jobs]
    for k in ("profile", "sharing"):
        if over.get(k) or doc.get(k):
            specs = [s.model_copy(update={k: over.get(k) or doc[k]}) for s in specs]
    if doc.get("experiment"):
        specs = [s.model_copy(update={"provenance": {**s.provenance, "experiment": doc["experiment"]}})
                 for s in specs]
    if len(specs) == 1:
        return [runner.run(conn, specs[0], interval_s=a.interval, lock=not a.no_lock)]
    return runner.corun(conn, specs, interval_s=a.interval, lock=not a.no_lock)


def cmd_preflight(a) -> int:
    from gauge.preflight import preflight

    conn = store.connect(a.db)
    print(json.dumps(preflight(conn, _spec(a.workload, _kv(a.param), a)), indent=2))
    return 0


def cmd_report(a) -> int:
    from gauge.report import write_report

    print(write_report(store.connect(a.db), Path(a.out)))
    return 0


def cmd_backtest(a) -> int:
    from gauge.backtest import backtest

    f = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
    print(f"{'workload':22} {'target':16} {'predictor':10} {'n':>4} {'cens':>4} {'no_act':>6} {'cover':>5} {'mape':>6} {'med a/p':>7}")
    for r in backtest(store.connect(a.db)):
        print(f"{r['workload']:22} {r['target']:16} {r['predictor']:10} {r['n']:>4} {r['censored']:>4}"
              f" {f(r['coverage']):>5} {f(r['mape']):>6} {f(r['median_ratio']):>7}")
    return 0


def cmd_startbench(a) -> int:
    from gauge import startbench
    from gauge.runner import gpu_lock

    conn = store.connect(a.db)
    if not a.analyze_only:
        with gpu_lock("gauge startbench", not a.no_lock):
            a.session = startbench.run_bench(conn, a.reps, log=lambda m: print(m, flush=True))
    for d in startbench.analyze(conn, a.session):
        ph = lambda k: f"{d[k]['p50'] * 1000:7.0f}" if k in d else "      -"  # noqa: E731
        print(f"{d['variant']:21} {d['task']:6} n={d['n']:<4} total p50 {d['total']['p50'] * 1000:6.0f}ms"
              f" (p90 {d['total']['p90'] * 1000:5.0f})  overhead {d['overhead_vs_native_p50'] * 1000:6.0f}ms |"
              f" create{ph('client_to_created')} start{ph('created_to_started')} first{ph('started_to_first')}"
              f" task{ph('task_run')} return{ph('end_to_return')}  rm {(d['rm_p50'] or 0) * 1000:.0f}")
    return 0


def cmd_dataset(a) -> int:
    from gauge.dataset import OUT, export

    for k, v in export(store.connect(a.db)).items():
        print(f"{k}: {v} rows")
    print(OUT)
    return 0


def cmd_trace_export(a) -> int:
    from gauge.trace_export import OUT, export_run

    conn = store.connect(a.db)
    for rid in a.run_ids:
        print(export_run(conn, rid))
    print(OUT)
    return 0


def cmd_sequence(a) -> int:
    from gauge import sequence

    conn = store.connect(a.db)
    doc = yaml.safe_load(Path(a.file).read_text())
    for pair in doc["pairs"]:
        for _ in range(doc.get("repeats", 1)):
            r = sequence.run_sequence(conn, _job_arg(pair["a"], a), _job_arg(pair["b"], a),
                                      idle_s=doc.get("idle_s", 5), label=pair.get("label", ""))
            res = sequence.analyze(conn, r["id"])
            print(f"== {r['id']} {pair.get('label', '')}", flush=True)
            for w in res["windows"]:
                print(f"  {w['step']:5} {w['seconds']:5.1f}s  SM {w.get('SMs Active') or 0:5.1f}  tensor {w.get('Tensor Active') or 0:5.1f}"
                      f"  DRAMr {w.get('DRAM Read Bandwidth') or 0:5.1f}  DRAMw {w.get('DRAM Write Bandwidth') or 0:5.1f}"
                      f"  issue {w.get('SM Issue') or 0:5.1f}  runs {w['run_ids']}", flush=True)
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
    p = sub.add_parser("doctor")
    p.add_argument("--mps", action="store_true", help="also start and stop the MPS control daemon")
    p.set_defaults(fn=cmd_doctor)
    sub.add_parser("build").set_defaults(fn=cmd_build)
    sub.add_parser("prepare").set_defaults(fn=cmd_prepare)

    def runopts(p):
        p.add_argument("--interval", type=float, default=0.1)
        p.add_argument("--no-lock", action="store_true")
        p.add_argument("--timeout", type=float, default=1800)
        p.add_argument("--mem", default=None, help="docker --memory override, e.g. 4g")
        p.add_argument("--warmness", default="unknown", choices=["cold", "warm", "unknown"])
        p.add_argument("--profile", default="none", choices=["none", "trace", "metrics"],
                       help="Nsight inside the container: trace = this process's kernels; metrics = + GPU counters")
        p.add_argument("--order", default="interleave", choices=["interleave", "file"],
                       help="interleave (default): one repeat of every condition per round; file: as written")
        p.add_argument("--seed", type=int, default=None, help="with interleave: shuffle each round with this seed")
        p.add_argument("--sharing", default="timeslice", choices=["timeslice", "mps"],
                       help="mps: run the job(s) as clients of one MPS server (gauge.mps)")

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
    p.add_argument("--out", default=str(paths.DATA / "reports" / "report.html"))
    p.set_defaults(fn=cmd_report)
    sub.add_parser("backtest", help="replay current predictors over recorded runs").set_defaults(fn=cmd_backtest)
    p = sub.add_parser("startbench", help="container start overhead vs native, phase-decomposed")
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--no-lock", action="store_true")
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--session", default=None)
    p.set_defaults(fn=cmd_startbench)
    sub.add_parser("dataset", help="flat documented dataset in data/gauge/dataset").set_defaults(fn=cmd_dataset)
    p = sub.add_parser("trace-export", help="per-node execution trace: node JSON + Perfetto JSON")
    p.add_argument("run_ids", nargs="+", type=int)
    p.set_defaults(fn=cmd_trace_export)
    p = sub.add_parser("sequence", help="idle -> A -> B -> A+B under one GPU-counter observer")
    p.add_argument("file")
    p.set_defaults(fn=cmd_sequence)
    p = sub.add_parser("export")
    p.add_argument("--out", default=str(paths.DATA / "export"))
    p.set_defaults(fn=cmd_export)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
