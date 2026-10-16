"""Container start overhead: the same task natively vs in Docker, decomposed into phases.

Every timestamp is host CLOCK_REALTIME (the container shares the kernel clock):
  t_req       host, just before the client call
  created     daemon (inspect .Created)
  started     daemon (inspect .State.StartedAt): after runc + OCI hooks (nvidia) ran
  first       inside the container, first instruction (`date +%s.%N`)
  task_end    inside the container, after the task
  finished    daemon (inspect .State.FinishedAt)
  t_ret       host, client call returned
  rm_s        separate `docker rm` duration (what --rm would add)
The native baseline runs the image's own stress-ng through its own musl loader, so the task binary is identical.
Variants are interleaved in random order every rep, so drift (thermal, background load) hits all of them alike.
"""
from __future__ import annotations

import http.client
import json
import random
import socket
import struct
import subprocess
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import quote

from gauge.docker_run import _ts

IMAGE = "gauge/startbench:1"
BIG_IMAGE = "ollama/ollama:0.35.0"  # 9.3 GB, ubuntu: image size effect
TORCH_IMAGE = "gauge/torch:1"       # 12 GB, ubuntu + CUDA libs
ROOTFS = Path(__file__).resolve().parents[1] / "data" / "rootfs" / "startbench"
STRESS_OPS = 400  # one matrixprod worker; roughly 1 s on this box

SCHEMA = """
CREATE TABLE IF NOT EXISTS startbench (
  id INTEGER PRIMARY KEY, session TEXT, variant TEXT, task TEXT, burst INTEGER, rep INTEGER,
  t_req REAL, created REAL, started REAL, first REAL, task_end REAL, finished REAL, t_ret REAL, rm_s REAL,
  error TEXT
);
"""

NOOP = "date +%s.%N; date +%s.%N"
STRESS = f"date +%s.%N; stress-ng --cpu 1 --cpu-method matrixprod --cpu-ops {STRESS_OPS} --quiet; date +%s.%N"


def _stamps(out: str) -> tuple[float | None, float | None]:
    vals = []
    for line in out.split():
        try:
            vals.append(float(line))
        except ValueError:
            pass
    return (vals[0], vals[-1]) if len(vals) >= 2 else (None, None)


def _inspect(name: str) -> dict:
    r = subprocess.run(["docker", "inspect", name], capture_output=True, text=True)
    if r.returncode:
        return {}
    d = json.loads(r.stdout)[0]
    return {"created": _ts(d.get("Created")), "started": _ts(d["State"].get("StartedAt")),
            "finished": _ts(d["State"].get("FinishedAt"))}


def _rm(name: str) -> float:
    t = time.time()
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    return time.time() - t


# ---- variants: each returns a row dict ----

def native(task: str) -> dict:
    r = ROOTFS
    stress = (f"{r}/lib/ld-musl-x86_64.so.1 --library-path {r}/lib:{r}/usr/lib {r}/usr/bin/stress-ng "
              f"--cpu 1 --cpu-method matrixprod --cpu-ops {STRESS_OPS} --quiet")
    cmd = "date +%s.%N; date +%s.%N" if task == "noop" else f"date +%s.%N; {stress}; date +%s.%N"
    t_req = time.time()
    out = subprocess.run(["sh", "-c", cmd], capture_output=True, text=True).stdout
    t_ret = time.time()
    first, end = _stamps(out)
    return {"t_req": t_req, "first": first, "task_end": end, "t_ret": t_ret}


def cli_run(task: str, *, image=IMAGE, net="none", gpu=False) -> dict:
    name = f"gauge-sb-{uuid.uuid4().hex[:10]}"
    argv = ["docker", "run", "--name", name, "--label", "gauge-startbench=1", "--entrypoint", "/bin/sh"]
    argv += ["--network", net] if net else []
    argv += ["--gpus", "all"] if gpu else []
    cmd = NOOP if task == "noop" else STRESS
    if image != IMAGE:
        cmd = NOOP  # big images lack stress-ng; noop only
    t_req = time.time()
    r = subprocess.run(argv + [image, "-c", cmd], capture_output=True, text=True)
    t_ret = time.time()
    first, end = _stamps(r.stdout)
    row = {"t_req": t_req, "first": first, "task_end": end, "t_ret": t_ret, **_inspect(name), "rm_s": _rm(name)}
    if r.returncode:
        row["error"] = r.stderr.strip()[:300]
    return row


def cli_start_precreated(task: str) -> dict:
    """Create ahead of time (untimed), then time only `docker start -a`: a pre-provisioned node."""
    name = f"gauge-sb-{uuid.uuid4().hex[:10]}"
    subprocess.run(["docker", "create", "--name", name, "--label", "gauge-startbench=1", "--network", "none",
                    "--entrypoint", "/bin/sh", IMAGE, "-c", NOOP if task == "noop" else STRESS],
                   capture_output=True, check=True)
    t_req = time.time()
    r = subprocess.run(["docker", "start", "-a", name], capture_output=True, text=True)
    t_ret = time.time()
    first, end = _stamps(r.stdout)
    ins = _inspect(name)
    ins.pop("created", None)  # creation happened before t_req
    return {"t_req": t_req, "first": first, "task_end": end, "t_ret": t_ret, **ins, "rm_s": _rm(name)}


class WarmPool:
    """One long-lived container; tasks arrive with `docker exec` (the warm-worker alternative)."""

    def __init__(self):
        self.name = f"gauge-sb-warm-{uuid.uuid4().hex[:8]}"
        subprocess.run(["docker", "run", "-d", "--name", self.name, "--label", "gauge-startbench=1", "--network",
                        "none", "--entrypoint", "/bin/sh", IMAGE, "-c", "sleep infinity"], capture_output=True,
                       check=True)

    def __call__(self, task: str) -> dict:
        t_req = time.time()
        r = subprocess.run(["docker", "exec", self.name, "sh", "-c", NOOP if task == "noop" else STRESS],
                           capture_output=True, text=True)
        t_ret = time.time()
        first, end = _stamps(r.stdout)
        return {"t_req": t_req, "first": first, "task_end": end, "t_ret": t_ret}

    def close(self):
        _rm(self.name)


class _Unix(http.client.HTTPConnection):
    def __init__(self, path="/var/run/docker.sock"):
        super().__init__("localhost", timeout=120)
        self.path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.path)


def _api(method: str, path: str, body=None):
    c = _Unix()
    c.request(method, path, body=json.dumps(body) if body is not None else None,
              headers={"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


def _demux(raw: bytes) -> str:
    """Docker log stream frames: 8-byte header (stream, 0,0,0, big-endian size) + payload."""
    out, i = [], 0
    while i + 8 <= len(raw):
        size = struct.unpack(">I", raw[i + 4:i + 8])[0]
        out.append(raw[i + 8:i + 8 + size])
        i += 8 + size
    return b"".join(out).decode(errors="replace")


def api_run(task: str) -> dict:
    """Engine API over the unix socket, no CLI process: what an orchestrator would call."""
    name = f"gauge-sb-{uuid.uuid4().hex[:10]}"
    spec = {"Image": IMAGE, "Entrypoint": ["/bin/sh"], "Cmd": ["-c", NOOP if task == "noop" else STRESS],
            "Labels": {"gauge-startbench": "1"}, "HostConfig": {"NetworkMode": "none"}}
    t_req = time.time()
    st, data = _api("POST", f"/containers/create?name={quote(name)}", spec)
    if st != 201:
        return {"t_req": t_req, "error": f"create {st}: {data[:200]!r}"}
    cid = json.loads(data)["Id"]
    _api("POST", f"/containers/{cid}/start")
    _api("POST", f"/containers/{cid}/wait")
    _, logs = _api("GET", f"/containers/{cid}/logs?stdout=1")
    t_ret = time.time()
    first, end = _stamps(_demux(logs))
    return {"t_req": t_req, "first": first, "task_end": end, "t_ret": t_ret, **_inspect(name), "rm_s": _rm(name)}


def burst(n: int, task: str) -> list[dict]:
    """n `docker run`s launched at once: per-container latency under fan-out."""
    rows: list[dict] = [None] * n  # type: ignore[list-item]
    gate = threading.Barrier(n)

    def one(i):
        gate.wait()
        rows[i] = cli_run(task)

    ts = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return rows


def run_bench(conn, reps: int, bursts=(4, 8, 16), log=print) -> str:
    conn.executescript(SCHEMA)
    session = uuid.uuid4().hex[:8]
    pool = WarmPool()
    variants = {
        "native": native,
        "cli_run_bridge": lambda t: cli_run(t, net="bridge"),
        "cli_run_nonet": cli_run,
        "cli_run_gpu": lambda t: cli_run(t, gpu=True),
        "cli_run_big_image": lambda t: cli_run(t, image=BIG_IMAGE),
        "cli_run_torch_gpu": lambda t: cli_run(t, image=TORCH_IMAGE, gpu=True),
        "cli_start_precreated": cli_start_precreated,
        "api_run_nonet": api_run,
        "exec_warm": pool,
    }
    big = {"cli_run_big_image", "cli_run_torch_gpu"}
    jobs = [(v, t) for v in variants for t in ("noop", "stress") if not (v in big and t == "stress")]
    jobs += [(f"burst{n}", "noop") for n in bursts]

    def put(variant, task, n, rep, row):
        conn.execute("INSERT INTO startbench (session, variant, task, burst, rep, t_req, created, started, first,"
                     " task_end, finished, t_ret, rm_s, error) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (session, variant, task, n, rep, row.get("t_req"), row.get("created"), row.get("started"),
                      row.get("first"), row.get("task_end"), row.get("finished"), row.get("t_ret"),
                      row.get("rm_s"), row.get("error")))

    try:
        for rep in range(reps):
            random.shuffle(jobs)
            for v, t in jobs:
                if v.startswith("burst"):
                    n = int(v[5:])
                    for row in burst(n, t):
                        put(v, t, n, rep, row)
                else:
                    put(v, t, 1, rep, variants[v](t))
            log(f"rep {rep + 1}/{reps} done")
    finally:
        pool.close()
        subprocess.run("docker ps -aq --filter label=gauge-startbench=1 | xargs -r docker rm -f", shell=True,
                       capture_output=True)
    return session


PHASES = (("client_to_created", "t_req", "created"), ("created_to_started", "created", "started"),
          ("started_to_first", "started", "first"), ("req_to_first", "t_req", "first"),
          ("task_run", "first", "task_end"), ("end_to_return", "task_end", "t_ret"))


def analyze(conn, session: str | None = None) -> list[dict]:
    import statistics

    q = "SELECT * FROM startbench WHERE error IS NULL" + (" AND session=?" if session else "")
    rows = conn.execute(q, (session,) if session else ()).fetchall()
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["variant"], r["task"]), []).append(r)
    native_total = {t: statistics.median([x["t_ret"] - x["t_req"] for x in groups.get(("native", t), [])] or [0])
                    for t in ("noop", "stress")}
    out = []
    for (v, t), rs in sorted(groups.items()):
        d = {"variant": v, "task": t, "n": len(rs)}
        for name, a, b in PHASES:
            vals = sorted(r[b] - r[a] for r in rs if r[a] is not None and r[b] is not None)
            if vals:
                d[name] = {"p10": vals[len(vals) // 10], "p50": statistics.median(vals), "p90": vals[int(len(vals) * .9)]}
        tot = sorted(r["t_ret"] - r["t_req"] for r in rs)
        d["total"] = {"p10": tot[len(tot) // 10], "p50": statistics.median(tot), "p90": tot[int(len(tot) * .9)]}
        rm = [r["rm_s"] for r in rs if r["rm_s"] is not None]
        d["rm_p50"] = statistics.median(rm) if rm else None
        d["overhead_vs_native_p50"] = d["total"]["p50"] - native_total[t]
        out.append(d)
    return out
