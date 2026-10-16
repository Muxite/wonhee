"""cgroup v2 reader for one container scope. Totals come from cumulative counters, not sampled rates."""
from __future__ import annotations

from pathlib import Path

CGROUP_ROOT = Path("/sys/fs/cgroup")


def scope_path(container_id: str, root: Path = CGROUP_ROOT) -> Path | None:
    """systemd driver puts containers at <slice>/docker-<id>.scope; cgroupfs driver at docker/<id>."""
    for cand in (root / "system.slice" / f"docker-{container_id}.scope", root / "docker" / container_id):
        if cand.is_dir():
            return cand
    hits = list(root.glob(f"*.slice/docker-{container_id}.scope"))
    return hits[0] if hits else None


def _kv(text: str) -> dict[str, int]:
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            out[parts[0]] = int(parts[1])
    return out


def parse_pressure(text: str) -> dict[str, int]:
    """'some avg10=0.00 avg60=0.00 avg300=0.00 total=123' -> {'some_total_usec': 123, ...}"""
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        for field in parts[1:]:
            k, _, v = field.partition("=")
            if k == "total":
                out[f"{parts[0]}_total_usec"] = int(v)
    return out


def parse_io(text: str) -> dict[str, int]:
    """Sum io.stat across devices: '8:0 rbytes=1 wbytes=2 rios=3 wios=4 ...'."""
    tot = {"rbytes": 0, "wbytes": 0, "rios": 0, "wios": 0}
    for line in text.splitlines():
        for field in line.split()[1:]:
            k, _, v = field.partition("=")
            if k in tot:
                tot[k] += int(v)
    return tot


class Cgroup:
    def __init__(self, path: Path):
        self.path = Path(path)

    def _read(self, name: str) -> str | None:
        try:
            return (self.path / name).read_text()
        except (FileNotFoundError, ProcessLookupError, OSError):
            return None

    def _int(self, name: str) -> int | None:
        t = self._read(name)
        if t is None:
            return None
        t = t.strip()
        return None if t == "max" or not t else int(t)

    def fast(self) -> dict | None:
        """Cheap per-sample read. None once the cgroup is gone."""
        cpu = self._read("cpu.stat")
        if cpu is None:
            return None
        stat = _kv(self._read("memory.stat") or "")
        io = parse_io(self._read("io.stat") or "")
        return {"cpu_usage_usec": _kv(cpu).get("usage_usec"), "mem_current": self._int("memory.current"),
                "mem_anon": stat.get("anon"), "mem_file": stat.get("file"),
                "swap_current": self._int("memory.swap.current"),
                "io_rbytes": io["rbytes"], "io_wbytes": io["wbytes"]}

    def full(self) -> dict | None:
        """Start/end snapshot of every cumulative counter and kernel high-water mark we use."""
        cpu = self._read("cpu.stat")
        if cpu is None:
            return None
        snap = {"cpu": _kv(cpu), "io": parse_io(self._read("io.stat") or ""),
                "memory_events": _kv(self._read("memory.events") or ""),
                "memory_current": self._int("memory.current"), "memory_peak": self._int("memory.peak"),
                "swap_peak": self._int("memory.swap.peak"), "memory_max": self._int("memory.max"),
                "pids_peak": self._int("pids.peak")}
        for res in ("cpu", "memory", "io"):
            snap[f"{res}_pressure"] = parse_pressure(self._read(f"{res}.pressure") or "")
        return snap

    def pids(self) -> list[int]:
        """Every pid in this scope and its descendant cgroups."""
        out = []
        for f in [self.path / "cgroup.procs", *self.path.glob("**/cgroup.procs")]:
            try:
                out += [int(x) for x in f.read_text().split()]
            except OSError:
                pass
        return sorted(set(out))
