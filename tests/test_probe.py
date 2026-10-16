from types import SimpleNamespace

from gauge.probe.cgroup import Cgroup, parse_io, parse_pressure, scope_path
from gauge.probe.gpu import Gpu, in_scope


def fake_scope(tmp_path, cid="abc"):
    s = tmp_path / "system.slice" / f"docker-{cid}.scope"
    (s / "child").mkdir(parents=True)
    files = {
        "cpu.stat": "usage_usec 4000000\nuser_usec 3000000\nsystem_usec 1000000\nthrottled_usec 0\n",
        "memory.current": "1048576\n", "memory.peak": "2097152\n", "memory.swap.peak": "0\n", "memory.max": "max\n",
        "memory.swap.current": "0\n", "pids.peak": "7\n",
        "memory.stat": "anon 700000\nfile 300000\nkernel 1\n",
        "memory.events": "low 0\nhigh 0\nmax 2\noom 1\noom_kill 1\n",
        "io.stat": "8:0 rbytes=100 wbytes=50 rios=2 wios=1 dbytes=0 dios=0\n259:0 rbytes=1 wbytes=1 rios=1 wios=1\n",
        "cpu.pressure": "some avg10=0.00 avg60=0.00 avg300=0.00 total=10\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=5\n",
        "memory.pressure": "some avg10=0 avg60=0 avg300=0 total=0\nfull avg10=0 avg60=0 avg300=0 total=0\n",
        "io.pressure": "some avg10=0 avg60=0 avg300=0 total=3\n",
        "cgroup.procs": "11\n12\n",
    }
    for k, v in files.items():
        (s / k).write_text(v)
    (s / "child" / "cgroup.procs").write_text("13\n")
    return s


def test_cgroup_reads(tmp_path):
    s = fake_scope(tmp_path)
    assert scope_path("abc", tmp_path) == s
    cg = Cgroup(s)
    fast = cg.fast()
    assert fast["cpu_usage_usec"] == 4_000_000 and fast["mem_anon"] == 700000 and fast["io_rbytes"] == 101
    full = cg.full()
    assert full["memory_peak"] == 2097152 and full["memory_max"] is None  # "max" -> unlimited
    assert full["memory_events"]["oom_kill"] == 1
    assert full["cpu_pressure"] == {"some_total_usec": 10, "full_total_usec": 5}
    assert cg.pids() == [11, 12, 13]  # descendants included


def test_cgroup_gone_is_none(tmp_path):
    assert Cgroup(tmp_path / "nope").fast() is None


def test_parsers():
    assert parse_io("8:0 rbytes=5 wbytes=6\n8:1 rbytes=1 wbytes=0")["rbytes"] == 6
    assert parse_pressure("some avg10=1.0 total=42") == {"some_total_usec": 42}


def fake_proc(tmp_path, mapping):
    for pid, cg in mapping.items():
        (tmp_path / str(pid)).mkdir()
        (tmp_path / str(pid) / "cgroup").write_text(f"0::{cg}\n")
    return tmp_path


def test_in_scope_includes_descendants(tmp_path):
    proc = fake_proc(tmp_path, {1: "/system.slice/docker-abc.scope", 2: "/system.slice/docker-abc.scope/x",
                                3: "/system.slice/docker-abcd.scope"})
    assert in_scope(1, "/system.slice/docker-abc.scope", proc)
    assert in_scope(2, "/system.slice/docker-abc.scope", proc)
    assert not in_scope(3, "/system.slice/docker-abc.scope", proc)  # prefix of a different id
    assert not in_scope(99, "/system.slice/docker-abc.scope", proc)


class FakeNvml:
    NVML_CLOCK_SM = 1
    NVML_TEMPERATURE_GPU = 0

    def __init__(self, procs, energy_raises=False):
        self.procs, self.energy_raises = procs, energy_raises

    def nvmlInit(self): pass
    def nvmlDeviceGetHandleByIndex(self, i): return "h"
    def nvmlDeviceGetComputeRunningProcesses(self, h): return self.procs
    def nvmlDeviceGetGraphicsRunningProcesses(self, h): return []
    def nvmlDeviceGetUtilizationRates(self, h): return SimpleNamespace(gpu=50, memory=20)
    def nvmlDeviceGetMemoryInfo(self, h): return SimpleNamespace(used=4, free=12, total=16)
    def nvmlDeviceGetPowerUsage(self, h): return 150_000

    def nvmlDeviceGetTotalEnergyConsumption(self, h):
        if self.energy_raises:
            raise RuntimeError("not supported")
        return 1000


def test_gpu_scope_vram_and_capabilities(tmp_path):
    proc = fake_proc(tmp_path, {10: "/system.slice/docker-abc.scope", 20: "/user.slice/x"})
    nv = FakeNvml([SimpleNamespace(pid=10, usedGpuMemory=500), SimpleNamespace(pid=20, usedGpuMemory=900)],
                  energy_raises=True)
    g = Gpu(nv)
    assert g.scope_vram("/system.slice/docker-abc.scope", proc) == (500, [10])
    assert g.energy_mj() is None and g.capabilities["energy"].startswith("unsupported")
    assert g.energy_mj() is None  # not retried once known unsupported
    d = g.device()
    assert d["dev_util"] == 50.0 and d["dev_power_w"] == 150.0
    assert "dev_sm_mhz" not in d  # FakeNvml lacks it: absent, not zero


def test_gpu_unknown_process_memory(tmp_path):
    proc = fake_proc(tmp_path, {10: "/system.slice/docker-abc.scope"})
    g = Gpu(FakeNvml([SimpleNamespace(pid=10, usedGpuMemory=2**64 - 1)]))  # NVML_VALUE_NOT_AVAILABLE
    assert g.scope_vram("/system.slice/docker-abc.scope", proc) == (None, [10])


def test_no_nvml():
    class Broken:
        def nvmlInit(self):
            raise OSError("no driver")
    g = Gpu(Broken())
    assert not g.ok and g.device() == {} and g.processes() == []
