import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("capacity", Path(__file__).parents[2] / "profiling/analysis/capacity.py")
cap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cap)


def _rung(ladder, level, rate, **res):
    return {"ladder": ladder, "level": level, "rate": rate, "failed": None, "unit": "x/s", "res": res}


def test_plateau_and_single_limiter():
    rs = [_rung("L", 1, 100, power=0.5), _rung("L", 2, 190, power=0.8), _rung("L", 4, 200, power=0.95, sm_busy=0.7),
          _rung("L", 8, 205, power=0.97)]
    d = cap.summarize(rs)["L"]
    assert d["plateau_level"] == 2  # 2 -> 4 adds 5% for one doubling
    assert d["limiter"] == "software / serialization"  # judged at the plateau rung: power 0.8 < 0.9


def test_limiter_named_only_when_alone():
    rs = [_rung("A", 1, 100, power=0.95), _rung("A", 2, 101, power=0.96)]
    assert cap.summarize(rs)["A"]["limiter"] == "power"
    rs = [_rung("B", 1, 100, power=0.95, vram=0.92), _rung("B", 2, 101, power=0.96, vram=0.95)]
    assert cap.summarize(rs)["B"]["limiter"].startswith("co-limited")


def test_failure_rung_is_the_wall_and_no_plateau_judges_last():
    rs = [_rung("C", 1, 100), _rung("C", 2, 200), {**_rung("C", 4, None), "failed": "cuda_oom"}]
    d = cap.summarize(rs)["C"]
    assert d["oom_or_failure_at"] == 4 and d["plateau_level"] is None and d["judged_at_level"] == 2


def test_clock_is_not_a_limiter_and_capped_power_counts():
    rs = [_rung("D", 1, 100, sm_clock=0.95, power=0.5), _rung("D", 2, 101, sm_clock=0.95, power=0.5)]
    assert cap.summarize(rs)["D"]["limiter"] == "software / serialization"
    rs = [_rung("E", 1, 100, power=0.84, power_capped_share=0.8), _rung("E", 2, 101, power=0.84, power_capped_share=0.8)]
    assert cap.summarize(rs)["E"]["limiter"] == "power"
