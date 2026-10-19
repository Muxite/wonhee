"""gauge additions for profiling round 2: phases from marks, MPS sharing, CUVID decode, experiment labels."""
import pytest

from gauge import mps
from gauge.dataset import _experiment
from gauge.docker_run import docker_argv
from gauge.spec import JobSpec
from gauge.summarize import phases
from gauge.workloads.base import get


def test_phases_split_load_compute_tail():
    marks = [{"mark": "model_loaded", "t": 5.0}, {"mark": "compute_started", "t": 5.1},
             {"mark": "compute_finished", "t": 8.1}, {"mark": "compute_started", "t": 8.2},
             {"mark": "compute_finished", "t": 9.2}]
    ph = phases(marks, runtime_s=10.0)
    assert ph["load"] == 5.0 and ph["spans"] == pytest.approx([3.0, 1.0])
    assert ph["compute"] == pytest.approx(4.0) and ph["tail"] == pytest.approx(0.8)


def test_phases_without_marks_are_unknown_not_zero():
    assert phases([], 3.0) == {"load": None, "compute": None, "spans": [], "tail": None}


def test_sharing_changes_the_hash_but_old_hashes_stay():
    base = JobSpec(workload="train.small", params={"batch": 64})
    assert base.spec_hash() == JobSpec(workload="train.small", params={"batch": 64}, sharing="timeslice").spec_hash()
    assert base.spec_hash() != base.model_copy(update={"sharing": "mps"}).spec_hash()


def test_mps_client_args_reach_docker():
    mounts, env, extra = mps.client_args()
    argv = docker_argv("n", "img", ["true"], cidfile="/tmp/c", gpu=True, mem_limit="1g", mounts=mounts, env=env,
                       extra=extra)
    assert "--ipc=host" in argv and f"CUDA_MPS_PIPE_DIRECTORY={mps.PIPE}" in argv
    assert argv.index("--ipc=host") < argv.index("img")  # docker options go before the image


def test_cuvid_keeps_frames_on_the_gpu():
    w = get("ffmpeg.transcode")
    cmd = w.command(w.validate({"codec": "h264_nvenc", "preset": "p5", "decode": "cuvid", "height": 720}))
    assert cmd[cmd.index("-hwaccel") + 1] == "cuda" and "scale_cuda=-2:720" in cmd
    with pytest.raises(ValueError):
        w.command(w.validate({"codec": "libx264", "decode": "cuvid"}))


def test_batched_prior_grows_with_context():
    w = get("llamacpp.batched")
    small = w.prior(w.validate({"model": "qwen2.5:7b", "parallel": "1,2"}))["vram_peak_bytes"]
    big = w.prior(w.validate({"model": "qwen2.5:7b", "parallel": "1,64", "prompt_tokens": 2048}))["vram_peak_bytes"]
    assert big > small + 5 * 2**30  # the KV cache for 64 x 2k tokens


@pytest.mark.parametrize("spec,corun,label", [
    ({"workload": "wonhee.gpu", "params": {"task": "mask"}}, None, "X8 new nodes"),
    ({"workload": "wonhee.gpu", "params": {"task": "mask", "hold": 60}}, "g", "X9 resident"),
    ({"workload": "wonhee.gpu", "params": {"task": "separate"}}, "g", "X9 pairs"),
    ({"workload": "ffmpeg.transcode", "params": {"decode": "cuvid"}}, None, "X10 cuvid"),
    ({"workload": "llamacpp.phase", "params": {}, "sharing": "mps"}, "g", "X11 mps"),
    ({"workload": "llamacpp.batched", "params": {"parallel": "1,16,64"}}, None, "X12 batch limits"),
    ({"workload": "llamacpp.batched", "params": {"parallel": "1,2,4,8"}}, None, "X4 batched"),
    ({"workload": "train.small", "params": {}}, "g", "X4/X5 co-run"),
])
def test_experiment_labels(spec, corun, label):
    assert _experiment(spec, corun) == label


def test_explicit_experiment_label_wins():
    spec = {"workload": "ffmpeg.transcode", "params": {}, "provenance": {"experiment": "X10 cuvid"}}
    assert _experiment(spec, "g") == "X10 cuvid"
    assert _experiment({"workload": "ffmpeg.transcode", "params": {}}, "g") == "X4/X5 co-run"  # old runs keep theirs


def test_sweep_experiment_key_reaches_provenance():
    from gauge.cli import expand
    jobs = expand({"experiment": "X8 new nodes", "jobs": [{"workload": "wonhee.gpu", "grid": {"task": ["mask"]}}]})
    assert jobs == [("wonhee.gpu", {"task": "mask"}, {"provenance": {"experiment": "X8 new nodes"}})]


def test_overlap_frac_uses_the_compute_window():
    from gauge.interference import overlap_frac

    def run(start, end, marks=None):
        return {"container_started": start, "workload_finished": end,
                "sj": __import__("json").dumps({"marks": marks or []})}
    me = run(0, 30, [{"mark": "compute_started", "t": 10}, {"mark": "compute_finished", "t": 30}])
    assert overlap_frac(me, [run(0, 20)]) == pytest.approx(0.5)  # partner left halfway through my compute
    assert overlap_frac(me, [run(0, 15), run(12, 40)]) == pytest.approx(1.0)
    assert overlap_frac(run(0, 10), [run(20, 30)]) == 0


def test_interleave_alternates_conditions():
    from gauge.cli import order
    jobs = ["a", "a", "a", "b", "b", "b", "c"]
    assert order(jobs, "file") == jobs
    assert order(jobs) == ["a", "b", "c", "a", "b", "a", "b"]
    shuffled = order(jobs, seed=1)
    assert sorted(shuffled[:3]) == ["a", "b", "c"] and sorted(shuffled) == sorted(jobs)
    assert order(jobs, seed=1) == shuffled  # reproducible


def test_input_hashes_name_media_files(tmp_path, monkeypatch):
    import hashlib

    from gauge import runner
    from gauge.workloads import media
    (tmp_path / "clip.mp4").write_bytes(b"abc")
    monkeypatch.setattr(media, "MEDIA_DIR", tmp_path)
    got = runner.input_hashes(["ffmpeg", "-i", "/media/clip.mp4", "-f", "null", "/media/missing.mp4"])
    assert got == {"/media/clip.mp4": hashlib.sha256(b"abc").hexdigest()}


def test_corun_file_groups_solo_and_override(monkeypatch):
    import argparse

    from gauge import cli, runner
    calls = []
    monkeypatch.setattr(runner, "run", lambda conn, spec, **k: calls.append(("run", spec.sharing, spec.provenance)) or (1, {}))
    monkeypatch.setattr(runner, "corun", lambda conn, specs, **k: calls.append(("corun", [s.sharing for s in specs])) or [])
    a = argparse.Namespace(interval=0.1, no_lock=True, warmness="unknown", timeout=10, mem=None, profile="none",
                           sharing="timeslice")
    doc = {"experiment": "C2 sharing"}
    cli.run_group(None, ["stress.gpu:mode=bw"], doc, a)
    cli.run_group(None, {"jobs": ["stress.gpu:mode=bw", "stress.gpu:mode=compute"], "sharing": "mps"}, doc, a)
    assert calls == [("run", "timeslice", {"experiment": "C2 sharing"}), ("corun", ["mps", "mps"])]


def test_sweep_job_can_set_the_sampler_interval(monkeypatch, tmp_path):
    import argparse

    import yaml

    from gauge import cli, runner, store
    seen = []
    monkeypatch.setattr(runner, "run", lambda conn, spec, interval_s, lock: seen.append((spec.workload, interval_s)) or (1, {}))
    monkeypatch.setattr(store, "connect", lambda *a: None)
    f = tmp_path / "s.yaml"
    f.write_text(yaml.safe_dump({"jobs": [{"workload": "micro.idle", "interval": 1.0}, {"workload": "micro.idle"}]}))
    a = argparse.Namespace(db=":memory:", file=str(f), order="file", seed=None, interval=0.1, no_lock=True,
                           warmness="unknown", timeout=10, mem=None, profile="none", sharing="timeslice")
    cli.cmd_sweep(a)
    assert seen == [("micro.idle", 1.0), ("micro.idle", 0.1)]
