"""One GPU model, loaded once, then timed: `python -m wonhee.bench <task> --clip X [--repeat R] [--hold S]`.

gauge runs this in a container to split a node's cost into model load and compute (marks.py), and to hold a model
resident while measuring what else fits next to it. Each repeat is one compute span, so repeat 1 against repeat 2
is the cold-call vs warm-call difference a long-lived worker removes.

tasks: whisper (faster-whisper), mask (Grounding-DINO + SAM 2.1, or the CPU ONNX U2-Net), separate (demucs),
embed (SigLIP2 keyframe vectors), worker (`repeat` speech_subs jobs drained by one warm worker).
"""
from __future__ import annotations

import time

T_PROCESS = time.time()  # before the heavy imports: launch -> here is container + interpreter start

import argparse  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from wonhee import marks  # noqa: E402

# Hugging Face cache folders each task loads from. --cache cold|warm puts them in a known page-cache state first, so
# a load is from disk or from memory on purpose, not by whatever the previous run left behind.
HF_MODELS = {"whisper": ["models--mobiuslabsgmbh--faster-whisper-large-v3-turbo"],
             "worker": ["models--mobiuslabsgmbh--faster-whisper-large-v3-turbo"],
             "mask": ["models--IDEA-Research--grounding-dino-tiny", "models--facebook--sam2.1-hiera-base-plus"],
             "separate": ["models--adefossez--HTDemucs"],
             "embed": ["models--google--siglip2-so400m-patch14-384"]}


def set_cache(task: str, state: str) -> int:
    """Put the task's model files in a known page-cache state before loading. cold: posix_fadvise(DONTNEED) via
    `dd iflag=nocache count=0` (works on a read-only mount; pages another process has mapped stay resident).
    warm: read every file once. Returns the number of files touched."""
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    files = [f for d in HF_MODELS.get(task, []) for f in (hub / d / "blobs").glob("*")]
    for f in files:
        if state == "cold":
            subprocess.run(["dd", f"if={f}", "iflag=nocache", "count=0"], capture_output=True)
        else:
            with f.open("rb") as fh:
                while fh.read(1 << 24):
                    pass
    return len(files)


def frames(clip: Path, n: int, px: int) -> list[np.ndarray]:
    """n frames spread evenly over the clip, long side scaled to px."""
    cap = cv2.VideoCapture(str(clip))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    out = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((i + 0.5) * total / n))
        ok, im = cap.read()
        if not ok:
            continue
        s = px / max(im.shape[:2])
        out.append(cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
    cap.release()
    if not out:
        raise ValueError(f"no frames read from {clip}")
    return out


def _torch_vram() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            return {"torch_alloc_peak_bytes": torch.cuda.max_memory_allocated(),
                    "torch_reserved_bytes": torch.cuda.memory_reserved()}
    except ImportError:
        pass
    return {}


def setup(a):
    """-> (load fn, compute fn returning a facts dict). Imports stay inside so each task pulls only its stack."""
    clip = Path(a.clip)
    if a.task == "whisper":
        from wonhee.analyze.speech import WhisperTranscriber
        t = WhisperTranscriber(a.model or "turbo", "cuda", a.compute_type)
        if a.whisper_batch > 1:  # faster-whisper's batched pipeline: VAD chunks decoded together in one pass
            from faster_whisper import BatchedInferencePipeline
            pipe = {}

            def batched():
                pipe.setdefault("p", BatchedInferencePipeline(model=t._load()))
                segs, _ = pipe["p"].transcribe(str(clip), batch_size=a.whisper_batch)
                return {"segments": len(list(segs))}
            return t.ensure_ready, batched
        return t.ensure_ready, lambda: {"segments": len(t.transcribe(clip).segments)}
    if a.task == "mask":
        from wonhee.edit import mask as mk
        ims = frames(clip, a.images, a.px)
        if a.engine == "sam2":
            return mk._gpu_models, lambda: {"masked": sum(mk.subject_text(im, a.prompt) is not None for im in ims)}
        models = Path(a.models_dir)
        return (lambda: mk._session(models, "person"),
                lambda: {"masked": sum(float(mk.subject(im, models, "person").mean()) > 0.01 for im in ims)})
    if a.task == "separate":
        from wonhee.edit import separate as sep
        tmp = Path(tempfile.mkdtemp())
        return (lambda: sep._separator(a.model or "htdemucs"),
                lambda: sep.split_voice(clip, tmp / "v.wav", tmp / "b.wav", a.model or "htdemucs") or {})
    if a.task == "embed":
        from wonhee.analyze import embed
        ims = frames(clip, a.images, 384)
        return embed.load, lambda: {"vectors": len(embed.image_vecs(ims, batch=a.batch))}
    raise ValueError(a.task)


def worker_bench(a) -> int:
    """The warm path end to end: `repeat` speech_subs runs queued in a throwaway home, drained by one worker that
    preloads whisper. Each job is a compute span (marks from wonhee/worker.py); compare with gauge's cold
    wonhee.speech_subs, where every call is a fresh process."""
    import os

    os.environ["WONHEE_HOME"] = tempfile.mkdtemp(prefix="wonhee-worker-bench-")
    from wonhee import worker
    from wonhee.config import Settings
    from wonhee.db import repo
    from wonhee.db.connect import open_db
    from wonhee.nodes.chain import parse_chain
    from wonhee.pipeline.deps import Deps

    settings = Settings()
    conn = open_db(settings)
    sha = repo.save_file(conn, settings.media_dir, Path(a.clip), "video", "video/mp4")
    chain = parse_chain({"nodes": [{"id": "s", "node": "speech_subs", "in": {"video": sha}}], "no_cache": True})
    for _ in range(a.repeat):
        iid = repo.create_intent(conn, "bench", "chain", 1, chain.model_dump(by_alias=True, mode="json"))
        repo.confirm_intent(conn, iid)
        rid = repo.start_run(conn, repo.create_project(conn, "bench", "chain", iid), 1.0)
        worker.enqueue(conn, rid, "cuda")
    t0 = time.perf_counter()
    w = worker.Worker(settings, name="bench", preload=["whisper"])
    done = w.serve(Deps.live(settings), drain=True, progress=print)
    jobs = conn.execute("SELECT status, finished_at - claimed_at AS s FROM job ORDER BY id").fetchall()
    marks.outcome(task="worker", jobs=done, job_s=[j["s"] for j in jobs], total_s=time.perf_counter() - t0,
                  failed=sum(j["status"] != "done" for j in jobs))
    return 0 if done == a.repeat and all(j["status"] == "done" for j in jobs) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="wonhee.bench")
    ap.add_argument("task", choices=["whisper", "mask", "separate", "embed", "worker"])
    ap.add_argument("--clip", required=True)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--hold", type=float, default=0, help="after loading, stay resident this many seconds")
    ap.add_argument("--model", default=None)
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--engine", choices=["sam2", "onnx"], default="sam2")
    ap.add_argument("--images", type=int, default=4)
    ap.add_argument("--px", type=int, default=1080)
    ap.add_argument("--prompt", default="person")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--whisper-batch", type=int, default=0, help="whisper: >1 uses the batched pipeline")
    ap.add_argument("--models-dir", default="/models")
    ap.add_argument("--cache", choices=["as_is", "cold", "warm"], default="as_is",
                    help="page-cache state of the model files before loading: cold = evicted, warm = read in")
    a = ap.parse_args(argv)
    marks.mark("started")  # after imports: started - container start = launch + interpreter + imports
    import_s = time.time() - T_PROCESS
    cache_files = set_cache(a.task, a.cache) if a.cache != "as_is" else 0
    if a.cache != "as_is":
        marks.mark(f"page_cache_{a.cache}")
    if a.task == "worker":
        return worker_bench(a)

    if a.task in ("mask", "separate", "embed"):  # the framework import, apart from loading the weights
        import torch  # noqa: F401
        if a.task in ("mask", "embed"):
            import transformers  # noqa: F401
    marks.mark("framework_ready")
    load, compute = setup(a)
    t0 = time.perf_counter()
    load()
    load_s = time.perf_counter() - t0
    marks.mark("model_loaded")
    if a.hold:
        compute()  # one real call first, so the held footprint includes activations and workspaces
        marks.mark("holding")
        marks.outcome(task=a.task, load_s=load_s, held_s=a.hold, **_torch_vram())
        time.sleep(a.hold)
        return 0
    spans, facts = [], {}
    for _ in range(a.repeat):
        marks.mark("compute_started")
        t = time.perf_counter()
        facts = compute() or {}
        spans.append(time.perf_counter() - t)
        marks.mark("compute_finished")
    marks.outcome(task=a.task, import_s=import_s, load_s=load_s, compute_s=spans, cache=a.cache, cache_files=cache_files,
                  **facts, **_torch_vram())
    print(f"{a.task}: load {load_s:.2f}s, compute {', '.join(f'{s:.2f}s' for s in spans)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
