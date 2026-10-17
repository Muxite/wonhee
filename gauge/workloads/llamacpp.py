"""llama.cpp run directly (the engine inside Ollama): one process, clean exit, so its GPU trace is complete.

llamacpp.phase   one phase per job: prefill (process P prompt tokens) or decode (generate N tokens).
llamacpp.batched N sequences decoded together in one process (the batching side of "many agents").
Same GGUF files as llm.generate, from the same read-only volume.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from gauge.workloads.base import Workload, register
from gauge.workloads.llm import MODEL_VOLUME, MODELS

MB = 2**20
IMAGE = "ghcr.io/ggml-org/llama.cpp:full-cuda"

# Resolve the model's GGUF blob from Ollama's manifest, then run the given tool on it.
RESOLVE = r'''
set -e
name=${1%%:*}; tag=${1#*:}; shift
d=$(grep -o '"application/vnd.ollama.image.model","digest":"sha256:[0-9a-f]*' /ollama/models/manifests/registry.ollama.ai/library/$name/$tag | grep -o '[0-9a-f]*$')
M=/ollama/models/blobs/sha256-$d
tool=$1; shift
echo GAUGE_MARK start_tool
exec /app/$tool -m $M "$@"
'''


class PhaseParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal[tuple(MODELS)] = "qwen2.5:3b"  # type: ignore[valid-type]
    phase: Literal["prefill", "decode"] = "decode"
    tokens: int = 256  # prompt tokens (prefill) or generated tokens (decode)
    ngl: int = 99  # layers on the GPU; fewer = partial offload to CPU
    flash_attn: bool = True
    reps: int = 3


def phase_cmd(p) -> list[str]:
    pp, tg = (p.tokens, 0) if p.phase == "prefill" else (0, p.tokens)
    return ["bash", "-c", RESOLVE, "lc", p.model, "llama-bench", "-p", str(pp), "-n", str(tg), "-ngl", str(p.ngl),
            "-fa", "on" if p.flash_attn else "off", "-r", str(p.reps), "-o", "jsonl"]


def phase_prior(p) -> dict:
    m = MODELS[p.model]
    if p.phase == "prefill":
        t = p.tokens * 2 * m["params"] / 60e12  # compute-bound: 2 flops/param/token
    else:
        t = p.tokens * m["file_bytes"] / 700e9  # memory-bound: all weights read per token
    return {"runtime_s": 6 + m["file_bytes"] / 3e9 + t * (p.reps + 1), "vram_peak_bytes": m["file_bytes"] + 900 * MB,
            "ram_peak_bytes": 600 * MB + m["file_bytes"]}


register(Workload(
    name="llamacpp.phase", family="inference", image=IMAGE, params=PhaseParams, gpu=True,
    mounts=(f"{MODEL_VOLUME}:/ollama:ro",), command=phase_cmd,
    size_params=("model", "tokens", "ngl"), size=lambda p: MODELS[p.model]["file_bytes"],
    prior=phase_prior, features=lambda p: {**p.model_dump(), **MODELS[p.model]},
    doc="llama-bench, one phase: prefill (compute-bound) or decode (bandwidth-bound)"))


class BatchedParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal[tuple(MODELS)] = "qwen2.5:3b"  # type: ignore[valid-type]
    prompt_tokens: int = 128
    gen_tokens: int = 128
    parallel: str = "1,2,4,8"  # sequences decoded at once, each level measured in turn


register(Workload(
    name="llamacpp.batched", family="inference", image=IMAGE, params=BatchedParams, gpu=True,
    mounts=(f"{MODEL_VOLUME}:/ollama:ro",),
    command=lambda p: ["bash", "-c", RESOLVE, "lc", p.model, "llama-batched-bench", "-ngl", "99", "-fa", "on",
                       "-c", str((p.prompt_tokens + p.gen_tokens) * max(int(x) for x in p.parallel.split(",")) + 256),
                       "-npp", str(p.prompt_tokens), "-ntg", str(p.gen_tokens), "-npl", p.parallel, "--output-format", "jsonl"],
    size_params=("model",), size=lambda p: MODELS[p.model]["file_bytes"],
    prior=lambda p: {"runtime_s": 20, "vram_peak_bytes": MODELS[p.model]["file_bytes"] + 1500 * MB},
    doc="llama-batched-bench: throughput and latency as 1..N sequences share one model"))
