"""Finite local-LLM inference: a fresh Ollama server per job (start, load, N requests, exit).

The job container owns the model, so this is `isolated`, not a request to a shared server.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from gauge.workloads.base import Workload, register

MB = 2**20
OLLAMA_IMAGE = "ollama/ollama:0.35.0"
MODEL_VOLUME = "badmodel-lab_badmodel_ollama"  # mounted read-only; holds the qwen2.5 size ladder

# Facts known before launch. file_bytes from the volume's manifests; shapes from the models' configs.
MODELS = {
    "qwen2.5:0.5b": dict(file_bytes=397_807_936, params=0.49e9, layers=24, kv_heads=2, head_dim=64),
    "qwen2.5:1.5b": dict(file_bytes=986_048_512, params=1.54e9, layers=28, kv_heads=2, head_dim=128),
    "qwen2.5:3b": dict(file_bytes=1_929_903_008, params=3.09e9, layers=36, kv_heads=2, head_dim=128),
    "qwen2.5:7b": dict(file_bytes=4_683_073_952, params=7.62e9, layers=28, kv_heads=4, head_dim=128),
    "qwen2.5:14b": dict(file_bytes=8_988_110_688, params=14.7e9, layers=48, kv_heads=8, head_dim=128),
}

SCRIPT = r'''
set -u
model=$1 ctx=$2 ptok=$3 gen=$4 reqs=$5
export OLLAMA_MODELS=/ollama/models OLLAMA_CONTEXT_LENGTH=$ctx OLLAMA_NUM_PARALLEL=1 OLLAMA_KEEP_ALIVE=-1
ollama serve >/tmp/serve.log 2>&1 &
until ollama list >/dev/null 2>&1; do sleep 0.05; done
echo GAUGE_MARK server_ready
# ~7 tokens per sentence
prompt=$(for i in $(seq 1 $(( ptok / 7 + 1 ))); do printf 'The cat sat on the mat. '; done)
body="{\"model\":\"$model\",\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"${prompt}Continue the story.\"}],\"options\":{\"num_ctx\":$ctx,\"num_predict\":$gen,\"temperature\":0,\"seed\":1}}"
for r in $(seq 1 $reqs); do
  exec 3<>/dev/tcp/127.0.0.1/11434
  printf 'POST /api/chat HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s' "${#body}" "$body" >&3
  resp=$(cat <&3 | tr -d '\r' | grep '^{' | tail -1)
  exec 3>&-
  [ -z "$resp" ] && { echo "empty response"; tail -20 /tmp/serve.log; exit 2; }
  case "$resp" in *'"error"'*) echo "$resp"; tail -20 /tmp/serve.log; exit 3;; esac
  # drop the generated text; keep the timing/count fields
  echo "GAUGE_OUTCOME {\"request\":$r,${resp#*\"done\":true,}"
  [ "$r" = 1 ] && echo GAUGE_MARK first_response
done
grep -iE "offload|layers.model|model weights|kv cache|compute graph" /tmp/serve.log | tail -8
'''


class GenParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal[tuple(MODELS)] = "qwen2.5:3b"  # type: ignore[valid-type]
    num_ctx: int = 4096
    prompt_tokens: int = 500
    gen_tokens: int = 200
    requests: int = 1


def kv_bytes(m: dict, ctx: int) -> float:
    return 2 * m["layers"] * m["kv_heads"] * m["head_dim"] * 2 * ctx  # K and V, f16


def gen_prior(p) -> dict:
    m = MODELS[p.model]
    load = 1.5 + m["file_bytes"] / 3e9  # server start + read weights (page-cache speed)
    prefill = p.prompt_tokens * 2 * m["params"] / 30e12 * p.requests
    decode = p.gen_tokens * m["file_bytes"] / 700e9 * p.requests  # memory-bandwidth bound
    runtime = load + prefill + decode
    return {"runtime_s": runtime,
            "cpu_time_s": runtime * 1.1,
            # mmapped weights are page cache charged to whichever cgroup first touches them
            "ram_peak_bytes": 400 * MB + m["file_bytes"],
            "vram_peak_bytes": m["file_bytes"] + kv_bytes(m, p.num_ctx) + 600 * MB}


def gen_features(p) -> dict:
    m = MODELS[p.model]
    return {**p.model_dump(), **m, "kv_bytes": kv_bytes(m, p.num_ctx)}


register(Workload(
    name="llm.generate", family="inference", image=OLLAMA_IMAGE, params=GenParams, gpu=True,
    mounts=(f"{MODEL_VOLUME}:/ollama:ro",),
    command=lambda p: ["bash", "-c", SCRIPT, "gen", p.model, str(p.num_ctx), str(p.prompt_tokens),
                       str(p.gen_tokens), str(p.requests)],
    # everything that scales cost is a size param, so one config pools across models and contexts
    size_params=("model", "num_ctx", "prompt_tokens", "gen_tokens"),
    size=lambda p: MODELS[p.model]["file_bytes"] + kv_bytes(MODELS[p.model], p.num_ctx),
    prior=gen_prior, features=gen_features,
    doc="fresh ollama server: load model, N chat requests, exit"))
