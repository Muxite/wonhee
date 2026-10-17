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
model=$1 ctx=$2 ptok=$3 gen=$4 reqs=$5 conc=$6 evict=$7
export OLLAMA_MODELS=/ollama/models OLLAMA_CONTEXT_LENGTH=$ctx OLLAMA_NUM_PARALLEL=$conc OLLAMA_KEEP_ALIVE=-1
if [ "$evict" = 1 ]; then
  # drop this model's weights from the host page cache (fadvise DONTNEED), so the load reads the disk
  tag=${model#*:}; name=${model%%:*}
  digest=$(grep -o '"application/vnd.ollama.image.model","digest":"sha256:[0-9a-f]*' /ollama/models/manifests/registry.ollama.ai/library/$name/$tag | grep -o '[0-9a-f]*$')
  dd if=/ollama/models/blobs/sha256-$digest iflag=nocache count=0 2>/dev/null && echo GAUGE_MARK page_cache_evicted
fi
ollama serve >/tmp/serve.log 2>&1 &
srv=$!
until ollama list >/dev/null 2>&1; do sleep 0.05; done
echo GAUGE_MARK server_ready
prompt=$(for i in $(seq 1 $(( ptok / 7 + 1 ))); do printf 'The cat sat on the mat. '; done)
body="{\"model\":\"$model\",\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"${prompt}Continue the story.\"}],\"options\":{\"num_ctx\":$ctx,\"num_predict\":$gen,\"temperature\":0,\"seed\":1}}"
one() {  # $1 = request number; prints the response JSON line
  exec 3<>/dev/tcp/127.0.0.1/11434
  printf 'POST /api/chat HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s' "${#body}" "$body" >&3
  cat <&3 | tr -d '\r' | grep '^{' | tail -1
  exec 3>&-
}
for r in $(seq 1 $reqs); do
  t0=$(date +%s.%N)
  for c in $(seq 1 $conc); do one $r > /tmp/resp_${r}_${c} & done
  wait $(jobs -p | grep -v "^$srv$") 2>/dev/null
  t1=$(date +%s.%N)
  for c in $(seq 1 $conc); do
    resp=$(cat /tmp/resp_${r}_${c})
    [ -z "$resp" ] && { echo "empty response"; tail -20 /tmp/serve.log; kill $srv; exit 2; }
    case "$resp" in *'"error"'*) echo "$resp"; tail -20 /tmp/serve.log; kill $srv; exit 3;; esac
    echo "GAUGE_OUTCOME {\"request\":$r,\"slot\":$c,\"batch_wall_s\":$(awk "BEGIN{print $t1 - $t0}"),${resp#*\"done\":true,}"
  done
  echo GAUGE_MARK request_${r}_done
done
grep -iE "offload|layers.model|model weights|kv cache|compute graph" /tmp/serve.log | tail -8
# unload the model cleanly so its runner exits on its own (profilers flush at exit), then stop the server
ub="{\"model\":\"$model\",\"keep_alive\":0}"
exec 3<>/dev/tcp/127.0.0.1/11434
printf 'POST /api/generate HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s' "${#ub}" "$ub" >&3
cat <&3 >/dev/null; exec 3>&-
sleep 1
kill $srv; wait $srv 2>/dev/null
exit 0
'''


class GenParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal[tuple(MODELS)] = "qwen2.5:3b"  # type: ignore[valid-type]
    num_ctx: int = 4096
    prompt_tokens: int = 500
    gen_tokens: int = 200
    requests: int = 1
    # Ollama 0.35 ships PTX the driver JIT-compiles for this GPU: ~60 s per fresh container unless the
    # compute cache persists. "shared" mounts one cache volume across jobs (cold on its first use).
    jit_cache: Literal["none", "shared"] = "shared"
    concurrent: int = 1  # requests sent at once to one server (OLLAMA_NUM_PARALLEL = this)
    evict_page_cache: bool = False  # drop the model file from host page cache first: load from disk


def kv_bytes(m: dict, ctx: int) -> float:
    return 2 * m["layers"] * m["kv_heads"] * m["head_dim"] * 2 * ctx  # K and V, f16


JIT_S = 60.0  # measured once on this box (run 12); a fact about the stack, so the prior may use it


def gen_prior(p) -> dict:
    m = MODELS[p.model]
    load = 1.5 + m["file_bytes"] / 3e9 + (JIT_S if p.jit_cache == "none" else 0.0)
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
    param_mounts=lambda p: ("gauge-cuda-cache:/root/.nv",) if p.jit_cache == "shared" else (),
    command=lambda p: ["bash", "-c", SCRIPT, "gen", p.model, str(p.num_ctx), str(p.prompt_tokens),
                       str(p.gen_tokens), str(p.requests), str(p.concurrent), "1" if p.evict_page_cache else "0"],
    # everything that scales cost is a size param, so one config pools across models and contexts
    size_params=("model", "num_ctx", "prompt_tokens", "gen_tokens"),
    size=lambda p: MODELS[p.model]["file_bytes"] + kv_bytes(MODELS[p.model], p.num_ctx),
    prior=gen_prior, features=gen_features,
    doc="fresh ollama server: load model, N chat requests, exit"))
