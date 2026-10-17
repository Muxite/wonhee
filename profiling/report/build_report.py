"""Assemble the report page: new body, static appendix tables from the gauge ledger, chart code reused."""
import html
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gauge import paths, store  # noqa: E402

# Chart data is carried over from the current page; tables below are rebuilt from the ledger.
B = Path(__file__).resolve().parent
old = open(B / "shared-gpu-node.html").read()
DATA = json.loads(old[old.index("const DATA = ") + 13: old.index(";\nconst css")])
tmpl = open(B / "template.html").read()
STYLE = tmpl[tmpl.index("<style>"): tmpl.index("</style>") + 8]
SCRIPT = tmpl[tmpl.index("<script>"): tmpl.rindex("</script>") + 9]
HEAD = tmpl[: tmpl.index("<style>")]
c = store.connect(paths.DATA / "gauge.sqlite")
E = html.escape


def table(headers, rows, cls="data"):
    h = "".join(f"<th>{E(x)}</th>" for x in headers)
    b = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for r in rows)
    return f'<div class="tbl"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


f0 = lambda v: f"{v:,.0f}"  # noqa: E731
f1 = lambda v: f"{v:,.1f}"  # noqa: E731
f2 = lambda v: f"{v:,.2f}"  # noqa: E731

# --- extra numbers straight from the ledger ---
ceil = {}
for (oj,) in c.execute("SELECT outcome_json FROM run WHERE workload='micro.ceilings' AND id BETWEEN 188 AND 190"):
    for r in json.loads(oj).get("responses", []):
        key = (r["test"], r.get("format") or r.get("rows") or ("pinned" if r.get("pinned") else "pageable" if "pinned" in r else ""))
        val = r.get("tflops") if r["test"] == "matmul" else r.get("us_per_launch") if r["test"] == "launch" else r.get("gb_per_s")
        if val:
            ceil.setdefault(key, []).append(val)
CEIL_ROWS = []
for (test, k), v in sorted(ceil.items(), key=lambda kv: str(kv[0])):
    unit = {"matmul": "TFLOPS", "launch": "µs per kernel launch"}.get(test, "GB/s")
    name = {"matmul": f"Matrix multiply, {k}", "gemv": f"Matrix × vector, fp16, {k} rows", "device_copy": "GPU memory copy",
            "h2d": f"CPU → GPU copy, {k} memory", "d2h": f"GPU → CPU copy, {k} memory", "launch": "Kernel launch"}[test]
    CEIL_ROWS.append([E(name), f2(st.median(v)) + " " + unit, str(len(v))])

x2 = {}
for rid, sj, oj in c.execute("SELECT id, spec_json, outcome_json FROM run WHERE workload='llamacpp.phase' AND id BETWEEN 191 AND 206"):
    p = json.loads(sj)["params"]
    x2.setdefault((p["model"], p["phase"]), []).append(json.loads(oj)["avg_ts"])
prof = {r[0]: json.loads(r[1]) for r in c.execute("SELECT run_id, json FROM gpu_profile")}
def dm(rid, key):
    for k, v in prof.get(rid, {}).get("device_metrics", {}).items():
        if k.startswith(key):
            return v.get("mean_gpu_window") or 0
    return 0
MODEL_BYTES = {"qwen2.5:0.5b": 397807936, "qwen2.5:3b": 1929903008, "qwen2.5:7b": 4683073952, "qwen2.5:14b": 8988110688}
X2_ROWS = []
first_run = {("qwen2.5:0.5b", "prefill"): 191, ("qwen2.5:0.5b", "decode"): 193, ("qwen2.5:3b", "prefill"): 195, ("qwen2.5:3b", "decode"): 197,
             ("qwen2.5:7b", "prefill"): 199, ("qwen2.5:7b", "decode"): 201, ("qwen2.5:14b", "prefill"): 203, ("qwen2.5:14b", "decode"): 205}
for (m, ph), v in sorted(x2.items(), key=lambda kv: (list(MODEL_BYTES).index(kv[0][0]), kv[0][1])):
    rid = first_run[(m, ph)]
    tps = st.median(v)
    bw = MODEL_BYTES[m] * tps / 1e9 if ph == "decode" else None
    X2_ROWS.append([m.split(":")[1], ph, f0(tps), f"{dm(rid, 'DRAM Read'):.0f}%", f"{dm(rid, 'Tensor Active'):.0f}%",
                    f"{dm(rid, 'SMs Active'):.0f}%", f0(bw) if bw else "–"])

OFFLOAD = []
for rid, sj, oj in c.execute("SELECT id, spec_json, outcome_json FROM run WHERE id BETWEEN 207 AND 211 ORDER BY id"):
    p = json.loads(sj)["params"]
    OFFLOAD.append([str(p["ngl"] if p["ngl"] < 99 else 28) + " of 28", f1(json.loads(oj)["avg_ts"]),
                    f"{prof.get(rid, {}).get('process', {}).get('kernel_busy_frac_of_window', 0) * 100:.0f}%"])

KMIX = []
for rid, label in ((195, "prefill, qwen2.5 3B"), (197, "decode, qwen2.5 3B"), (212, "CNN training fp32"), (216, "speech-to-text"), (218, "NVENC encode")):
    p = prof[rid]["process"]
    top = list(p["by_category"].items())[:3]
    KMIX.append([label, ", ".join(f"{k} {v['share'] * 100:.0f}%" for k, v in top), f"{p['kernel_median_us']:.1f}",
                 f0(p["launches_per_s"]), f"{p['kernel_busy_frac_of_window'] * 100:.0f}%"])

# tables from embedded DATA
CONC = [[r["mode"], str(r["n"]), f1(r["ms_per_token"]), f0(r["per_stream_tps"]), f0(r["total_tps"]), f2(r["total_vram_gib"]), str(r["n_obs"])]
        for r in sorted(DATA["concurrency"], key=lambda r: (r["mode"], r["n"]))]
MATRIX = [[E(m["job"]), E(m["alongside"]), "×" + f2(m["slowdown"]), str(m["n"])] for m in sorted(DATA["matrix"], key=lambda m: (m["job"], m["alongside"]))]
X7 = []
for p in DATA["x7"]["pairs"]:
    for k, m in p["metrics"].items():
        X7.append([E(p["label"]), E(k), f1(m["A"]), f1(m["B"]), f1(m["A+B observed"]), f1(m["time_sliced"]), f1(m["additive"])])
SHARES = [[E(p["label"]), f"{p['share_A'] * 100:.0f}%", f"{p['share_B'] * 100:.0f}%"] for p in DATA["x7"]["pairs"]]
START = [[E(r["label"]), f0(r["p50"]), f0(r["p10"]), f0(r["p90"]), "–" if r["label"].startswith("Native") else "+" + f0(r["p50"] - DATA["overhead"]["native"]["noop"]), str(r["n"])]
         for r in sorted(DATA["merge"]["start"], key=lambda r: r["p50"])]
PHASES = [[E(p["label"]), f0(p["create"]), f0(p["start"]), f0(p["first"]), f0(p["ret"]), f0(p["rm"])] for p in DATA["overhead"]["phases"]]
FIXED = [[E(r["label"]), "+" + f0(r["overhead_noop"]), "+" + f0(r["overhead_stress"])] for r in DATA["overhead"]["overhead"] if r["overhead_stress"] is not None and not r["label"].startswith("Native")]
L = DATA["merge"]["ladder"]
LADDER = [["already in GPU memory", f"{L['resident']['median']:.3f}", str(L["resident"]["n"])], ["in RAM (page cache)", f2(L["cache"]["median"]), str(L["cache"]["n"])],
          ["read from disk", f2(L["disk"]["median"]), str(L["disk"]["n"])]]
REGIME = [[E(r["node"]), str(r["run_id"]), f"{r['dram_read_pct']:.0f}%", f"{r['tensor_pct']:.0f}%", f"{r['sm_active_pct']:.0f}%", f"{r['busy_frac'] * 100:.0f}%"] for r in DATA["regime"]]

BODY = f'''<div class="wrap">
  <header>
    <div class="kicker">Report · GPU workload profiling · measured 2026-10-16</div>
    <h1>Shared GPU, same node</h1>
    <p class="lede">What happens when AI task nodes (small LLMs, model training, speech-to-text, video encoding) share one GPU, what it costs to start them, and what the GPU is doing inside each one. Every number is measured, per process, on one workstation.</p>
    <nav class="toc"><a href="#short">Short version</a><a href="#f1">1 · A + B</a><a href="#f2">2 · Copies vs batching</a><a href="#f3">3 · Who loses</a><a href="#f4">4 · What each uses</a><a href="#f5">5 · Starting</a><a href="#f6">6 · wonhee</a><a href="#do">What to do</a><a href="#limits">Limits &amp; ARM</a><a href="#open">Open questions</a><a href="#appendix">Appendix</a></nav>
  </header>

  <section id="short" class="short">
    <h2>The short version</h2>
    <ol>
      <li><b>Two tasks on one GPU take turns; they do not add up.</b> Run together, the GPU looks like a blend of the two, weighted by who got more time, minus 2–22% lost to switching between them.</li>
      <li><b>Copies of the same model as separate processes is the worst arrangement.</b> Four copies produce less in total than one. Batching the same four requests in one process gives 2.4× the output in less than a third of the memory.</li>
      <li><b>Who loses depends on the shape of the work.</b> A small LLM loses about half its speed next to model training; training barely notices. Tasks that use other hardware (the video encoder, the CPU) hardly interfere.</li>
      <li><b>Starting is cheap; starting cold is not.</b> A container adds about 0.15 s. A fresh LLM container that has to compile its GPU code adds about 59 s, and loading a model from disk instead of keeping it in GPU memory costs about 120×.</li>
      <li><b>Everything comes with per-node execution traces</b>: every GPU kernel each node ran, alone and shared, ready to check a simulator against.</li>
    </ol>
    <div class="spec">One workstation: RTX 5070 Ti 16 GB (Blackwell) on PCIe Gen4 x16 · Ryzen 7 3700X, 16 threads · 46 GiB RAM · Docker 29.8 · llama.cpp CUDA · Nsight Systems 2026.3. Details in the appendix.</div>
  </section>

  <section id="f1" class="finding">
    <div class="fnum">Finding 1</div>
    <h2>Two tasks on one GPU take turns: A + B is a blend, not a sum</h2>
    <p>We recorded the whole GPU's activity counters 10,000 times a second through one fixed sequence: idle, task A alone, idle, task B alone, idle, both together. Each task also recorded its own GPU work, which shows how the GPU split its time between them.</p>
    <div class="legend"><span><i style="background:var(--separate)"></i>memory (DRAM) bandwidth</span><span><i style="background:var(--batched)"></i>tensor cores</span><span><i style="background:var(--muted)"></i>compute units (SMs) active</span></div>
    <figure><div class="fig-label">Fig. 1 · GPU activity through each sequence, % of peak, 100 ms steps</div><div class="chart" id="f-seq" style="min-height:420px"></div>
      <figcaption>Shaded bands are the measured windows. In the A+B window the lines sit between A's and B's levels instead of adding up.</figcaption></figure>
    <figure><div class="fig-label">Fig. 2 · measured A + B against two simple models</div><div class="chart" id="f-model" style="min-height:300px"></div>
      <figcaption><b>Sum</b>: both run at once and their activity adds. <b>Time-split</b>: the GPU alternates, so A + B is each task's solo activity weighted by its share of GPU time. Time-split is close; the measured value falls below it by the time lost to switching.</figcaption></figure>
    <div class="nums">
      <div><b>84% / 16%</b><span>GPU time to training vs the LLM when they share</span></div>
      <div><b>52% vs 62%</b><span>memory bandwidth used by two identical LLMs together vs one alone</span></div>
      <div><b>2–22%</b><span>of the GPU's work lost to switching, largest when two similar tasks alternate evenly</span></div>
    </div>
    <p class="means"><b>What it means:</b> to estimate a shared GPU, weight each task's solo profile by its share of GPU time and subtract a switching cost. Never add them.</p>
  </section>

  <section id="f2" class="finding">
    <div class="fnum">Finding 2</div>
    <h2>Separate copies of a model lose to batching in one process</h2>
    <p>One LLM task (qwen2.5 3B generating 256 tokens) run alone, as 2 and 4 separate processes, and as 2, 4 and 8 requests batched inside one process.</p>
    <div class="legend"><span><i style="background:var(--separate)"></i>separate processes</span><span><i style="background:var(--batched)"></i>batched in one process</span></div>
    <div class="figs">
      <figure><div class="fig-label">Fig. 3 · wait per token, each request</div><div class="chart" id="f-lat"></div>
        <figcaption>Separate processes take turns, so each one waits for the others.</figcaption></figure>
      <figure><div class="fig-label">Fig. 4 · total output</div><div class="chart" id="f-tps"></div>
        <figcaption>Batching reads each weight once for every request; separate processes read it once each.</figcaption></figure>
    </div>
    <figure><div class="fig-label">Fig. 5 · GPU memory</div><div class="chart" id="f-mem" style="min-height:200px"></div>
      <figcaption>Each separate process loads its own copy of the model. Eight copies would not fit this card.</figcaption></figure>
    <div class="nums">
      <div><b>3.5 → 15.8 ms</b><span>per token, alone vs one of four separate processes</span></div>
      <div><b>284 → 258</b><span>tokens/s in total: four separate processes make less than one</span></div>
      <div><b>614 tokens/s</b><span>the same four requests batched, at 6.5 ms per token</span></div>
      <div><b>8.9 vs 2.5 GiB</b><span>GPU memory, four separate processes vs batched</span></div>
    </div>
    <p class="means"><b>What it means:</b> serve each model from one process that batches requests from many agents, rather than one model process per agent.</p>
  </section>

  <section id="f3" class="finding">
    <div class="fnum">Finding 3</div>
    <h2>Who loses depends on the shape of the work</h2>
    <p>Four task types shared the GPU in every pairing, each compared with its own runs alone.</p>
    <div class="figs">
      <figure><div class="fig-label">Fig. 6 · slowdown of a task beside another</div><div class="chart" id="f-heat" style="min-height:260px"></div>
        <figcaption>Row task's slowdown with the column task on the same GPU. The LLM loses about half its speed beside training; training slows only 16%.</figcaption></figure>
      <figure><div class="fig-label">Fig. 7 · one LLM's GPU time, alone vs one of four</div><div class="chart" id="f-trace" style="min-height:190px"></div>
        <figcaption>From the task's own kernel trace, 600 ms in 5 ms steps. Alone it keeps the GPU <b id="busy-solo">—</b> busy; as one of four it gets <b id="busy-four">—</b>, in bursts.</figcaption></figure>
    </div>
    <p class="means"><b>What it means:</b> the LLM runs many very short GPU steps (about 7 µs each) with pauses between tokens, so a task with long GPU steps, such as training, holds the GPU and the LLM keeps losing its turn. Pair LLMs with tasks that use different hardware.</p>
  </section>

  <section id="f4" class="finding">
    <div class="fnum">Finding 4</div>
    <h2>Each task leans on a different part of the GPU</h2>
    <figure><div class="fig-label">Fig. 8 · memory bandwidth vs tensor-core use, each task alone</div><div class="chart" id="f-reg" style="min-height:260px"></div>
      <figcaption>Hover a dot for details. Generating text (decode) is limited by memory bandwidth and uses no tensor cores. Reading the prompt (prefill) and training use tensor cores. Speech-to-text and video encoding barely use either.</figcaption></figure>
    <p class="means"><b>What it means:</b> generating text re-reads the whole model for every token, so on this card a 7B model reaches about 95% of the measured memory ceiling. Faster generation needs more memory bandwidth, not more compute. Appendix table B.3 has the numbers per model size.</p>
  </section>

  <section id="f5" class="finding">
    <div class="fnum">Finding 5</div>
    <h2>Starting a task is cheap; starting it cold is not</h2>
    <p>The same tiny command was timed 40 times for each way of launching it, in shuffled order: 1,760 container starts in all.</p>
    <div class="nums">
      <div><b>+149 ms</b><span>a plain <code>docker run</code> over a native process; image size makes no difference</span></div>
      <div><b>+46 ms</b><span>handing work to an already-running container (<code>docker exec</code>)</span></div>
      <div><b>64 s vs 5.3 s</b><span>a fresh LLM container without vs with a shared GPU-code cache</span></div>
      <div><b>0.05 / 2.7 / 6.1 s</b><span>first token, 7B model already loaded / in RAM / on disk</span></div>
    </div>
    <figure><div class="fig-label">Fig. 9 · request to result, by launch method</div><div class="chart" id="f-start" style="min-height:300px"></div>
      <figcaption>Bar: median of 40. Whisker: 10th to 90th percentile. Brackets: overhead over a plain process (3 ms).</figcaption></figure>
    <div class="legend" id="phase-legend"></div>
    <figure><div class="fig-label">Fig. 10 · where the overhead goes</div><div class="chart" id="f-phase" style="min-height:260px"></div>
      <figcaption>Most of the time is between Docker reporting "started" and the container's first instruction. GPU access adds about 65 ms there; the default network adds about 30 ms there and 95 ms on exit.</figcaption></figure>
    <div class="figs">
      <figure><div class="fig-label">Fig. 11 · overhead with an empty vs a real task</div><div class="chart" id="f-fixed" style="min-height:230px"></div>
        <figcaption>About the same with 340 ms of real work inside: a fixed cost per launch, which the task itself does not pay (339 vs 336 ms).</figcaption></figure>
      <figure><div class="fig-label">Fig. 12 · starting many at once</div><div class="chart" id="f-burst" style="min-height:220px"></div>
        <figcaption>Each container gets slower when many start together: 16 at once is about 2.8× slower each.</figcaption></figure>
    </div>
    <figure><div class="fig-label">Fig. 13 · first token from a 7B model, by where the model is</div><div class="chart" id="f-ladder" style="min-height:150px"></div>
      <figcaption>Keeping a model loaded is about 120× faster than reading it from disk.</figcaption></figure>
    <p class="means"><b>What it means:</b> for short tasks, keep warm containers and hand them work; for LLMs, share the GPU-code cache between containers and keep busy models loaded.</p>
  </section>

  <section id="f6" class="finding">
    <div class="fnum">Finding 6</div>
    <h2>The wonhee video nodes: startup-bound and CPU-bound</h2>
    <p>Two nodes from wonhee, the existing video pipeline used here as a stand-in for agent work nodes, run on clips of 10 to 100 seconds.</p>
    <div class="figs">
      <figure><div class="fig-label">Fig. 14 · speech-to-text: time vs clip length</div><div class="chart" id="f-speech" style="min-height:220px"></div>
        <figcaption>Blue: whisper turbo; pink: smaller models at 40 s. About 15 s regardless of length; the GPU is busy only 3% of it.</figcaption></figure>
      <figure><div class="fig-label">Fig. 15 · subtitle burn-in: CPU time vs clip length</div><div class="chart" id="f-burn" style="min-height:220px"></div>
        <figcaption>Grows in a straight line, about 3.3 CPU-seconds per second of 1080×1920 video, all on the CPU encoder.</figcaption></figure>
    </div>
    <p class="means"><b>What it means:</b> a long-running speech worker would remove most of that node's time; moving subtitle encoding to the GPU's video encoder would free the CPU.</p>
  </section>

  <section id="do">
    <h2>What to do with this</h2>
    <ul class="dolist">
      <li><b>One model, one batching server.</b> Route all agents' requests for a model to one process that batches them (Finding 2).</li>
      <li><b>Pair by hardware, not by convenience.</b> Put an LLM next to video encoding or CPU work, not next to training (Findings 1, 3, 4).</li>
      <li><b>Budget memory per copy.</b> Every separate process holds its own weights (Fig. 5).</li>
      <li><b>Keep short tasks warm.</b> Reuse running containers; skip networking where it is not needed (Finding 5).</li>
      <li><b>Avoid the cold paths.</b> Share the GPU-code cache across containers and keep hot models loaded (Finding 5).</li>
      <li><b>Model shared GPUs as time-sharing.</b> A simulator should blend solo profiles by time share and subtract a switching cost (Finding 1).</li>
    </ul>
  </section>

  <section id="limits">
    <h2>Limits, and would it hold on a unified-memory ARM system?</h2>
    <p>One machine and one GPU model. 2–12 repeats per point (repeat agreement is listed in the appendix). The shapes should carry over to other NVIDIA GPUs; the absolute numbers belong to this card. Profiled runs carry a small profiler cost, which is equal across every compared pair.</p>
    <p>Everything here used x86-64 with a separate GPU joined over PCIe. Unified-memory aarch64 systems (NVIDIA Grace-based machines, Jetson, Apple silicon) share one memory pool between CPU and GPU. These are expectations, not measurements:</p>
    <div class="carry">
      <div><h3>Likely to carry over</h3><ul>
        <li><b>Text generation is memory-bandwidth-bound</b>; speed follows that system's bandwidth, often lower on unified LPDDR designs.</li>
        <li><b>Batching beats separate processes</b>: a property of the workload.</li>
        <li><b>Separate processes take turns</b> and lose time switching; the size needs remeasuring.</li>
        <li><b>Container start is a fixed cost</b>; milliseconds depend on the CPU.</li></ul></div>
      <div><h3>Needs remeasuring</h3><ul>
        <li><b>Model loading</b>: no PCIe copy, so "in RAM" vs "in GPU memory" may converge.</li>
        <li><b>Running part of a model on the CPU</b>: the PCIe cliff goes away; the trade-off changes.</li>
        <li><b>CPU tasks vs GPU tasks</b>: they share one memory bandwidth, so new interference is likely.</li>
        <li><b>Memory accounting</b>: one pool, so per-process numbers and limits need re-validating.</li>
        <li><b>Counters, power, software</b>: availability varies by chip; images need arm64 builds.</li></ul></div>
    </div>
  </section>

  <section id="open">
    <h2>Open questions</h2>
    <ul class="open">
      <li><b>What is the switching cost made of?</b> Context-switch time, cache refills or launch queuing. NVIDIA MPS, which removes context switches, would separate them.</li>
      <li><b>Can a node's own profile predict its time share?</b> Kernel length and Fig. 8's position might predict who wins the GPU; the per-node traces are the test set.</li>
      <li><b>Where does batching stop paying off?</b> Larger batches, longer contexts and larger models would find the limit.</li>
      <li><b>Real agent traffic.</b> Bursty calls with gaps, model swaps and cold starts are not measured here.</li>
      <li><b>Other hardware.</b> Rerun on a datacenter GPU and on a unified-memory ARM system to see which shapes carry over.</li>
    </ul>
  </section>

  <section id="appendix" class="appendix">
    <h2>Appendix</h2>
    <p>Written to be used without the rest of this conversation: on another machine, next to other documents, or as input to someone building a simulator. A.x explains the method; B.x holds the numbers behind every figure.</p>

    <h3 id="a1">A.1 Glossary</h3>
    <dl class="gloss">
      <dt>Node / task</dt><dd>One unit of work in an agent pipeline (generate text, transcribe a clip, train for N steps), run as one container.</dd>
      <dt>Prefill / decode</dt><dd>The two phases of LLM inference: reading the whole prompt at once (large matrix multiplies, compute-bound) and generating tokens one at a time (matrix × vector, memory-bandwidth-bound).</dd>
      <dt>Kernel</dt><dd>One function launched on the GPU. A decode step is hundreds of them; each has a start time, a duration and a launch shape (grid, block).</dd>
      <dt>SM</dt><dd>Streaming multiprocessor: the GPU's compute units (the "CUDA cores" live inside them). "SMs active" is the share of them doing work.</dd>
      <dt>Tensor cores</dt><dd>Matrix-multiply units inside the SMs, used by prefill and training.</dd>
      <dt>DRAM bandwidth</dt><dd>How fast the GPU reads and writes its own memory, as a share of the maximum. The limit for text generation.</dd>
      <dt>Time-slicing</dt><dd>How the GPU shares itself between separate processes by default: it switches between their contexts instead of running them at once.</dd>
      <dt>Switching cost</dt><dd>The gap between measured A+B activity and the time-split blend: work lost to context switches and refills.</dd>
      <dt>Batching</dt><dd>One process serving several requests with each weight read once for all of them.</dd>
      <dt>Page cache</dt><dd>Linux keeps recently read files in RAM; a model "in RAM" loads without touching the disk.</dd>
      <dt>cgroup</dt><dd>The Linux mechanism that gives each container its own CPU, memory and IO counters.</dd>
      <dt>NVML / CUPTI / Nsight Systems</dt><dd>NVIDIA's management library (memory, power, utilization), profiling interface (kernel traces) and profiler (traces plus hardware counters).</dd>
    </dl>

    <h3 id="a2">A.2 Environment</h3>
    {table(["Item", "Value"], [
        ["GPU", "NVIDIA GeForce RTX 5070 Ti, Blackwell GB203, 16,303 MiB, driver 595.91.07"],
        ["GPU link", "PCIe Gen4 x16 (the card supports Gen5; the board limits it). Drops to Gen1 at idle."],
        ["CPU / RAM", "AMD Ryzen 7 3700X, 8 cores / 16 threads; 46 GiB RAM"],
        ["OS", "Ubuntu 24.04, Linux 6.8.0-146-generic, cgroup v2 (systemd driver)"],
        ["Containers", "Docker 29.8.2, nvidia-container-toolkit runtime"],
        ["Profiler", "Nsight Systems 2026.3.2 (extracted to ~/opt, run inside each container); GPU counters at 10 kHz"],
        ["Profiling permission", "Driver option RmProfilingAdminOnly=1, so counter-reading containers get --cap-add SYS_ADMIN"],
        ["LLM engine", "llama.cpp ghcr.io/ggml-org/llama.cpp:full-cuda, build 11681 (5e03a1ff4), digest sha256:7dc27d9f…f6"],
        ["LLM server", "Ollama 0.35.0 (cold-start and JIT experiments)"],
        ["Models", "qwen2.5 0.5B / 1.5B / 3B / 7B / 14B, Q4_K_M GGUF from Ollama's library"],
        ["Training", "PyTorch 2.8.0, CUDA 12.8, torchvision ResNet18/50, synthetic 224×224 data"],
        ["Video", "linuxserver/ffmpeg 7.1.1 (libx264 and h264_nvenc)"],
        ["Speech", "faster-whisper via wonhee's node runner (tiny / base / small / turbo)"],
        ["Measurement tool", "gauge (Python 3.13), part of the wonhee repo: gauge/ and profiling/"],
        ["Background", "Measured idle floor under the profiler: ~27 W, 0% SM. The display server and game-stream server were stopped for the later runs."],
    ])}

    <h3 id="a3">A.3 How gauge measures a task</h3>
    <ol class="steps">
      <li><b>Declare.</b> A task is a <i>workload</i> (container image + command + parameters). <code>gauge run</code>, <code>sweep</code> (a grid from YAML), <code>corun</code> (deliberate pairs) and <code>sequence</code> (idle → A → B → A+B) launch them.</li>
      <li><b>Predict, then freeze.</b> Before launch, gauge writes predictions it may never change (the database rejects edits). Kept for later prediction work; not used in this report.</li>
      <li><b>Launch.</b> Each task is one container under a small shell wrapper. The wrapper runs the task, prints its exit code, then waits on stdin, so the container stays alive until the host has read its final counters. Without that, Docker deletes the counters the moment the task exits.</li>
      <li><b>Measure, on the host:</b> every 100 ms, the container's cgroup counters (CPU time, memory, IO, pressure) and NVML (GPU memory per process ID, mapped to the container; utilization, power, clocks, video engines). At the end: the kernel's exact peak-memory mark, out-of-memory events and the GPU energy counter.</li>
      <li><b>Trace, inside:</b> optionally Nsight Systems inside the container records every GPU kernel and memory copy of the task's own processes ("trace"), plus the whole GPU's hardware counters at 10 kHz ("metrics"; one task at a time).</li>
      <li><b>Classify and store.</b> Outcome (success, out of memory on CPU or GPU, timeout, error), a summary of about 70 fields, and the raw samples go to a SQLite ledger. Out-of-memory runs are kept but marked censored.</li>
    </ol>
    {table(["Scope", "Measures", "Isolated to one task?"], [
        ["container", "CPU time, RAM peak, IO, out-of-memory events (cgroup v2)", "Yes, exactly, even when sharing"],
        ["process", "GPU memory per process; every kernel and copy (CUPTI)", "Yes. Under sharing a kernel's duration also includes time the GPU spent on others"],
        ["device", "GPU utilization, power, clocks, SM / tensor / DRAM / PCIe counters", "No: whole GPU. Equals the task only when it ran alone (each run records this)"],
    ])}
    <p>The GPU lock: a machine-wide advisory lock (<code>gpu-lock</code>) is held for every measured run, so other agents' GPU work does not land inside a measurement.</p>

    <h3 id="a4">A.4 Experiments and how to rerun them</h3>
    {table(["ID", "What", "Repeats", "Runs", "Command (in the wonhee repo)"], [
        ["X0", "Idle floor under the profiler (before and after stopping display/streaming)", "3 + 3", "185–187, 286–288", "<code>gauge sweep profiling/sweeps/x/x0_idle_ceilings.yaml</code>"],
        ["X1", "Hardware ceilings: matmul per format, memory and PCIe bandwidth, launch cost", "3", "188–190", "same file"],
        ["X2", "LLM prefill vs decode per model size; partial offload", "2 (each with 3 inner reps)", "191–211", "<code>gauge sweep profiling/sweeps/x/x2_llm_anatomy.yaml</code>"],
        ["X3", "Training (fp32, mixed precision), speech-to-text, NVENC anatomy", "2", "212–219", "<code>gauge sweep profiling/sweeps/x/x3_task_anatomy.yaml</code>"],
        ["X4", "Batched 1/2/4/8 in one process vs 2/4 separate processes", "2 / 4–8", "220–223, 238–261", "<code>x4_batched.yaml</code> (sweep), <code>x4_separate.yaml</code> (corun)"],
        ["X5", "Every pair of LLM / training / NVENC / speech, plus traced solos", "2", "224–231, 262–285", "<code>x5_solos.yaml</code> (sweep), <code>x5_pairs.yaml</code> (corun)"],
        ["X6", "Cold start: model on disk, in RAM, in GPU memory", "3", "232–237", "<code>gauge sweep profiling/sweeps/x/x6_cold_start.yaml</code>"],
        ["X7", "idle → A → B → A+B under one counter observer, 3 pairs", "2–3", "289–316", "<code>gauge sequence profiling/sweeps/x/x7_sequence.yaml</code>"],
        ["Start", "Container start overhead, 9 launch methods + bursts", "40 per method, shuffled", "1,760 starts", "<code>gauge startbench --reps 40</code>"],
        ["JIT", "Fresh Ollama container with / without shared GPU-code cache", "4 + 4", "171–178", "<code>gauge sweep profiling/sweeps/jit.yaml</code>"],
    ])}
    <p>Repeat agreement: X7 pair repeats within 0.3 percentage points; X2 throughput within 2%; most co-run slowdowns within about 0.1×, except the LLM beside training, which ranged ×1.7–2.4 over its two runs. Earlier exploratory runs (1–170) are in the ledger, but this report does not rely on them.</p>

    <h3 id="a5">A.5 Metrics and formulas</h3>
    <dl class="gloss">
      <dt>Slowdown</dt><dd>Run time with a partner ÷ median run time of the identical task alone, in the same profiling mode. For LLM decode we use tokens/s alone ÷ tokens/s shared.</dd>
      <dt>Time share</dt><dd>From each task's own kernel trace inside the A+B window: busy time ÷ window length. When two shares sum past 1 (a kernel's span includes time it was switched out), both are scaled to sum to 1.</dd>
      <dt>Time-split model</dt><dd>A+B ≈ floor + (A − floor) × shareA ÷ busyA,solo + (B − floor) × shareB ÷ busyB,solo, where floor is the idle reading.</dd>
      <dt>Sum model</dt><dd>A+B ≈ A + B − floor, capped at 100%.</dd>
      <dt>Switching cost</dt><dd>(time-split − measured) ÷ time-split, per counter. DRAM read: 9% (LLM + training), 2% (LLM + NVENC), 22% (LLM + LLM).</dd>
      <dt>Decode bandwidth</dt><dd>Model file bytes × tokens/s. Decode reads every weight once per token, so this approximates the memory traffic.</dd>
      <dt>Energy per token</dt><dd>Mean GPU power while busy ÷ tokens/s. Valid for decode; prefill bursts are shorter than NVML's ~1 s power averaging.</dd>
      <dt>Overhead</dt><dd>Median request-to-result time minus the native median for the same work.</dd>
    </dl>

    <h3 id="a6">A.6 Data files</h3>
    {table(["Path (wonhee repo)", "Contents"], [
        ["data/gauge/gauge.sqlite (6.9 MB)", "The ledger: run, prediction, sample (100 ms), summary, evaluation, gpu_profile, startbench, sequence"],
        ["data/gauge/dataset/ (2.3 MB)", "Flat CSVs with DATA_CARD.md: runs, workload_outcomes, gpu_process, gpu_kernel_categories, gpu_top_kernels, gpu_device_counters, kernel_timeline_10ms, samples_100ms, corun_interference, container_starts. Rebuild with <code>gauge dataset</code>"],
        ["data/gauge/traces/ (45 MB)", "Per-node traces: run&lt;id&gt;.node.json.gz and run&lt;id&gt;.perfetto.json.gz (open in ui.perfetto.dev). Make more with <code>gauge trace-export &lt;run ids&gt;</code>"],
        ["data/gauge/nsys/ (20 GB)", "Raw Nsight reports and their SQLite exports for every profiled run and sequence"],
        ["docs/profiling/DATA_CARD.md", "Column-level description of the dataset"],
    ])}
<pre>{{ "schema_version": 1,
  "node":      {{ "run_id": 254, "workload": "llamacpp.phase", "params": {{ "model": "qwen2.5:3b", "phase": "decode", "tokens": 256 }} }},
  "hardware":  {{ "gpu_name": "NVIDIA GeForce RTX 5070 Ti", "driver": "595.91.07", "vram_total": 17094934528 }},
  "isolation": {{ "concurrent_with_runs": [255, 256, 257], "interference_vs_solo": {{ "slowdown": 1.72, "kernel_stretch": 1.65 }} }},
  "timing":    {{ "container_s": 14.31, "gpu_window_s": 7.74 }},
  "resources": {{ "cpu_time_s": …, "ram_peak_bytes": …, "vram_peak_bytes": …, "dev_power_mean_w": …, "dev_energy_j": … }},
  "kernels":   [ {{ "name": "mul_mat_vec_q", "category": "gemv (decode matmul)", "start_us": 412833.97, "dur_us": 6.85,
                   "grid": [2048,1,1], "block": [32,4,1], "stream": 15, "regs_per_thread": 52, "shared_mem_bytes": 768 }}, … ],
  "copies":    [ {{ "kind": "h2d", "start_us": 0.0, "dur_us": 0.61, "bytes": 8192, "stream": 13 }}, … ] }}</pre>
    <p>Identical runs launch identical kernel sequences (353,970 for this task), so a node's solo and shared traces line up kernel for kernel.</p>

    <h3 id="a7">A.7 Setting this up on another machine</h3>
    <ol class="steps">
      <li>Linux with cgroup v2, Docker, and the NVIDIA container toolkit (<code>docker run --gpus all</code> works).</li>
      <li>Copy the wonhee repo (gauge lives in it); <code>uv sync --extra dev</code>; <code>uv run gauge doctor</code> (same as <code>wonhee profile doctor</code>) checks cgroup v2, Docker, the GPU runtime, NVML capabilities and the GPU lock.</li>
      <li>Nsight Systems: download the <code>nsight-systems</code> .deb for your OS and architecture from NVIDIA's CUDA repository and extract it (<code>dpkg-deb -x</code>); point <code>NSYS_DIR</code> in <code>gauge/nsight.py</code> at it. No system install is needed.</li>
      <li>GPU counters: if the driver restricts profiling to admins (<code>RmProfilingAdminOnly: 1</code> in <code>/proc/driver/nvidia/params</code>), gauge already adds SYS_ADMIN to counter-reading containers.</li>
      <li>Images: <code>uv run gauge build</code> (stress, torch, wonhee, startbench, from <code>profiling/images</code>); pull <code>ghcr.io/ggml-org/llama.cpp:full-cuda</code>, <code>ollama/ollama:0.35.0</code> and <code>linuxserver/ffmpeg:7.1.1</code>. On aarch64, use arm64 builds of each.</li>
      <li>Models: pull qwen2.5 into an Ollama volume (<code>ollama pull qwen2.5:3b</code>, etc.) and set <code>MODEL_VOLUME</code> in <code>gauge/workloads/llm.py</code> to that volume.</li>
      <li>Media: <code>uv run gauge prepare</code> makes the video source. The wonhee clips came from a third-party reel and are not included; use any clip with speech.</li>
      <li>Run the experiments in A.4, then <code>gauge dataset</code> and <code>gauge trace-export</code>.</li>
    </ol>

    <h3 id="a8">A.8 Pitfalls we hit</h3>
    <ul class="dolist">
      <li><b>Ollama kills its model process</b>, so its traces lose buffered records. For per-process traces we ran llama.cpp directly (the same engine).</li>
      <li><b>CUDA code compiles on first use</b> in every fresh Ollama container on this GPU (about 59 s) unless <code>/root/.nv</code> is shared between containers.</li>
      <li><b>File memory is charged to the first container that reads a file.</b> A model's first run looks like it needs 9 GB of RAM; later runs, 0.6 GB.</li>
      <li><b>The profiler raises the idle floor</b> (≈27 W vs 17 W unobserved). Only compare profiled runs with profiled runs.</li>
      <li><b>Only one session can read the GPU counters at a time.</b> X7 uses one observer and gives the tasks kernel traces only.</li>
      <li><b>Lock tokens can start with "-"</b>: pass <code>--token=…</code>, or the release fails silently and the GPU lock leaks.</li>
      <li><b>Concurrent jobs need separate database connections</b>; one shared SQLite connection crashed under parallel writes.</li>
      <li><b>stress-ng reads fractional timeouts as 0</b>, which means forever.</li>
    </ul>

    <h3 id="b1">B.1 Separate processes vs batching (Fig. 3–5)</h3>
    {table(["Mode", "Streams", "ms / token / stream", "tokens/s / stream", "tokens/s total", "GPU memory GiB", "Observations"], CONC)}
    <h3 id="b2">B.2 Pair slowdowns (Fig. 6)</h3>
    {table(["Task", "Beside", "Slowdown", "Paired runs"], MATRIX)}
    <h3 id="b3">B.3 LLM anatomy per model (Fig. 8)</h3>
    {table(["Model", "Phase", "tokens/s", "DRAM read", "Tensor", "SMs active", "Decode bandwidth GB/s"], X2_ROWS)}
    <p>Partial offload, 7B decode (layers on the GPU, out of 28):</p>
    {table(["Layers on GPU", "tokens/s", "GPU busy"], OFFLOAD)}
    <h3 id="b4">B.4 Kernel mix (what the GPU runs inside each task)</h3>
    {table(["Task", "Top kernel categories by time", "Median kernel µs", "Launches / s", "GPU busy in its window"], KMIX)}
    <h3 id="b5">B.5 Device counters per task alone (Fig. 8)</h3>
    {table(["Task", "Run", "DRAM read", "Tensor", "SMs active", "GPU busy"], REGIME)}
    <h3 id="b6">B.6 A, B and A + B (Fig. 1–2)</h3>
    {table(["Pair", "GPU time share A", "GPU time share B"], SHARES)}
    {table(["Pair", "Counter", "A", "B", "A+B measured", "Time-split", "Sum"], X7)}
    <h3 id="b7">B.7 Hardware ceilings</h3>
    {table(["Measurement", "Median", "Runs"], CEIL_ROWS)}
    <h3 id="b8">B.8 Starting a task (Fig. 9–13)</h3>
    {table(["Launch method", "Median ms", "p10", "p90", "Overhead ms", "n"], START)}
    {table(["Launch method", "Client → created", "Created → started", "Started → first instruction", "Exit → returned", "Remove"], PHASES)}
    {table(["Launch method", "Overhead, empty task", "Overhead, 340 ms task"], FIXED)}
    {table(["Model location (7B)", "First token s", "Requests"], LADDER)}
    <p>Fresh Ollama container (0.5B job): {DATA["merge"]["jit"]["none"]["median"]:.1f} s without a shared GPU-code cache vs {DATA["merge"]["jit"]["shared"]["median"]:.1f} s with it, {DATA["merge"]["jit"]["none"]["n"]} runs each.</p>
  </section>

  <footer><p>Generated from the gauge ledger on 2026-10-16. Repo: wonhee (gauge/, profiling/). Data in data/gauge/ (or $GAUGE_HOME). Raw traces stay on the measuring machine.</p></footer>
</div>
'''

EXTRA_CSS = '''
.short ol { margin: 0; padding-left: 22px; display: flex; flex-direction: column; gap: 10px; max-width: 70ch; font-size: 16.5px; }
.short b, .means b, .dolist b, .steps b { font-weight: 600; }
.finding { border-top: 1px solid var(--rule); padding-top: 26px; }
.fnum { font: 500 12px/1 var(--f-mono); letter-spacing: .08em; text-transform: uppercase; color: var(--batched); }
.means { background: var(--panel); border-left: 3px solid var(--batched); padding: 10px 14px; max-width: none; }
.dolist, .steps { margin: 0; padding-left: 20px; display: flex; flex-direction: column; gap: 8px; max-width: 72ch; }
.appendix { border-top: 2px solid var(--ink); padding-top: 26px; gap: 16px; }
.appendix h3 { font: 600 17px/1.3 var(--f-display); margin: 18px 0 0; scroll-margin-top: 16px; }
.gloss { display: grid; grid-template-columns: minmax(120px, max-content) 1fr; gap: 6px 18px; margin: 0; font-size: 14.5px; }
.gloss dt { font-weight: 500; }
.gloss dd { margin: 0; color: var(--muted); }
@media (max-width: 520px) { .gloss { grid-template-columns: 1fr; } .gloss dd { margin-bottom: 6px; } }
.tbl { overflow-x: auto; border: 1px solid var(--rule); background: var(--panel); }
table.data { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
table.data th { text-align: left; font: 500 11.5px/1.3 var(--f-mono); color: var(--muted); padding: 8px 10px; border-bottom: 1px solid var(--rule); white-space: nowrap; }
table.data td { padding: 6px 10px; border-bottom: 1px solid var(--rule); vertical-align: top; }
table.data tr:last-child td { border-bottom: 0; }
'''
style = STYLE.replace("</style>", EXTRA_CSS + "</style>")
script = SCRIPT.replace(SCRIPT[SCRIPT.index("const DATA = ") + 13: SCRIPT.index(";\nconst css")], "__DATA__", 1) if "__DATA__" not in SCRIPT else SCRIPT
page = HEAD + style + "\n\n" + BODY + "\n" + script.replace("__DATA__", json.dumps(DATA))
open(B / "shared-gpu-node.html", "w").write(page)
print(len(page))
