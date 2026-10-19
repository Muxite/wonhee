"""Hold one GPU resource busy for a fixed time, as a targeted co-runner (experiment C3).

  bw       device-to-device copies of 1 GiB: DRAM bandwidth, almost no math
  compute  fp16 matmuls on 4096^2 tiles that stay in L2-sized working sets: tensor cores, little DRAM traffic
  launch   tiny kernels back to back: launch / scheduling overhead, no bandwidth or math to speak of

Reports its own rate (work units per second) and a cumulative series every 0.5 s (unix time, units), so its
slowdown can be measured inside a partner's window, not diluted by the time it ran alone.
usage: stressor.py MODE SECONDS
"""
import json
import sys
import time

import torch

mode, seconds = sys.argv[1], float(sys.argv[2])
dev = "cuda"
if mode == "bw":
    x = torch.empty(2**30, dtype=torch.uint8, device=dev); y = torch.empty_like(x)
    step, unit, per = (lambda: y.copy_(x)), "GB", 2 * 2**30 / 1e9  # read + write
elif mode == "compute":
    a = torch.randn(4096, 4096, device=dev, dtype=torch.float16); b = torch.randn_like(a)
    step, unit, per = (lambda: a @ b), "TFLOP", 2 * 4096**3 / 1e12
elif mode == "launch":
    z = torch.zeros(1, device=dev)
    step, unit, per = (lambda: z.add_(1)), "kernel", 1
else:
    raise SystemExit(f"unknown mode {mode}")
step(); torch.cuda.synchronize()
print("GAUGE_MARK compute_started", flush=True)
t0, n, series = time.perf_counter(), 0, []
last = t0
while time.perf_counter() - t0 < seconds:
    for _ in range(50):
        step()
    torch.cuda.synchronize()
    n += 50
    now = time.perf_counter()
    if now - last >= 0.5:  # cumulative work over wall-clock time: the rate inside any partner's window
        series.append([round(time.time(), 3), round(n * per, 3)])
        last = now
dt = time.perf_counter() - t0
print("GAUGE_MARK compute_finished", flush=True)
print("GAUGE_OUTCOME " + json.dumps({"mode": mode, "seconds": dt, "units": n * per, "unit": unit,
                                     "rate": n * per / dt, "series": series}), flush=True)
