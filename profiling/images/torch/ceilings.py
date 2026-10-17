"""Hardware ceilings: what this GPU can do at best, to normalize every workload against.
Prints one GAUGE_OUTCOME per measurement."""
import json
import time

import torch

dev = "cuda"


def emit(**kw):
    print("GAUGE_OUTCOME " + json.dumps(kw), flush=True)


def timed(fn, reps):
    fn(); torch.cuda.synchronize()  # warmup
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / 1000 / reps


print("GAUGE_MARK compute_started", flush=True)
# 1. dense matmul throughput per number format (n x n x n, 2n^3 flops)
n = 8192
for name, dtype, tf32 in [("fp32", torch.float32, False), ("tf32", torch.float32, True), ("fp16", torch.float16, False),
                          ("bf16", torch.bfloat16, False)]:
    torch.backends.cuda.matmul.allow_tf32 = tf32
    a = torch.randn(n, n, device=dev, dtype=dtype); b = torch.randn(n, n, device=dev, dtype=dtype)
    t = timed(lambda: a @ b, 10)
    emit(test="matmul", format=name, n=n, seconds=t, tflops=2 * n**3 / t / 1e12)
    del a, b
try:
    a = torch.randn(n, n, device=dev).to(torch.float8_e4m3fn); b = torch.randn(n, n, device=dev).to(torch.float8_e4m3fn).t()
    one = torch.tensor(1.0, device=dev)
    t = timed(lambda: torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16), 10)
    emit(test="matmul", format="fp8_e4m3", n=n, seconds=t, tflops=2 * n**3 / t / 1e12)
except Exception as ex:  # unsupported on this build/GPU: record why
    emit(test="matmul", format="fp8_e4m3", n=n, error=str(ex)[:200])
torch.backends.cuda.matmul.allow_tf32 = False
# 2. matrix-vector (the shape of LLM decode): memory-bound, reads the whole matrix once per call
for rows in (4096, 8192, 16384):
    W = torch.randn(rows, rows, device=dev, dtype=torch.float16); x = torch.randn(rows, 1, device=dev, dtype=torch.float16)
    t = timed(lambda: W @ x, 50)
    emit(test="gemv", format="fp16", rows=rows, seconds=t, gb_per_s=W.numel() * 2 / t / 1e9, tflops=2 * rows * rows / t / 1e12)
    del W, x
# 3. device memory bandwidth: copy 1 GiB (reads 1 GiB + writes 1 GiB)
x = torch.empty(2**30, dtype=torch.uint8, device=dev); y = torch.empty_like(x)
t = timed(lambda: y.copy_(x), 20)
emit(test="device_copy", bytes=2**30, seconds=t, gb_per_s=2 * 2**30 / t / 1e9)
del x, y
# 4. host <-> device over PCIe, pinned and pageable, 1 GiB
for pinned in (True, False):
    h = torch.empty(2**30, dtype=torch.uint8, pin_memory=pinned); d = torch.empty(2**30, dtype=torch.uint8, device=dev)
    t = timed(lambda: d.copy_(h, non_blocking=pinned), 5)
    emit(test="h2d", pinned=pinned, bytes=2**30, seconds=t, gb_per_s=2**30 / t / 1e9)
    t = timed(lambda: h.copy_(d, non_blocking=pinned), 5)
    emit(test="d2h", pinned=pinned, bytes=2**30, seconds=t, gb_per_s=2**30 / t / 1e9)
    del h, d
# 5. kernel launch cost: many tiny kernels back to back (the regime of small-batch decode)
z = torch.zeros(1, device=dev)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(20000):
    z.add_(1)
torch.cuda.synchronize(); dt = time.perf_counter() - t0
emit(test="launch", kernels=20000, seconds=dt, us_per_launch=dt / 20000 * 1e6)
print("GAUGE_MARK compute_finished", flush=True)
