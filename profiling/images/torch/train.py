"""Bounded training job: fixed steps on synthetic data, so cost depends only on the declared config."""
import argparse
import json
import sys
import time

import torch
import torchvision

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="resnet18")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--image", type=int, default=224)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--amp", type=int, default=0)
a = ap.parse_args()

dev = "cuda"
model = getattr(torchvision.models, a.model)(num_classes=1000).to(dev)
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
x = torch.randn(a.batch, 3, a.image, a.image, device=dev)
y = torch.randint(0, 1000, (a.batch,), device=dev)
torch.cuda.synchronize()
print("GAUGE_MARK model_loaded", flush=True)


def step():
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(a.amp)):
        loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    opt.step()


try:
    step()  # warmup: cudnn autotune + allocator growth
    torch.cuda.synchronize()
    print("GAUGE_MARK compute_started", flush=True)
    t0 = time.time()
    for _ in range(a.steps):
        step()
    torch.cuda.synchronize()
    dt = time.time() - t0
except torch.OutOfMemoryError as e:
    print("GAUGE_MARK cuda_oom", flush=True)
    print(f"cuda oom: {str(e).splitlines()[0]}", flush=True)
    sys.exit(3)
print("GAUGE_MARK compute_finished", flush=True)
print("GAUGE_OUTCOME " + json.dumps({"imgs_per_s": a.batch * a.steps / dt, "step_s": dt / a.steps,
                                     "torch_max_allocated": torch.cuda.max_memory_allocated(),
                                     "torch_max_reserved": torch.cuda.max_memory_reserved()}), flush=True)
