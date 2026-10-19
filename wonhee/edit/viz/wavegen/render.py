"""Run one WaveSim render in the wavesim Docker image (built from ../vendor/wavesim.Dockerfile: pinned upstream
WaveSim plus the headless/config patch). Needs `docker run --gpus all`, so callers hold the GPU lock."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

def render(params: dict[str, Any], image: str) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        params_path = tmp_path / "params.json"
        out_path = tmp_path / "out.png"
        params_path.write_text(json.dumps(params), encoding="utf-8")

        result = subprocess.run(
            ["docker", "run", "--rm", "--gpus", "all", "-v", f"{tmp_path}:/data",
             image, "/data/params.json", "/data/out.png"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"wavesim_headless failed (exit {result.returncode}):\n"
                                f"stdout: {result.stdout}\nstderr: {result.stderr}")
        if not out_path.exists():
            raise RuntimeError(f"wavesim_headless exited 0 but produced no output:\n{result.stdout}")
        return out_path.read_bytes()
