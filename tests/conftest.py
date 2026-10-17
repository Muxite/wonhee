import pytest
from wonhee.config import Settings


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("WONHEE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WONHEE_ENV_FILE", str(tmp_path / "no.env"))
    monkeypatch.setenv("WONHEE_CREATOR_PATH", str(tmp_path / "creator.yaml"))
    monkeypatch.setenv("WONHEE_LLM_PRICE_IN_PER_MTOK", "0")  # tests assert exact costs
    monkeypatch.setenv("WONHEE_LLM_PRICE_OUT_PER_MTOK", "0")
    return Settings()


from wonhee.db.connect import open_db


@pytest.fixture
def conn(settings):
    c = open_db(settings)
    yield c
    c.close()


import shutil, subprocess


@pytest.fixture(scope="session")
def tiny_video(tmp_path_factory):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    out = tmp_path_factory.mktemp("vid") / "tiny.mp4"
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=red:s=320x568:d=1:r=30",
        "-f", "lavfi", "-i", "color=c=blue:s=320x568:d=1:r=30",
        "-f", "lavfi", "-i", "color=c=green:s=320x568:d=1:r=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-filter_complex", "[0][1][2]concat=n=3:v=1:a=0[v]",
        "-map", "[v]", "-map", "3", "-shortest",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out),
    ], check=True)
    return out


import json


class FakeLLM:
    def __init__(self, replies_path, calls_path):
        self.replies_path, self.calls_path = replies_path, calls_path

    def queue(self, **by_schema):
        """fake_llm.queue(ScoutExtract=[{...}, {...}]) — replies are consumed in order per schema."""
        current = json.loads(self.replies_path.read_text()) if self.replies_path.exists() else {}
        for k, v in by_schema.items():
            current.setdefault(k, []).extend(v)
        self.replies_path.write_text(json.dumps(current))

    def calls(self):
        if not self.calls_path.exists():
            return []
        return [json.loads(l) for l in self.calls_path.read_text().splitlines()]


@pytest.fixture
def fake_llm(tmp_path, monkeypatch):
    """Replaces wonhee.llm._backend with a queue keyed by the request's schema name."""
    from wonhee import llm
    f = FakeLLM(tmp_path / "replies.json", tmp_path / "calls.jsonl")
    f.replies_path.write_text("{}")

    def backend(settings, req):
        name = req["response_format"]["json_schema"]["name"]
        replies = json.loads(f.replies_path.read_text())
        queue = replies.get(name) or []
        if not queue:
            raise llm.LLMError(f"fake llm: no reply queued for {name}")
        reply = queue.pop(0)
        f.replies_path.write_text(json.dumps(replies))
        with open(f.calls_path, "a") as fh:
            fh.write(json.dumps(req) + "\n")
        text = reply if isinstance(reply, str) else json.dumps(reply)
        if text.startswith("Error:"):
            raise llm.LLMError(text[:500])
        return {"text": text, "usage": {"prompt_tokens": 100, "completion_tokens": 20, "model": "fake"}}

    monkeypatch.setattr(llm, "_backend", backend)
    return f


@pytest.fixture
def toy_registry():
    """Register toy nodes in the node registry; removed again afterwards."""
    from wonhee.nodes import registry

    added: list[str] = []

    def add(node):
        registry.register(node)
        added.append(node.name)
        return node

    yield add
    for name in added:
        registry.unregister(name)


@pytest.fixture(scope="session")
def tiny_jpg(tmp_path_factory):
    from PIL import Image
    out = tmp_path_factory.mktemp("img") / "tiny.jpg"
    Image.new("RGB", (160, 200), (120, 140, 160)).save(out, quality=90)
    return out
