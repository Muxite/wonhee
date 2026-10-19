"""publish_muksite against a fake Supabase Storage (httpx.MockTransport): object paths, the overwrite refusal, the
YAML entry, missing keys."""
import httpx
import pytest
import yaml
from PIL import Image

from tests.fakes import fake_deps
from tests.test_nodes_subs import flags, out, run_chain
from wonhee import media, muksite
from wonhee.db import repo
from wonhee.nodes import publish as pubn

URL = "https://proj.supabase.co"


class FakeStorage:
    def __init__(self):
        self.objects: dict[str, tuple[bytes, str]] = {}
        self.requests: list[httpx.Request] = []

    def handler(self, req: httpx.Request) -> httpx.Response:
        self.requests.append(req)
        assert req.headers["authorization"] == "Bearer svc" and req.headers["apikey"] == "svc"
        path = req.url.path.removeprefix(f"/storage/v1/object/{muksite.BUCKET}/")
        if path in self.objects and req.headers.get("x-upsert") != "true":
            return httpx.Response(400, json={"statusCode": "409", "error": "Duplicate",
                                             "message": "The resource already exists"})
        self.objects[path] = (req.content, req.headers["content-type"])
        return httpx.Response(200, json={"Key": f"{muksite.BUCKET}/{path}"})


@pytest.fixture
def storage(monkeypatch):
    fake = FakeStorage()
    real = httpx.Client
    monkeypatch.setattr(pubn.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(fake.handler)))
    monkeypatch.setenv("SUPABASE_URL", URL)
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "svc")
    return fake


def _png(conn, settings, tmp_path, name, colour):
    p = tmp_path / f"{name}.png"
    Image.new("RGB", (8, 4), colour).save(p)
    return repo.save_file(conn, settings.media_dir, p, "image", "image/png")


PARAMS = {"slug": "awork", "name": "dependency-graph", "caption": "Backend graph: 432 files", "alt": "Graph"}


def test_publishes_light_and_dark_and_returns_the_yaml_entry(conn, settings, tmp_path, storage):
    light, dark = _png(conn, settings, tmp_path, "l", "white"), _png(conn, settings, tmp_path, "d", "black")
    rid, status = run_chain(conn, settings, [{"id": "p", "node": "publish_muksite",
                                              "in": {"image": light, "dark": dark}, "params": PARAMS}], fake_deps())
    assert status == "done"
    assert set(storage.objects) == {"projects/awork/dependency-graph.png", "projects/awork/dependency-graph-dark.png"}
    assert storage.objects["projects/awork/dependency-graph.png"][1] == "image/png"
    assert all("x-upsert" not in r.headers for r in storage.requests)
    text = media.path_for(settings.media_dir, out(conn, rid, "p", "entry")).read_text()
    (entry,) = yaml.safe_load(text)["photos"]
    base = f"{URL}/storage/v1/object/public/portfolio-images/projects/awork/dependency-graph"
    assert entry == {"url": base + ".png", "darkUrl": base + "-dark.png", "caption": PARAMS["caption"],
                     "alt": "Graph", "size": "hero", "fit": "partial"}
    assert list(entry) == ["url", "darkUrl", "caption", "alt", "size", "fit"]  # the order muksite files use
    assert flags(conn, rid, "p").count("PUBLISHED") >= 1


def test_refuses_to_replace_a_live_image_unless_asked(conn, settings, tmp_path, storage):
    sha = _png(conn, settings, tmp_path, "l", "white")
    node = [{"id": "p", "node": "publish_muksite", "in": {"image": sha}, "params": PARAMS}]
    assert run_chain(conn, settings, node, fake_deps())[1] == "done"
    rid, status = run_chain(conn, settings, node, fake_deps())
    assert status == "failed" and "PUBLISH_FAILED" in flags(conn, rid, "p")
    node[0]["params"] = {**PARAMS, "overwrite": True}
    assert run_chain(conn, settings, node, fake_deps())[1] == "done"
    assert storage.requests[-1].headers["x-upsert"] == "true"


def test_missing_keys_fail_before_any_upload(conn, settings, tmp_path, storage, monkeypatch):
    monkeypatch.delenv("SUPABASE_SERVICE_KEY")
    sha = _png(conn, settings, tmp_path, "l", "white")
    rid, status = run_chain(conn, settings, [{"id": "p", "node": "publish_muksite", "in": {"image": sha},
                                              "params": PARAMS}], fake_deps())
    assert status == "failed" and "PUBLISH_FAILED" in flags(conn, rid, "p") and not storage.requests


def test_object_paths_reject_unsafe_names():
    assert muksite.object_path("awork", "graph", ".png", dark=True) == "projects/awork/graph-dark.png"
    for slug, name in (("../x", "g"), ("awork", "a/b"), ("Awork", "g"), ("", "g")):
        with pytest.raises(muksite.PublishError):
            muksite.object_path(slug, name, ".png")


def test_image_kind_is_sniffed_from_bytes():
    assert pubn._kind(b"\x89PNG\r\n\x1a\n....") == (".png", "image/png")
    assert pubn._kind(b"\xff\xd8\xff\xe0") == (".jpg", "image/jpeg")
    assert pubn._kind(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == (".webp", "image/webp")
    with pytest.raises(Exception, match="png, jpg or webp"):
        pubn._kind(b"GIF89a")
