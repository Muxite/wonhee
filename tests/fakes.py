from pathlib import Path
from wonhee.analyze.speech import Segment, Transcript
from wonhee.pipeline.deps import Deps


class FakeApify:
    def __init__(self, responses=None, cost=0.01):
        self.responses = responses or {}  # actor -> list of item-lists, consumed in order
        self.calls, self.cost, self.kwargs = [], cost, []

    def run(self, actor, run_input, *, max_total_charge_usd=None, max_items=None):
        self.calls.append((actor, run_input))
        self.kwargs.append({"max_total_charge_usd": max_total_charge_usd, "max_items": max_items})
        q = self.responses.get(actor) or [[]]
        return (q.pop(0) if len(q) > 1 else q[0]), self.cost


class FakeSearch:
    def __init__(self, results=None):
        self.results, self.queries = results or [], []

    def search(self, query, count=10):
        self.queries.append(query)
        return list(self.results)


class FakeDownloader:
    def __init__(self, src: Path | None = None, fail: set[str] = frozenset()):
        self.src, self.fail, self.calls = src, set(fail), []

    def download(self, video_url, page_url, dest):
        from wonhee.sources.fetch import DownloadError
        self.calls.append(page_url)
        if page_url in self.fail:
            raise DownloadError("direct: 403; yt-dlp: exit 1")
        dest.write_bytes(self.src.read_bytes())
        return "direct"


class FakeTranscriber:
    def transcribe(self, path):
        return Transcript("en", [Segment(400, 1500, "when the pump hits")])


def fake_deps(**kw):
    return Deps(apify=kw.get("apify", FakeApify()), search=kw.get("search", FakeSearch()),
                downloader=kw.get("downloader", FakeDownloader()), transcriber=kw.get("transcriber", FakeTranscriber()))


def make_item(code, owner, day, plays, audio=None, tags=()):
    item = {"shortCode": code, "url": f"https://www.instagram.com/reel/{code}/", "ownerUsername": owner,
            "ownerId": owner, "timestamp": f"{day}T12:00:00.000Z", "videoPlayCount": plays,
            "hashtags": list(tags), "caption": f"caption {code}", "videoUrl": f"https://cdn/{code}.mp4"}
    if audio:
        item["musicInfo"] = {"audio_id": audio, "song_name": "s", "artist_name": "a"}
    return item
