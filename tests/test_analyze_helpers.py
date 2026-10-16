import pytest
from wonhee.analyze import frames, phash, scenes, speech


def test_detect_three_shots(tiny_video):
    shots = scenes.detect_shots(tiny_video)
    assert len(shots) == 3
    assert shots[0][0] == 0 and 900 <= shots[0][1] <= 1100


def test_frame_and_phash(tiny_video, tmp_path):
    a = frames.extract_frame(tiny_video, 500, tmp_path / "a.jpg")
    b = frames.extract_frame(tiny_video, 1500, tmp_path / "b.jpg")
    ha, hb = phash.phash_hex(a), phash.phash_hex(b)
    assert len(ha) == 16 and ha == phash.phash_hex(a)
    # flat red vs blue frames have identical phash (no structure); hash is luminance-structure only
    assert len(hb) == 16


def test_first_word():
    t = speech.Transcript("en", [speech.Segment(1200, 2000, " hi"), speech.Segment(2000, 3000, "there")])
    assert speech.first_word_ms(t) == 1200
    assert speech.first_word_ms(speech.Transcript("en", [speech.Segment(0, 500, "  ")])) is None


@pytest.mark.gpu
def test_whisper_runs_on_gpu(tiny_video):
    t = speech.WhisperTranscriber("tiny", "cuda", "float16").transcribe(tiny_video)
    assert isinstance(t.segments, list)


def test_extract_frame_past_end(tiny_video, tmp_path):
    with pytest.raises(RuntimeError):
        frames.extract_frame(tiny_video, 60_000, tmp_path / "x.jpg")


def test_detect_shots_no_cut_fallback(tiny_video, monkeypatch):
    monkeypatch.setattr(scenes, "detect", lambda *a, **k: [])
    dur = scenes.probe(tiny_video).duration_ms
    assert scenes.detect_shots(tiny_video) == [(0, dur)]


def test_cuda_device_missing(monkeypatch):
    import ctranslate2
    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 0)
    with pytest.raises(RuntimeError, match="no CUDA device"):
        speech.WhisperTranscriber("tiny", "cuda", "float16")._load()


def test_preload_missing_package(monkeypatch):
    import importlib.util
    monkeypatch.setattr(speech, "_preloaded", False)
    def boom(name):
        raise ModuleNotFoundError(name)
    monkeypatch.setattr(importlib.util, "find_spec", boom)
    with pytest.raises(RuntimeError, match="pip install nvidia-cublas-cu12"):
        speech._preload_cuda_libs()


def _fake_pkgs(monkeypatch, tmp_path, files):
    import importlib.util
    from types import SimpleNamespace
    for pkg in ("cublas", "cudnn"):
        (tmp_path / pkg / "lib").mkdir(parents=True)
    for pkg, name in files:
        (tmp_path / pkg / "lib" / name).write_text("")
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda m: SimpleNamespace(submodule_search_locations=[str(tmp_path / m.split(".")[1])]))
    monkeypatch.setattr(speech, "_preloaded", False)


def test_preload_no_match(monkeypatch, tmp_path):
    _fake_pkgs(monkeypatch, tmp_path, [])
    with pytest.raises(RuntimeError, match="no files matching"):
        speech._preload_cuda_libs()
    assert speech._preloaded is False


def test_preload_idempotent(monkeypatch, tmp_path):
    _fake_pkgs(monkeypatch, tmp_path, [("cublas", "libcublasLt.so.12"), ("cublas", "libcublas.so.12"),
                                       ("cudnn", "libcudnn.so.9")])
    calls = []
    monkeypatch.setattr(speech.ctypes, "CDLL", lambda p, mode=None: calls.append(p))
    speech._preload_cuda_libs()
    speech._preload_cuda_libs()
    assert len(calls) == 3 and len(set(calls)) == 3
    assert speech._preloaded is True


@pytest.fixture(scope="session")
def silent_video(tmp_path_factory):
    import shutil, subprocess
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    out = tmp_path_factory.mktemp("vid") / "noaudio.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=320x568:d=1:r=30",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


def test_no_audio_stream_is_empty_transcript_not_crash(silent_video, tiny_video):
    from wonhee.media import has_audio
    assert has_audio(tiny_video) and not has_audio(silent_video)
    t = speech.WhisperTranscriber("tiny", "cuda", "float16")
    t._load = lambda: pytest.fail("model must not load for a video with no audio")
    out = t.transcribe(silent_video)
    assert out.segments == [] and out.language is None
