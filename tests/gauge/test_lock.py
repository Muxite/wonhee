import os
import stat
import subprocess

from gauge import runner


def test_token_starting_with_dash_releases(tmp_path, monkeypatch):
    """Lock tokens are urlsafe base64 and may start with '-'; argparse must not read it as a flag."""
    log = tmp_path / "calls"
    fake = tmp_path / "gpu-lock"
    fake.write_text(f"""#!/usr/bin/env python3
import argparse, json, sys
open({str(log)!r}, "a").write(" ".join(sys.argv[1:]) + "\\n")
if sys.argv[1] == "acquire":
    print(json.dumps({{"pid": 1, "token": "-dashfirst"}}, indent=2)); sys.exit(0)
ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--token"); ap.add_argument("--force", action="store_true")
ap.parse_args()
""")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(runner, "GPU_LOCK", str(fake))
    with runner.gpu_lock("t", True) as tok:
        assert tok == "-dashfirst"
    assert log.read_text().splitlines()[-1] == "release --token=-dashfirst"
