"""Reconcile durable reel submissions, execute recorded node phases and expose private output downloads."""
from __future__ import annotations

import contextlib
import fcntl
import json
import html
import re
import signal
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, urlsplit

from pydantic import BaseModel, ConfigDict, Field

from wonhee import creative, media
from wonhee.db import queries, repo
from wonhee.db.connect import open_db, tx
from wonhee.db.repo import creative as writes


class MakeOptions(BaseModel):
    model_config = ConfigDict(extra='forbid')
    title: str | None = None
    cta: str | None = None
    items: list[str] = Field(default_factory=list)
    slots: dict = Field(default_factory=dict)
    tighten: bool | dict = False
    look: str | None = None
    quality: int = Field(default=20, ge=10, le=35)
    cost_cap_usd: float = Field(default=1, gt=0, le=1)


def _phase(conn, settings, deps, submission, phase: str, nodes: list, cap: float) -> tuple[int, dict]:
    from wonhee.nodes.chain import parse_chain, validate
    from wonhee.nodes import registry
    from wonhee import service
    chain = parse_chain({'nodes': nodes, 'cost_cap_usd': cap})
    validate(chain, registry.all_nodes())
    with tx(conn):
        linked = conn.execute('SELECT run_id FROM creative_run WHERE submission_id=? AND phase=?',
                              (submission['id'], phase)).fetchone()
        if linked:
            run_id = linked[0]
        else:
            intent = repo.create_intent(conn, f"creative {submission['id']} {phase}", 'chain', 1,
                                        chain.model_dump(by_alias=True, mode='json'))
            repo.confirm_intent(conn, intent)
            project = repo.create_project(conn, f"creative {submission['id']} {phase}", 'chain', intent)
            run_id = repo.start_run(conn, project, cap)
            writes.link_run(conn, submission['id'], phase, run_id)
    # Only this locked daemon executes these runs. Pure local phases can safely reuse their current run after loss.
    for step in conn.execute("SELECT id FROM step WHERE run_id=? AND status='running'", (run_id,)).fetchall():
        repo.add_flag(conn, step[0], 'WORKER_LOST', 'creative daemon restarted; reconcile this local phase')
        repo.finish_step(conn, step[0], 'failed')
    service.chain_exec(conn, settings, deps, run_id)
    row = queries.run_row(conn, run_id)
    if row['status'] not in {'done', 'weak'}:
        raise RuntimeError(f"{phase} failed in run {run_id}; inspect wonhee chain show {run_id}")
    outputs = service._step_out(conn, run_id, nodes[-1]['id'])
    if outputs is None:
        raise RuntimeError(f'{phase} has no recorded outputs')
    return run_id, outputs


def execute(conn, settings, deps, submission: dict) -> dict:
    """Each phase is linked to its run in the same transaction that creates the run."""
    from wonhee import service
    from wonhee.edit import reel_slots as slots_engine
    from wonhee.nodes.subtitles import line_words, load_subs
    asset = creative.media_info(conn, settings, asset_id=submission['asset_id'])
    options = MakeOptions.model_validate(submission['params'])
    tpl = service._reel_template(submission['style'])
    speech_run, out = _phase(conn, settings, deps, submission, 'speech',
                             [{'id':'s','node':'speech_subs','in':{'video':asset['ref']}}], options.cost_cap_usd)
    subs = load_subs(settings.media_dir, out['subtitles'])
    words = [w for line in subs.lines for w in line_words(line)[0]]
    transcript = ' '.join(line.text for line in subs.lines).strip()
    if transcript:
        creative.memory_remember(conn, settings, project_id=submission['project_id'], kind='transcript',
                                  content=transcript[:20000], source={'asset_id':asset['id'],'run_id':speech_run})
    lecture = options.slots if submission['style'] == 'lecture' else None
    # Lecture's automatic writing stays a recorded node rather than an invisible model call.
    if submission['style'] == 'lecture' and not (lecture.get('theses') and lecture.get('subtitle')):
        _, lo = _phase(conn, settings, deps, submission, 'writing', [{'id':'l','node':'larp','params':{
            'task': service._lecture_task(options.title, words),'registers':['lecture'],'engine':'local'}}], options.cost_cap_usd)
        lecture = {**json.loads(media.path_for(settings.media_dir,lo['lines']).read_text())['lecture'], **lecture}
    filled, notes = slots_engine.fill_slots(submission['style'], tpl['slots'], words, asset['probe']['duration_ms'],
        title=options.title, cta=options.cta, items=options.items, n_insets=0, lecture=lecture)
    filled.update({k:v for k,v in options.slots.items() if k in tpl['slots']})
    used = conn.execute('SELECT coalesce(sum(s.cost_usd),0) FROM step s JOIN creative_run r ON r.run_id=s.run_id '
                        'WHERE r.submission_id=?', (submission['id'],)).fetchone()[0]
    params = {'template':submission['style'],'slots':filled,'tighten':options.tighten,'quality':options.quality,
              'encoder':'x264'}
    if options.look:
        params['look'] = options.look
    run_id, rendered = _phase(conn, settings, deps, submission, 'render', [{'id':'v','node':'compose_video',
        'in':{'video':asset['ref'],'subtitles':out['subtitles']},'params':params}], max(.000001,options.cost_cap_usd-used))
    ref = rendered['video']
    if media.path_for(settings.media_dir, ref).stat().st_size > 50_000_000:
        try:
            _, preview = _phase(conn, settings, deps, submission, 'preview',
                [{'id': 'p', 'node': 'delivery_preview', 'in': {'video': ref}}], .000001)
            creative.outbox_add(conn, settings, ref=preview['video'],
                filename=f"preview-{submission['id']}.mp4", route=submission['route'],
                content='Preview only (up to 60 seconds). The original is available in the private outbox.',
                project_id=submission['project_id'], submission_id=submission['id'])
        except Exception as exc:
            writes.maintenance(conn, f"preview:{submission['id']}", str(exc)[:500])
    with tx(conn):
        creative.outbox_add(conn, settings, ref=ref, filename=f"reel-{submission['id']}-{submission['style']}.mp4",
            route=submission['route'], content=f"Your {submission['style']} reel is ready. Full file check: not yet run.",
            project_id=submission['project_id'], submission_id=submission['id'])
        writes.update(conn, 'creative_submission', submission['id'], status='done', phase='done', result_ref=ref, error=None)
    creative.memory_remember(conn, settings, project_id=submission['project_id'], kind='output',
        content=f"Finished {submission['style']} reel {submission['id']}. Title: {options.title or ''}. " + ' '.join(notes),
        source={'submission_id':submission['id'],'run_id':run_id,'ref':ref})
    return creative.make_show(conn, settings, submission_id=submission['id'])


@contextlib.contextmanager
def gpu_lease(description: str):
    lock = Path('/home/muk/projects/gpu-lock')
    acquired = subprocess.run([str(lock),'acquire',description,'--wait','--ttl','3600'], capture_output=True,text=True)
    if acquired.returncode:
        raise RuntimeError('cannot acquire shared GPU lock: '+acquired.stderr[-300:])
    token = json.loads(acquired.stdout)['token']
    try:
        yield
    finally:
        subprocess.run([str(lock),'release',f'--token={token}'], capture_output=True,text=True)


def _handler(settings):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            conn = open_db(settings)
            try:
                if path in ('/', '/health'):
                    if path == '/health':
                        body = json.dumps(creative.health(conn, settings)).encode()
                        mime = 'application/json'
                    else:
                        outputs = creative.outbox_list(conn, settings)['outputs']
                        links = ''.join(
                            f'<p><a href="/files/{r["token"]}">{html.escape(r["filename"])}</a>'
                            f' — {r["bytes"]} bytes; full check unknown</p>' for r in outputs)
                        body = ('<!doctype html><meta name="viewport" content="width=device-width">'
                                '<title>Wonhee outbox</title><h1>Wonhee outbox</h1>' + links).encode()
                        mime = 'text/html; charset=utf-8'
                    self.send_response(200)
                    self.send_header('Content-Type', mime)
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                token = path.removeprefix('/files/')
                if not path.startswith('/files/') or not re.fullmatch('[a-f0-9]{32}', token):
                    self.send_error(404)
                    return
                row = conn.execute('SELECT * FROM creative_delivery WHERE token=?', (token,)).fetchone()
                if row is None:
                    self.send_error(404)
                    return
                file = media.path_for(settings.media_dir, row['ref'])
                if not file.is_file():
                    self.send_error(404, 'The registered output is missing from storage')
                    return
                size = file.stat().st_size
                start, end = 0, size - 1
                partial = self.headers.get('Range')
                if partial:
                    match = re.fullmatch(r'bytes=(\d*)-(\d*)', partial)
                    if not match or not any(match.groups()):
                        self.send_error(416)
                        return
                    if not match[1]:
                        start = max(0, size - int(match[2]))
                    else:
                        start = int(match[1])
                        end = min(size - 1, int(match[2])) if match[2] else size - 1
                    if start > end or start >= size:
                        self.send_error(416)
                        return
                self.send_response(206 if partial else 200)
                self.send_header('Content-Type', row['mime'])
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Content-Length', str(end - start + 1))
                self.send_header('Cache-Control', 'private, no-store')
                self.send_header('Content-Disposition', "attachment; filename*=UTF-8''" + quote(row['filename']))
                if partial:
                    self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
                self.end_headers()
                with file.open('rb') as stream:
                    stream.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        chunk = stream.read(min(1 << 20, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                conn.close()

        def log_message(self, format, *args):
            pass  # Download tokens and resource names do not belong in access logs.
    return Handler


def serve(settings, *, host: str | None = None, once: bool = False) -> None:
    """One daemon owns submissions; HTTP binds only to the tailnet unless explicitly overridden."""
    settings.home.mkdir(parents=True, exist_ok=True)
    with (settings.home / 'creative-worker.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('creative worker is already running') from None
        if host is None:
            result = subprocess.run(['tailscale', 'ip', '-4'], capture_output=True, text=True, check=True)
            ips = result.stdout.strip().splitlines()
            if not ips:
                raise ValueError('no Tailscale IPv4 address is available')
            host = ips[0]
        server = None
        if not once:
            server = ThreadingHTTPServer((host, settings.creative_port), _handler(settings))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            from wonhee.fsutil import write_atomic
            write_atomic(settings.home / 'creative-outbox' / 'endpoint.json',
                         json.dumps({'url': settings.creative_download_url or f'http://{host}:{settings.creative_port}'}))
        stop = threading.Event()

        def request_stop(*_):
            stop.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, request_stop)
        conn = open_db(settings)
        writes.recover_deliveries(conn)
        from wonhee.pipeline.deps import Deps
        try:
            while not stop.is_set():
                row = writes.claim_submission(conn)
                if row is not None:
                    submission = creative._record(row)
                    try:
                        if submission['attempts'] >= 3:
                            raise RuntimeError('submission interrupted three times; inspect and resubmit explicitly')
                        with gpu_lease(f"wonhee: creative submission {row['id']}"):
                            # A writing node must not leave a shared Ollama GPU resident after this lease.
                            job_settings = settings.model_copy(update={'local_llm_keep_alive': '0s'})
                            deps = Deps.live(job_settings)
                            try:
                                execute(conn, job_settings, deps, submission)
                            finally:
                                del deps
                                import gc
                                import sys
                                gc.collect()
                                if 'torch' in sys.modules:
                                    sys.modules['torch'].cuda.empty_cache()
                    except Exception as exc:
                        writes.update(conn, 'creative_submission', row['id'], status='failed',
                                      error=f'{type(exc).__name__}: {exc}'[:1000])
                creative._export_after_write(conn, settings)
                creative._index_pending(conn, settings)
                if once:
                    break
                stop.wait(1)
        finally:
            conn.close()
            if server:
                server.shutdown()
                server.server_close()
