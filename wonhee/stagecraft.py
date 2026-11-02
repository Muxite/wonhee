"""Delegate staged prop creation to Stagecraft, retain trusted delivery ownership here."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Literal

from wonhee.clock import now_iso
from wonhee.db.connect import tx


def _call(command: str, data: dict | None = None, job_id: str | None = None) -> dict:
    repo = Path(os.environ.get('WONHEE_STAGECRAFT_REPO', str(Path(__file__).resolve().parents[2] / 'stagecraft')))
    argv = [str(repo / '.venv/bin/stagecraft'), command]
    if job_id:
        if not re.fullmatch(r'[0-9a-f]{32}', job_id):
            raise ValueError('invalid Stagecraft job ID')
        argv.append(job_id)
    env = {k:v for k,v in os.environ.items() if k in {'PATH', 'HOME', 'STAGECRAFT_HOME'}}
    env.setdefault('STAGECRAFT_HOME', '/mnt/arteta/stagecraft')
    result = subprocess.run(argv, cwd=repo, env=env, input=json.dumps(data) if data is not None else None,
                            capture_output=True, text=True, timeout=30, check=False)
    if result.returncode:
        raise ValueError('Stagecraft rejected the operation: ' + result.stderr.strip()[-1500:])
    return json.loads(result.stdout)


def stagecraft_schema(conn, settings) -> dict:
    """Read production spec/schema and optional custom React app contract before authoring a demo."""
    return _call('schema')


def stagecraft_list(conn, settings, *, project_id: int | None = None, limit: int = 20) -> dict:
    """List recent jobs for the active project. Select a base_job_id before revising an app."""
    from wonhee import creative
    if project_id is None:
        raise ValueError('select a creative project first')
    creative.project_show(conn, settings, project_id=project_id)
    return _call('list', {'project_id': project_id, 'wonhee_home': str(settings.home.resolve()), 'limit': limit})


def stagecraft_benchmarks(conn, settings, *, project_id: int | None = None, limit: int = 5) -> dict:
    """Read saved paragraph HTTP/stdio measurements, paired samples and evidence IDs. Never queues work.
    Reuse profiler_asset_id for a real data replay. Ratios measure local echo RTT, not thousands of agents.
    Missing evidence is explicit; use stagecraft_dataset only when a fresh measurement is requested."""
    from wonhee import creative, media
    from wonhee.profiler_data import ProfilerData
    if project_id is None:
        raise ValueError('select a creative project first')
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError('limit must be 1..20')
    creative.project_show(conn, settings, project_id=project_id)
    rows = conn.execute("SELECT id,ref,filename FROM creative_intake WHERE project_id=? AND status='stored' "
                        "AND filename IN ('plot-data.json','manifest.json') ORDER BY id DESC LIMIT 200",
                        (project_id,)).fetchall()
    def read(ref, maximum):
        path = media.path_for(settings.media_dir, ref)
        if not path.is_file() or path.stat().st_size > maximum:
            raise ValueError('missing or oversized evidence file')
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != ref:
            raise ValueError('evidence hash differs from stored asset')
        return content
    manifests, errors = [], []
    for row in rows:
        if row['filename'] == 'manifest.json':
            try:
                manifest = json.loads(read(row['ref'], 120_000))
                if not isinstance(manifest, dict) or not isinstance(manifest.get('files'), dict):
                    raise ValueError('manifest must contain a files hash map')
                manifests.append((row['id'], manifest))
            except (OSError, ValueError) as exc:
                errors.append({'asset_id':row['id'], 'error':str(exc)[:200]})
    datasets, seen = [], set()
    for row in rows:
        if row['filename'] != 'plot-data.json' or row['ref'] in seen:
            continue
        seen.add(row['ref'])
        try:
            data = ProfilerData.model_validate_json(read(row['ref'], 120_000))
            manifest_id, manifest = next(((i,m) for i,m in manifests
                if m.get('files',{}).get('plot-data.json') == row['ref']), (None, {}))
            evidence = {'plot-data.json':row['id']}
            missing = []
            if manifest_id:
                if (manifest.get('summary') != data.summary.model_dump()
                        or manifest.get('paragraphs') != data.paragraph_count
                        or manifest.get('trials') != data.sample_count_per_transport // data.paragraph_count
                        or manifest.get('rows') != data.sample_count_per_transport * 2
                        or manifest.get('seed') != data.seed):
                    raise ValueError('manifest summary/recipe differs from profiler data')
                evidence['manifest.json'] = manifest_id
                for name in ('paragraph_echo.py','paragraphs.csv','paragraphs.jsonl','samples.csv','pairs.csv'):
                    digest = manifest.get('files',{}).get(name)
                    if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
                        raise ValueError('manifest contains invalid SHA-256 for '+name)
                    asset = conn.execute("SELECT id FROM creative_intake WHERE project_id=? AND ref=? AND status='stored' "
                                         'ORDER BY id DESC LIMIT 1',(project_id,digest)).fetchone()
                    if not asset:
                        missing.append(name)
                    else:
                        read(digest, 25_000_000)
                        evidence[name] = asset['id']
            else:
                missing = ['manifest.json','paragraph_echo.py','paragraphs.csv','paragraphs.jsonl','samples.csv','pairs.csv']
            summary = data.summary.model_dump()
            datasets.append({'profiler_asset_id':row['id'], 'sha256':row['ref'], 'summary':summary,
                'paragraph_count':data.paragraph_count,
                'trials':data.sample_count_per_transport // data.paragraph_count,
                'observations_per_transport':data.sample_count_per_transport,
                'scope':data.scope, 'animation':data.animation,
                'sample_pairs': [p.model_dump() for p in data.pairs[:12]],
                'sample_pairs_note':f'First {min(12,len(data.pairs))} saved replay pairs; sample_stride={data.sample_stride}. Summary uses all observations.',
                'method':{k:manifest.get(k) for k in ('method','http','stdio','timing','payload_bytes')},
                'evidence_asset_ids':evidence, 'missing_evidence':missing,
                'evidence_status':'hash-linked source and raw samples' if not missing else 'incomplete stored evidence',
                'supported_claim':f"Local paragraph echo median RTT: stdio {summary['stdio']['median_us']:.3f}µs vs "
                    f"HTTP {summary['http']['median_us']:.3f}µs; HTTP/stdio median RTT ratio "
                    f"{summary['http_over_stdio_median_rtt']:.3f}× on this workload.",
                'claim_limits':['Does not measure a wrapper unless that wrapper was executed in this experiment.',
                    'Does not measure live mesh/LLM/agent latency or thousands of concurrent agents.',
                    'Graph scale and replay timing can be theatrical; measured profiler values remain unchanged.']})
            if len(datasets) == limit:
                break
        except (OSError, ValueError, TypeError) as exc:
            errors.append({'asset_id':row['id'], 'error':str(exc)[:200]})
    return {'datasets':datasets, 'errors':errors[:20],
            'next_step':'Reuse profiler_asset_id for a requested video; stagecraft_dataset measures fresh data. Discovery starts no work.'}


def stagecraft_build(conn, settings, *, brief: str, title: str,
                     duration_seconds: int | None = None, surface: Literal['browser','terminal','hybrid'] | None = None,
                     orientation: Literal['landscape','portrait'] | None = None,
                     primitive: Literal['stdio_mesh','scene'] | None = None,
                     presentation: Literal['standard','mesh_story'] | None = None,
                     width: int | None = None, height: int | None = None,
                     dashboard_aspect_ratio: str | None = None, caption_above_dashboard: bool | None = None,
                     forbidden_text: list[str] | None = None, base_job_id: str | None = None,
                     caption: str | None = None, prepare_only: bool = False, profiler_asset_id: int | None = None,
                     benchmark: Literal['auto','refresh','off'] = 'auto',
                     reference_asset_ids: list[int] | None = None, project_id: int | None = None,
                     request_key: str = '', route: dict | None = None) -> dict:
    """Queue a build or revision with requested requirements (not yet validated footage).
    Explicit width/height override orientation. Omitted revision settings/references inherit.
    prepare_only=True saves raw assets without caption finishing or delivery.
    presentation=mesh_story uses a fixed narrative dashboard (1080x1440) with primitive=scene; no coding needed.
    Caption finishing belongs to Wonhee; profiler_asset_id supplies measured graph-ready data.
    Paragraph data is measured automatically when needed; benchmark=refresh remeasures, off disables it.
    For fresh data OMIT profiler_asset_id; use benchmark='auto' or 'refresh'. Never guess asset IDs.
    Host binds project/request/destination. No scripts, JSON imports or manual source edits are needed."""
    from wonhee import creative
    brief=creative._text(brief,'brief',10000)
    title=creative._text(title,'title',120)
    base_job_id = base_job_id or None
    if prepare_only and caption:
        raise ValueError('prepare_only needs a raw desktop, without a finishing caption')
    finishing_caption = None if prepare_only else title if caption is None else (creative._text(caption,'caption',1000) if caption else None)
    if finishing_caption:
        creative._caption_options('desktop_caption', {}, finishing_caption, None)
    if project_id is None or not request_key:
        raise ValueError('select a project and supply a source request key')
    creative.project_show(conn,settings,project_id=project_id)
    if (width is None) != (height is None):
        raise ValueError('width and height must be supplied together')
    base = _call('base-input', {'job_id':base_job_id,'project_id':project_id,
                              'wonhee_home':str(settings.home.resolve())}) if base_job_id else None
    spec = dict(base['settings']) if base else {}
    # Older social compositions become desktop revisions unless explicitly retained.
    if caption_above_dashboard is None:
        spec['caption_above_dashboard']=False
    if (base and width is None and orientation is None and spec.get('surface','browser')=='browser'
            and spec.get('height',0)>spec.get('width',0) and caption_above_dashboard is not True
            and spec.get('presentation')!='mesh_story' and presentation!='mesh_story'):
        spec.update(width=1280,height=800)
    if benchmark not in {'auto','refresh','off'}:
        raise ValueError('benchmark must be auto, refresh or off')
    if profiler_asset_id is not None and benchmark=='refresh':
        raise ValueError('choose a supplied profiler dataset or refresh, not both')
    if presentation is not None:
        if presentation not in {'standard','mesh_story'}:
            raise ValueError('unsupported presentation')
        spec['presentation']=presentation
    story=spec.get('presentation')=='mesh_story'
    if story:
        spec.update(component=None,css=None,coding=None,actions=[],expect={},graph_nodes=0)
        spec['initial']={k:v for k,v in spec.get('initial',{}).items() if k!='presentation'}
        if presentation=='mesh_story' and width is None and orientation is None:
            spec.update(width=1080,height=1440)
        if dashboard_aspect_ratio is None:
            spec['dashboard_aspect_ratio']=f"{spec.get('width',1080)}:{spec.get('height',1440)}"
    if story and primitive not in {None,'scene'}:
        raise ValueError('mesh_story uses primitive=scene; no coding dispatch is needed')
    spec.update(premise=brief,title=title)
    if duration_seconds is not None:
        if isinstance(duration_seconds,bool) or not isinstance(duration_seconds,int) or not 1<=duration_seconds<=45:
            raise ValueError('duration_seconds must be 1..45')
        new_duration=duration_seconds*1000
        old_duration=spec.get('duration_ms',new_duration)
        if new_duration != old_duration:
            spec['actions']=[{**action,'at_ms':min(new_duration-1,action['at_ms']*new_duration//old_duration)}
                             for action in spec.get('actions',[])]
        spec['duration_ms']=new_duration
    spec.setdefault('duration_ms',20000)
    if surface is not None:
        spec['surface']=surface
    spec.setdefault('surface','browser')
    if orientation is not None and orientation not in {'landscape','portrait'}:
        raise ValueError('unsupported orientation')
    if width is not None:
        spec.update(width=width,height=height)
    elif orientation is not None or not base:
        w,h=(1080,1440) if story and orientation is None else ((1080,1920) if orientation=='portrait' else (1280,800))
        spec.update(width=w,height=h)
    if story and dashboard_aspect_ratio is None:
        spec['dashboard_aspect_ratio']=f"{spec['width']}:{spec['height']}"
    if primitive is None:
        primitive = 'scene' if story else ('stdio_mesh' if not base or spec.get('coding') else 'scene')
    if primitive not in {'stdio_mesh','scene'}:
        raise ValueError('unsupported primitive')
    spec['coding']={'task':brief,'primitive':'stdio_mesh'} if primitive=='stdio_mesh' else None
    for key,value in [('dashboard_aspect_ratio',dashboard_aspect_ratio),
                      ('caption_above_dashboard',caption_above_dashboard),('forbidden_text',forbidden_text)]:
        if value is not None:
            spec[key]=value
    # Validate before converting references or saving the canonical brief. The queue validates again.
    if spec['surface'] not in {'browser','terminal','hybrid'}:
        raise ValueError('unsupported capture surface')
    for key,lo,hi in [('width',640,1920),('height',480,1920)]:
        value=spec[key]
        if isinstance(value,bool) or not isinstance(value,int) or not lo<=value<=hi or value%2:
            raise ValueError(f'{key} must be an even integer in {lo}..{hi}')
    if story and (spec['duration_ms']!=20000 or spec['width']*4!=spec['height']*3):
        raise ValueError('mesh_story uses 20 seconds and a 3:4 desktop (1080x1440); Wonhee frames the 9:16 reel')
    if spec['surface']=='hybrid' and spec['width']<1200:
        raise ValueError('hybrid capture needs width >= 1200')
    ratio=spec.get('dashboard_aspect_ratio')
    if ratio and not re.fullmatch(r'[1-9][0-9]{0,3}:[1-9][0-9]{0,3}',ratio):
        raise ValueError('dashboard_aspect_ratio must be positive integer W:H')
    phrases=spec.get('forbidden_text',[])
    if len(phrases)>20 or any(not text.strip() or len(text)>500 for text in phrases):
        raise ValueError('forbidden phrases must contain 1..500 characters, at most 20')
    if spec['surface']!='browser' and (ratio or spec.get('caption_above_dashboard') or phrases):
        raise ValueError('rendered layout/text requirements currently require browser capture')
    if profiler_asset_id is not None:
        try:
            asset=creative.media_info(conn,settings,asset_id=profiler_asset_id)
        except KeyError:
            raise ValueError('profiler dataset asset not found; omit profiler_asset_id and use benchmark=auto or refresh to create fresh data') from None
        if asset['project_id']!=project_id:
            raise ValueError('profiler dataset belongs to another creative project')
        dataset_path=Path(asset['path'])
        if asset['mime']!='application/json' or dataset_path.stat().st_size>120_000:
            raise ValueError('profiler dataset must be graph-ready JSON of at most 120 KB')
        content=dataset_path.read_bytes()
        if hashlib.sha256(content).hexdigest()!=asset['ref']:
            raise ValueError('profiler dataset hash differs from selected asset')
        dataset=json.loads(content)
        from wonhee.profiler_data import ProfilerData
        dataset=ProfilerData.model_validate(dataset).model_dump()
        spec['initial']={**spec.get('initial',{}),'benchmark':dataset,
                         'benchmark_source':{'asset_id':profiler_asset_id,'sha256':asset['ref']}}
    references=list(base['references']) if base and reference_asset_ids is None else []
    ids=list(dict.fromkeys(reference_asset_ids or []))
    if len(ids)>8:
        raise ValueError('at most eight references are supported')
    for asset_id in ids:
        asset=creative.media_info(conn,settings,asset_id=asset_id)
        if asset['project_id']!=project_id:
            raise ValueError('reference belongs to another creative project')
        reference={'asset_id':asset_id,'path':asset['path'],'sha256':asset['ref'],'mime':asset['mime']}
        if asset['mime'].startswith('video/'):
            converted=creative.media_info(conn,settings,asset_id=asset_id,analyze=True).get('contact_sheet')
            if not converted:
                raise ValueError('video reference contact-sheet conversion failed')
            from PIL import Image
            with Image.open(converted['path']) as image:
                mime=Image.MIME[image.format]
            reference.update(path=converted['path'],sha256=converted['ref'],mime=mime,
                             source_video={'asset_id':asset_id,'sha256':asset['ref'],'conversion':'contact_sheet'})
        if reference['mime'] not in {'image/jpeg','image/png','image/webp'} or Path(reference['path']).stat().st_size>8*1024*1024:
            raise ValueError('references must resolve to JPEG/PNG/WebP images of at most 8 MiB')
        if hashlib.sha256(Path(reference['path']).read_bytes()).hexdigest()!=reference['sha256']:
            raise ValueError('reference hash differs from the selected asset')
        references.append(reference)
    measure = (benchmark=='refresh' or (benchmark=='auto' and (primitive=='stdio_mesh' or story)
               and spec['surface']=='browser' and not spec.get('initial',{}).get('benchmark')))
    if story and not measure and not spec.get('initial',{}).get('benchmark'):
        raise ValueError('mesh_story needs measured profiler data; choose profiler_asset_id or benchmark=auto')
    if measure or spec.get('initial',{}).get('benchmark'):
        if spec['surface']!='browser':
            raise ValueError('measured profiler videos require browser capture; use stagecraft_dataset for data without a browser video')
        spec['graph_nodes']=0 if story else (spec.get('graph_nodes') or 3000)
        spec.setdefault('dashboard_aspect_ratio',f"{spec['width']}:{spec['height']}")
    creative.project_brief_save(conn,settings,project_id=project_id,content=brief,
                                source={**(route or {}),'request_key':request_key})
    return _call('submit',{'spec':spec,'project_id':project_id,'request_key':request_key,
                          'route':route or {},'wonhee_home':str(settings.home.resolve()),'references':references,
                          'base_job_id':base_job_id,'caption':finishing_caption,'prepare_only':prepare_only,
                          'benchmark':{'paragraph_count':512,'trials':5,'seed':24026} if measure else None})


def stagecraft_dataset(conn, settings, *, title: str = 'Paragraph HTTP/stdio dataset',
                       paragraph_count: int = 512, trials: int = 5, seed: int = 24026,
                       project_id: int | None = None, request_key: str = '', route: dict | None = None) -> dict:
    """Queue measured random-paragraph HTTP/stdio data and deliver its ZIP. No code or file import required.
    Returns a job ID immediately; use stagecraft_show for status, summary and project-owned asset IDs.
    For a video with a profiler, stagecraft_build creates this dataset automatically instead."""
    from wonhee import creative
    if project_id is None or not request_key:
        raise ValueError('select a project and supply a source request key')
    creative.project_show(conn,settings,project_id=project_id)
    title=creative._text(title,'title',120)
    if (type(paragraph_count) is not int or not 8<=paragraph_count<=2000
            or type(trials) is not int or not 1<=trials<=10 or type(seed) is not int):
        raise ValueError('paragraph_count must be8..2000, trials1..10, seed an integer')
    return _call('submit',{'operation':'dataset','project_id':project_id,'request_key':request_key,
        'route':route or {},'wonhee_home':str(settings.home.resolve()),
        'spec':{'title':title,'premise':'Measure identical random paragraph JSON echoes through local HTTP and stdio.'},
        'benchmark':{'paragraph_count':paragraph_count,'trials':trials,'seed':seed}})


def stagecraft_submit(conn, settings, *, spec: dict, project_id: int, request_key: str,
                      route: dict | None = None) -> dict:
    """Create one durable containerized demo job. Author title/premise, initial state and timed actions;
    optional component/css implements a custom React scene. Set spec.coding={task, primitive:'stdio_mesh'}
    to dispatch Codex to author a real stdio program and dashboard; independent Docker checks gate capture.
    Capture returns through the existing outbox.
    Reuse request_key on timeout; a changed idea is a new request. Queued does not mean delivered."""
    from wonhee import creative
    creative.project_show(conn, settings, project_id=project_id)
    return _call('submit', {'spec':spec,'project_id':project_id,'request_key':request_key,
                            'route':route or {},'wonhee_home':str(settings.home.resolve())})


def stagecraft_show(conn, settings, *, job_id: str, project_id: int | None = None) -> dict:
    """Inspect one Stagecraft job: phase, capture result, failure and staged delivery ID. Do not poll in a loop."""
    result = _call('show', job_id=job_id)
    if project_id is not None and result['project_id'] != project_id:
        raise ValueError('Stagecraft job belongs to another creative project')
    linked=conn.execute('SELECT delivery_id,manifest_json FROM stagecraft_delivery WHERE job_id=?',(job_id,)).fetchone()
    if linked:
        from wonhee import creative
        result['delivery']=creative.delivery_show(conn,settings,delivery_id=linked['delivery_id'])
        result['assets']=json.loads(linked['manifest_json']).get('assets',{})
    elif result.get('prepared'):
        result['assets']=result['prepared']['assets']
    if result.get('assets',{}).get('desktop_asset_id'):
        story=result.get('resolved_settings',{}).get('presentation')=='mesh_story'
        result['finishing']={'owner':'wonhee','source_asset_id':result['assets']['desktop_asset_id'],
            'tool':'make_submit','style':'desktop_reel' if story else 'desktop_caption',
            'instruction':'Reuse this source. For multiple cuts use stable variant labels. '
                          'Supply caption or caption_cues for desktop_reel; brief is context only. No rebuild needed.'}
    if result.get('failure'):
        review=_call('review',job_id=job_id)
        result['review']={'diagnostics':review['diagnostics'],'image_asset_ids':[]}
        for item in review['images'][:1]:
            from wonhee import creative
            image_path=Path(item['path'])
            if image_path.stat().st_size>8*1024*1024 or hashlib.sha256(image_path.read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError('review screenshot hash differs')
            asset=creative.inbox_add(conn,settings,source=str(image_path),project_id=result['project_id'],
                source_key=f"stagecraft:{job_id}:review:{item['sha256']}",role='reference',
                metadata={'producer':'stagecraft','job_id':job_id,'failure':result['failure']})
            result['review']['image_asset_ids'].append(asset['id'])
    return result


def stagecraft_review(conn, settings, *, job_id: str,
                      source_file: Literal['Scene.tsx','scene.css','MeshProfiler.tsx','Graph.tsx'] | None = None,
                      project_id: int | None = None) -> dict:
    """Read bounded project source excerpts, state dependencies and actual capture measurements.
    Summary comes first. Use source_file to inspect one of the listed files in more detail.
    Use this to diagnose missing graphs/charts instead of reading blocked host files. Starts no work."""
    status=_call('show',job_id=job_id)
    if project_id is None or status['project_id']!=project_id:
        raise ValueError('select the owning creative project before source review')
    report=_call('review-owned',{'job_id':job_id,'project_id':project_id,'wonhee_home':str(settings.home.resolve())})
    report.pop('images',None)
    sources=report.get('sources',[])
    if source_file is not None:
        if source_file not in {'Scene.tsx','scene.css','MeshProfiler.tsx','Graph.tsx'}:
            raise ValueError('choose a listed source_file')
        sources=[f for f in sources if f['path']==f'app/src/{source_file}']
    else:
        compact=[]
        for file in sources:
            code=file['excerpt']
            anchor=code.find('<MeshProfiler') if file['path'].endswith('Scene.tsx') else code.find('export default')
            start=max(0,anchor-100) if anchor>=0 else 0
            excerpt=code[start:start+1300]
            compact.append({**file,'excerpt':excerpt,'excerpt_offset':start,
                            'truncated':file.get('truncated',False) or len(excerpt)<len(code)})
        sources=compact
    settings_summary=status.get('resolved_settings') or {}
    return {'job':{'id':job_id,'state':status.get('state'),'phase':status.get('phase'),
                   'settings':{k:settings_summary.get(k) for k in ('title','width','height','duration_ms','presentation')}},
            'initial_state':report.get('initial_state'), 'rendered_checks':report.get('rendered_checks'),
            'visual_review':report.get('visual_review'),
            'diagnostics':[{**d,'text':d.get('text','')[-1500:]} for d in report.get('diagnostics',[])],
            'sources':sources,
            'next_step':'Use source_file with a listed basename for up to 5000 characters. No host file access is needed.'}



def stagecraft_retry(conn, settings, *, job_id: str, project_id: int | None = None) -> dict:
    """Recover a failed dataset/delivery while completing the authorized request. Successful capture is reused; exhausted coding needs a revision."""
    stagecraft_show(conn, settings, job_id=job_id, project_id=project_id)
    return _call('retry', job_id=job_id)


def stagecraft_complete(conn, settings, *, job_id: str, source: str, project_id: int, request_key: str,
                        manifest: dict, title: str, route: dict | None = None, caption: str | None = None,
                        prepare_only: bool = False) -> dict:
    """Host-only completion receipt. Atomically deduplicate outbox staging; never claims a transport send."""
    from wonhee import creative
    if not re.fullmatch(r'[0-9a-f]{32}', job_id):
        raise ValueError('invalid Stagecraft job ID')
    path = Path(source).resolve()
    if not path.is_file() or path.suffix != '.mp4':
        raise ValueError('Stagecraft completion needs a real MP4')
    capture = manifest.get('capture', {})
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if manifest.get('capture_status') != 'mechanical_checks_passed' or capture.get('sha256') != digest:
        raise ValueError('Stagecraft capture manifest/hash is invalid')
    route_json = json.dumps(route or {}, sort_keys=True)
    creative.project_show(conn, settings, project_id=project_id)
    if caption is not None and not (prepare_only and caption==''):
        caption = creative._text(caption, 'caption', 1000)

    if prepare_only:
        if caption:
            raise ValueError('prepare_only imports raw footage without caption finishing')
        asset=creative.inbox_add(conn,settings,source=str(path),project_id=project_id,
            source_key=f'stagecraft:{job_id}:prepared',role='raw',
            metadata={'producer':'stagecraft','manifest':manifest,'request_key':request_key})
        if asset['ref']!=digest or asset.get('metadata',{}).get('request_key')!=request_key:
            raise ValueError('prepared source key conflicts with the original capture/request')
        return {'prepared':True,'assets':{'desktop_asset_id':asset['id']},'source_sha256':digest}

    def existing_receipt():
        old = conn.execute('SELECT * FROM stagecraft_delivery WHERE job_id=?', (job_id,)).fetchone()
        if old:
            previous = json.loads(old['manifest_json'])
            original_hash = previous.get('capture', {}).get('sha256')
            previous_caption = previous.get('postproduction', {}).get('caption')
            if (original_hash, old['project_id'], old['request_key'], old['route_json'], previous_caption) != (
                    digest, project_id, request_key, route_json, caption):
                raise ValueError('Stagecraft completion key conflicts with the original output/route/caption')
            return {**creative.delivery_show(conn, settings, delivery_id=old['delivery_id']),
                    'assets':previous.get('assets',{})}
        return None

    if old := existing_receipt():
        return old
    final_path = path
    delivered_manifest = dict(manifest)
    assets = {}
    hero = path.parent / 'hero.png'
    declared = next((item.get('sha256') for item in manifest.get('files',[])
                     if item.get('path') == capture.get('hero')), None)
    if declared and hero.is_file() and hero.stat().st_size <= 8*1024*1024:
        if hashlib.sha256(hero.read_bytes()).hexdigest() != declared:
            raise ValueError('Stagecraft preview hash differs from capture manifest')
        preview = creative.inbox_add(conn,settings,source=str(hero),project_id=project_id,
            source_key=f'stagecraft:{job_id}:preview',role='reference',metadata={'job_id':job_id,'producer':'stagecraft'})
        assets['preview_asset_id'] = preview['id']
    if caption:
        from wonhee import media, service
        from wonhee.pipeline.deps import Deps
        asset = creative.inbox_add(conn, settings, source=str(path), project_id=project_id,
            source_key=f'stagecraft:{job_id}:desktop', role='raw',
            metadata={'producer':'stagecraft', 'manifest':manifest, 'request_key':request_key})
        assets['desktop_asset_id'] = asset['id']
        run_id, _ = service.node_run(conn, settings, Deps.live(settings), 'compose_video',
            {'video': asset['ref']}, {'template':'desktop_caption', 'slots':{'title':caption},
                                    'encoder':'x264', 'tighten':False})
        output = service._step_out(conn, run_id, 'compose_video')
        if not output:
            raise ValueError(f'Wonhee caption composition failed; inspect run {run_id}')
        final_path = media.path_for(settings.media_dir, output['video'])
        delivered_manifest['postproduction'] = {'owner':'wonhee', 'template':'desktop_caption',
            'caption':caption, 'run_id':run_id, 'source_sha256':digest, 'output_sha256':output['video']}
    with tx(conn):
        if old := existing_receipt():
            return old
        asset = creative.inbox_add(conn, settings, source=str(final_path), project_id=project_id,
            source_key=f'stagecraft:{job_id}', role='output', metadata={'producer':'stagecraft','manifest':delivered_manifest,
                                                                      'request_key':request_key})
        assets['final_asset_id']=asset['id']
        assets.setdefault('desktop_asset_id',asset['id'])
        delivered_manifest['assets']=assets
        delivery = creative.outbox_add(conn, settings, ref=asset['ref'], filename=f'stagecraft-{job_id}.mp4',
            project_id=project_id, route=route, content=f'{title[:160]} — staged demo ({manifest.get("mode", "simulated_story")})',
            as_document=True)
        conn.execute('INSERT INTO stagecraft_delivery VALUES(?,?,?,?,?,?,?,?)',
                     (job_id, project_id, request_key, asset['ref'], route_json, delivery['id'],
                      json.dumps(delivered_manifest), now_iso()))
        return {**delivery,'assets':assets}


def stagecraft_dataset_complete(conn, settings, *, job_id: str, project_id: int, request_key: str,
                                dataset: dict, title: str, route: dict | None = None, deliver: bool = False) -> dict:
    """Host-only fixed benchmark registration. Project assets and optional ZIP delivery are idempotent."""
    from wonhee import creative
    from wonhee.profiler_data import ProfilerData
    if not re.fullmatch(r'[0-9a-f]{32}',job_id):
        raise ValueError('invalid dataset job ID')
    creative.project_show(conn,settings,project_id=project_id)
    root=Path(dataset['directory']).resolve()
    names=('paragraph_echo.py','paragraphs.csv','paragraphs.jsonl','samples.csv','pairs.csv','plot-data.json','manifest.json','dataset.zip')
    for name in names:
        path=root/name
        if path.is_symlink() or not path.is_file() or path.stat().st_size>25_000_000:
            raise ValueError('invalid or oversized dataset artifact: '+name)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=dataset['hashes'].get(name):
            raise ValueError('dataset artifact hash differs: '+name)
    parsed=ProfilerData.model_validate_json((root/'plot-data.json').read_text())
    recipe=dataset['recipe']
    if parsed.paragraph_count!=recipe['paragraph_count'] or parsed.sample_count_per_transport!=recipe['paragraph_count']*recipe['trials'] or parsed.seed!=recipe['seed']:
        raise ValueError('dataset does not match the queued recipe')
    assets={}
    for name in names:
        asset=creative.inbox_add(conn,settings,source=str(root/name),project_id=project_id,
            source_key=f'stagecraft:{job_id}:dataset:{name}',role='reference',
            metadata={'producer':'stagecraft_dataset','job_id':job_id,'sha256':dataset['hashes'][name],
                      'scope':parsed.scope,'recipe':recipe})
        assets[name]=asset['id']
    summary=parsed.summary.model_dump()
    creative.memory_remember(conn,settings,project_id=project_id,kind='output',content=(
        f"Measured paragraph dataset ready: {parsed.paragraph_count} unique strings; {recipe['trials']} trials; "
        f"{parsed.sample_count_per_transport*2} verified observations. Median stdio {parsed.summary.stdio.median_us:.3f}µs, "
        f"HTTP {parsed.summary.http.median_us:.3f}µs. Scope: {parsed.scope}. "
        f"Use profiler_asset_id={assets['plot-data.json']} to reuse this measured data in a desktop build."),
        source={'producer':'stagecraft_dataset','job_id':job_id,'plot_asset_id':assets['plot-data.json'],
                'manifest_sha256':dataset['hashes']['manifest.json']})
    result={'assets':assets,'summary':summary,'scope':parsed.scope}
    if deliver:
        route_json=json.dumps(route or {},sort_keys=True)
        digest=dataset['hashes']['dataset.zip']
        with tx(conn):
            old=conn.execute('SELECT * FROM stagecraft_delivery WHERE job_id=?',(job_id,)).fetchone()
            if old:
                if (old['ref'],old['project_id'],old['request_key'],old['route_json'])!=(digest,project_id,request_key,route_json):
                    raise ValueError('dataset completion conflicts with original output/route')
                result['delivery']=creative.delivery_show(conn,settings,delivery_id=old['delivery_id'])
            else:
                asset=creative.media_info(conn,settings,asset_id=assets['dataset.zip'])
                delivery=creative.outbox_add(conn,settings,ref=asset['ref'],filename=f'paragraph-data-{job_id}.zip',
                    project_id=project_id,route=route,as_document=True,content=title+' — measured local paragraph echo dataset')
                receipt={'dataset':dataset,'assets':assets}
                conn.execute('INSERT INTO stagecraft_delivery VALUES(?,?,?,?,?,?,?,?)',
                    (job_id,project_id,request_key,digest,route_json,delivery['id'],json.dumps(receipt),now_iso()))
                result['delivery']=delivery
    return result
