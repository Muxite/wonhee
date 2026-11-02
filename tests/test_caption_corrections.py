"""Production feedback, direct Moeka design §Caption corrections (2026-10-26)."""
import subprocess
from pathlib import Path

import pytest

from wonhee import creative, creative_worker, service
from wonhee.edit import compose as C
from tests.fakes import fake_deps


@pytest.fixture(autouse=True)
def no_vectors(monkeypatch):
    monkeypatch.setattr(creative, '_docstore', lambda *a, **k: None)


def setup_source(conn, settings, tiny_video):
    landscape = tiny_video.parent / 'landscape.mp4'
    if not landscape.exists():
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(tiny_video),'-vf','transpose=1',
                        '-c:v','libx264','-threads','1','-an',str(landscape)],check=True)
    tiny_video = landscape
    project = creative.project_create(conn, settings, name='caption corrections')
    asset = creative.inbox_add(conn, settings, source=str(tiny_video), project_id=project['id'])
    return project, asset


def test_brief_only_caption_fails_before_queue(conn, settings, tiny_video):
    project, asset = setup_source(conn, settings, tiny_video)
    with pytest.raises(ValueError, match='brief is saved context'):
        creative.make_submit(conn, settings, project_id=project['id'], asset_id=asset['id'],
                             request_key='blank', style='desktop_caption', brief='Put a huge caption above this.')
    assert conn.execute('select count(*) from creative_submission').fetchone()[0] == 0


@pytest.mark.parametrize('extra,match', [
    ({'caption': 'copy', 'caption_lines':['different']}, 'not both'),
    ({'caption_lines': []}, 'one to four'),
    ({'caption_lines': [' ']}, 'caption line'),
    ({'params': {'slots': {'title':'copy', 'unknown':'ignored'}}}, 'no slot'),
    ({'caption':'copy', 'params':{'tighten':True}}, 'preserves'),
    ({'caption':'copy', 'params':{'title':'different'}}, 'conflicts'),
    ({'caption':'huge ' * 190}, 'shorten'),
])
def test_invalid_or_unreadable_copy_is_actionable(conn, settings, tiny_video, extra, match):
    project, asset = setup_source(conn, settings, tiny_video)
    with pytest.raises(ValueError, match=match):
        creative.make_submit(conn, settings, project_id=project['id'], asset_id=asset['id'],
                             request_key='bad', style='desktop_caption', **extra)


LINES = ['AGENT MESH, RE-ARCHITECTED.', 'Multiplexed local transport. 5.5× lower median RTT vs HTTP.',
         '512-paragraph local echo benchmark · 5 trials.', 'Workload-specific; not live mesh latency.']


def test_persistent_hierarchy_and_discovery(conn, settings, tiny_video):
    project, asset = setup_source(conn, settings, tiny_video)
    job = creative.make_submit(conn, settings, project_id=project['id'], asset_id=asset['id'],
                               request_key='hierarchy', style='desktop_caption', caption_lines=LINES)
    assert list(job['effective_caption'].values()) == LINES
    assert job['source_asset_id'] == asset['id']
    assert job['phase'] == 'render'
    layout = C.load_template('desktop_caption', job['params']['slots'])
    text = [ly for ly in layout.layers if ly.type == 'text']
    assert len({ly.role for ly in text}) == 4
    assert len({ly.size_frac for ly in text}) == 4
    assert all(ly.t is None and ly.anim == 'none' and ly.min_size_px == 32 for ly in text)
    assert max(ly.box.y + ly.box.h for ly in text) < layout.layers[0].box.y
    assert layout.layers[0].clip_fit == 'contain'
    assert 'caption_lines' in service.reel_styles()


def test_finished_caption_revision_recovers_original_and_changes_pixels(conn, settings, tiny_video):
    project, asset = setup_source(conn, settings, tiny_video)
    first = creative.make_submit(conn, settings, project_id=project['id'], asset_id=asset['id'],
                                 request_key='first', style='desktop_caption', caption='Old caption')
    first = creative_worker.execute(conn, settings, fake_deps(), first)
    finished = creative.inbox_add(conn, settings, source=first['deliveries'][0]['path'],
                                 project_id=project['id'], role='output')
    second = creative.make_submit(conn, settings, project_id=project['id'], asset_id=finished['id'],
                                  request_key='correction', style='desktop_caption', caption_lines=LINES)
    assert second['asset_id'] == asset['id']
    second = creative_worker.execute(conn, settings, fake_deps(), second)
    assert second['result_ref'] != first['result_ref']
    again = creative_worker.execute(conn, settings, fake_deps(), second)
    assert again['result_ref'] == second['result_ref']
    assert len(again['deliveries']) == 1


def test_recorded_stagecraft_source_must_be_project_owned(conn, settings, tiny_video):
    project, asset = setup_source(conn, settings, tiny_video)
    metadata = {'manifest':{'postproduction':{'template':'desktop_caption','source_sha256':'0'*64}}}
    wrapped = creative.inbox_add(conn, settings, source=str(tiny_video), project_id=project['id'],
                                source_key='wrapped', metadata=metadata)
    with pytest.raises(ValueError, match='original desktop is missing'):
        creative.make_submit(conn, settings, project_id=project['id'], asset_id=wrapped['id'],
                             request_key='missing', style='desktop_caption', caption='copy')


def test_fit_tries_exact_minimum_font_size():
    from wonhee.edit import text as T
    spec = T.TextSpec('L'*43, role='mono', size_frac=.021, min_size_px=32, stroke_em=0)
    assert T.render_block(spec, (860, 96), 1600).getbbox()


def test_empty_tighten_object_cannot_trim_desktop(conn, settings, tiny_video):
    project, asset = setup_source(conn, settings, tiny_video)
    with pytest.raises(ValueError, match='preserves'):
        creative.make_submit(conn, settings, project_id=project['id'], asset_id=asset['id'],
                             request_key='trim', style='desktop_caption', caption='copy', params={'tighten':{}})


def test_reimported_stagecraft_finish_recovers_original(conn, settings, tiny_video):
    import hashlib
    from wonhee import stagecraft
    project, asset = setup_source(conn, settings, tiny_video)
    path = asset['source']
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    delivered = stagecraft.stagecraft_complete(conn, settings, job_id='c'*32, project_id=project['id'],
        source=path, request_key='stagecraft-original', title='Desktop', caption='Old caption',
        manifest={'capture_status':'mechanical_checks_passed', 'capture':{'sha256':digest}})
    reimported = creative.inbox_add(conn, settings, source=delivered['path'], project_id=project['id'], role='raw')
    assert not reimported['metadata']
    revised = creative.make_submit(conn, settings, project_id=project['id'], asset_id=reimported['id'],
                                   request_key='reimported-correction', style='desktop_caption', caption_lines=LINES)
    assert creative.media_info(conn, settings, asset_id=revised['asset_id'])['ref'] == asset['ref']


def test_stagecraft_caption_fit_checked_before_build_queue(conn, settings, monkeypatch):
    from wonhee import stagecraft
    project = creative.project_create(conn, settings, name='preflight')
    calls = []
    monkeypatch.setattr(stagecraft, '_call', lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match='shorten'):
        stagecraft.stagecraft_build(conn, settings, project_id=project['id'], request_key='too-long',
                                   brief='desktop', title='Desktop', caption='huge ' * 190)
    assert not calls
