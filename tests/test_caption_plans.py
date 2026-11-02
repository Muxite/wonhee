"""Direct Moeka design §Two presentations of one mesh story."""
import asyncio
import hashlib

import pytest

from wonhee import creative
from wonhee.caption_plan import CaptionCue, render_layout, validate_cues


@pytest.fixture(autouse=True)
def no_vectors(monkeypatch):
    monkeypatch.setattr(creative, '_docstore', lambda *a, **k: None)


def cue(start, end, text='Architecture reveal'):
    return {'start_seconds':start,'end_seconds':end,'text':text}


def test_timed_captions_are_contiguous_source_time_above_complete_source():
    plan=render_layout({},[cue(0,4),cue(4,8),cue(8,12),cue(12,16),cue(16,20)],20000)
    assert (plan['canvas']['w'],plan['canvas']['h'])==(1080,1920)
    window=plan['layers'][0]
    assert window['clip_fit']=='contain'
    text=[l for l in plan['layers'] if l['type']=='text']
    assert [(l['t']['in_ms'],l['t']['out_ms']) for l in text]==[(i*4000,(i+1)*4000) for i in range(5)]
    assert all(l['box']['y']+l['box']['h'] < window['box']['y'] for l in text)
    assert all(l['anim']=='none' and l['min_size_px']==40 for l in text)


@pytest.mark.parametrize('plan',[
    [], [cue(1,20)], [cue(0,4),cue(5,20)], [cue(0,8),cue(4,20)], [cue(0,19)],
    [cue(0,21)], [cue(0,20,' ')], [cue(0,float('nan'))], [cue(0,.0004),cue(.0004,20)],
])
def test_invalid_timelines_rejected_before_render(plan):
    with pytest.raises(ValueError): validate_cues(plan,20000)


def test_reel_queue_has_typed_cues_and_uses_verified_raw_portrait(conn,settings,tiny_video):
    from wonhee.mcp_server import mcp
    from tests.test_direct_stagecraft import context
    project=creative.project_create(conn,settings,name='One source, two presentations')
    asset=creative.inbox_add(conn,settings,source=str(tiny_video),project_id=project['id'],role='raw',
        metadata={'producer':'stagecraft','manifest':{'capture_status':'mechanical_checks_passed','capture':{'sha256':hashlib.sha256(tiny_video.read_bytes()).hexdigest()}}})
    tools={t.name:t for t in asyncio.run(mcp.list_tools())}
    schema=tools['make_submit'].inputSchema
    assert 'CaptionCue' in schema['$defs']
    fn=mcp._tool_manager.get_tool('make_submit').fn
    job=fn(project_id=project['id'],asset_id=asset['id'],request_key='ignored-host-bound',style='desktop_reel',
        caption_cues=[cue(0,1),cue(1,2),cue(2,3)],ctx=context(project['id']))
    assert job['status']=='queued' and job['phase']=='render'
    assert len(job['effective_caption_cues'])==3
    assert job['asset_id']==asset['id']
    again=fn(project_id=project['id'],asset_id=asset['id'],request_key='ignored-host-bound',style='desktop_reel',
        caption_cues=[cue(0,1),cue(1,2),cue(2,3)],ctx=context(project['id']))
    assert again['id']==job['id']
    with pytest.raises(ValueError,match='cannot mix'):
        creative.make_submit(conn,settings,project_id=project['id'],asset_id=asset['id'],request_key='bad',
            style='desktop_reel',caption='conflicting',caption_cues=[cue(0,3)])
    with pytest.raises(ValueError,match='exactly'):
        creative.make_submit(conn,settings,project_id=project['id'],asset_id=asset['id'],request_key='duration',
            style='desktop_reel',caption_cues=[cue(0,20)])


def test_existing_request_options_normalized_for_new_default(conn,settings,tiny_video):
    import json
    project=creative.project_create(conn,settings,name='Legacy idempotency')
    asset=creative.inbox_add(conn,settings,source=str(tiny_video),project_id=project['id'])
    job=creative.make_submit(conn,settings,project_id=project['id'],asset_id=asset['id'],request_key='r'*450,style='legend')
    old=dict(job['params']);old.pop('caption_cues')
    conn.execute('update creative_submission set params_json=? where id=?',(json.dumps(old),job['id']))
    assert creative.make_submit(conn,settings,project_id=project['id'],asset_id=asset['id'],
        request_key='r'*450,style='legend')['id']==job['id']


@pytest.mark.parametrize('capture', [{}, {'sha256':'0'*64}])
def test_unknown_portrait_cannot_borrow_verified_metadata(conn,settings,tiny_video,capture):
    project=creative.project_create(conn,settings,name='unbound source')
    asset=creative.inbox_add(conn,settings,source=str(tiny_video),project_id=project['id'],role='raw',
        metadata={'producer':'stagecraft','manifest':{'capture_status':'mechanical_checks_passed','capture':capture}})
    with pytest.raises(ValueError,match='portrait'):
        creative.make_submit(conn,settings,project_id=project['id'],asset_id=asset['id'],request_key='bad',
            style='desktop_reel',caption='A caption')


def test_cues_validate_against_recovered_original_duration(conn,settings,tiny_video,tmp_path):
    import subprocess
    project=creative.project_create(conn,settings,name='source duration')
    original=creative.inbox_add(conn,settings,source=str(tiny_video),project_id=project['id'],role='raw',
        metadata={'producer':'stagecraft','manifest':{'capture_status':'mechanical_checks_passed',
                  'capture':{'sha256':hashlib.sha256(tiny_video.read_bytes()).hexdigest()}}})
    trimmed=tmp_path/'old-finish.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(tiny_video),'-t','1','-c:v','libx264','-threads','1',
                    '-an',str(trimmed)],check=True)
    finished=creative.inbox_add(conn,settings,source=str(trimmed),project_id=project['id'],role='output',
        metadata={'manifest':{'postproduction':{'template':'desktop_reel','source_sha256':original['ref']}}})
    replied=creative.inbox_add(conn,settings,source=str(trimmed),project_id=project['id'],role='raw',source_key='telegram-reimport')
    assert not replied['metadata']
    job=creative.make_submit(conn,settings,project_id=project['id'],asset_id=replied['id'],request_key='restore',
        style='desktop_reel',caption_cues=[cue(0,3)])
    assert job['source_asset_id']==original['id']


def test_two_variants_in_one_host_turn_have_independent_idempotent_identity(conn,settings,tiny_video):
    from wonhee.mcp_server import mcp
    from tests.test_direct_stagecraft import context
    project=creative.project_create(conn,settings,name='two in one turn')
    asset=creative.inbox_add(conn,settings,source=str(tiny_video),project_id=project['id'],role='raw',
        metadata={'producer':'stagecraft','manifest':{'capture_status':'mechanical_checks_passed',
                  'capture':{'sha256':hashlib.sha256(tiny_video.read_bytes()).hexdigest()}}})
    fn=mcp._tool_manager.get_tool('make_submit').fn
    common=dict(project_id=project['id'],asset_id=asset['id'],request_key='host-overrides',style='desktop_reel',ctx=context(project['id']))
    a=fn(**common,variant='viral-discovery',caption='A static hook')
    b=fn(**common,variant='slow-unravel',caption_cues=[cue(0,3)])
    assert a['id'] != b['id']
    assert a['variant']=='viral-discovery' and b['variant']=='slow-unravel'
    assert fn(**common,variant='viral-discovery',caption='A static hook')['id']==a['id']
    assert conn.execute('select count(*) from creative_submission').fetchone()[0]==2
    assert conn.execute('select count(*) from creative_delivery').fetchone()[0]==0


def test_caption_geometry_survives_phone_safe_zone_adjustments():
    from wonhee.edit import compose as C
    layout=C.load_template('desktop_reel',{'title':'A persistent caption above the complete desktop.'})
    ly=next(layer for layer in layout.layers if layer.type=='text')
    notes=C.Notes()
    context=C.Ctx(layout.canvas.w,layout.canvas.h,C.palette_of(layout),0,[],None,notes)
    image,box=C.layer_image(ly,1,context)
    assert not notes.adjusted
    assert box[1]>=round(layout.canvas.h*.12)
    assert box[3]<round(layout.canvas.h*layout.layers[0].box.y)
    assert image.getbbox()
