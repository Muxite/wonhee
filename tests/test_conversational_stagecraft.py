"""Supported conversational flow needs neither a benchmark script nor a JSON import."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from wonhee import creative, stagecraft


def test_dataset_tool_binds_host_identity_and_completes_once(conn,settings,monkeypatch,tmp_path):
    from wonhee.mcp_server import mcp
    from tests.test_direct_stagecraft import context
    project=creative.project_create(conn,settings,name='No-code dataset')
    home=tmp_path/'stagecraft'
    monkeypatch.setenv('STAGECRAFT_HOME',str(home))
    fn=mcp._tool_manager.get_tool('stagecraft_dataset').fn
    queued=fn(title='Random paragraph echoes',paragraph_count=16,trials=2,ctx=context(project['id']))
    assert not getattr(queued,'isError',False),queued
    assert queued['operation']=='dataset' and queued['state']=='queued'
    duplicate=fn(title='Random paragraph echoes',paragraph_count=16,trials=2,ctx=context(project['id']))
    assert queued['id']==duplicate['id']
    repo=Path(stagecraft.__file__).resolve().parents[2]/'stagecraft'
    subprocess.run([str(repo/'.venv/bin/stagecraft'),'worker','--once'],cwd=repo,
                   env={**os.environ,'STAGECRAFT_HOME':str(home)},capture_output=True,text=True,check=True,timeout=60)
    result=stagecraft.stagecraft_show(conn,settings,job_id=queued['id'],project_id=project['id'])
    assert result['state']=='done',result.get('failure')
    assert result['dataset']['summary']['stdio']['n']==32
    assert result['dataset']['summary']['http']['n']==32
    assets=result['dataset']['assets']
    assert assets['plot-data.json'] and assets['dataset.zip']
    asset=creative.media_info(conn,settings,asset_id=assets['plot-data.json'])
    assert asset['project_id']==project['id']
    assert result['delivery']['mime']=='application/zip'
    assert 'finishing' not in result
    assert conn.execute('select count(*) from creative_delivery').fetchone()[0]==1
    with pytest.raises(ValueError,match='another creative project'):
        stagecraft.stagecraft_show(conn,settings,job_id=queued['id'],project_id=project['id']+1)
    # Video and dataset commands do not collide within the same trusted turn.
    video=stagecraft.stagecraft_build(conn,settings,project_id=project['id'],request_key='direct-request',
        title='Desktop',brief='Use the measured profiler',primitive='scene',benchmark='off',caption='')
    assert video['id']!=queued['id']


def test_build_creates_data_automatically_and_explicit_data_wins(conn,settings,monkeypatch,tmp_path):
    from tests.test_profiler_data import data_fixture
    project=creative.project_create(conn,settings,name='One call')
    submitted=[]
    monkeypatch.setattr(stagecraft,'_call',lambda command,data=None,**kw:submitted.append(data) or {'id':'a'*32})
    args=dict(project_id=project['id'],request_key='r',title='Mesh',brief='Graph and measured profiler')
    stagecraft.stagecraft_build(conn,settings,**args)
    assert submitted[-1]['benchmark']=={'paragraph_count':512,'trials':5,'seed':24026}
    assert submitted[-1]['spec']['graph_nodes']==3000
    assert submitted[-1]['spec']['caption_above_dashboard'] is False
    path=tmp_path/'data.json';path.write_text(json.dumps(data_fixture()))
    asset=creative.inbox_add(conn,settings,source=str(path),project_id=project['id'])
    stagecraft.stagecraft_build(conn,settings,**args,profiler_asset_id=asset['id'])
    assert submitted[-1]['benchmark'] is None
    stagecraft.stagecraft_build(conn,settings,**args,benchmark='off')
    assert submitted[-1]['benchmark'] is None
    with pytest.raises(ValueError,match='choose a supplied'):
        stagecraft.stagecraft_build(conn,settings,**args,profiler_asset_id=asset['id'],benchmark='refresh')


def test_failure_review_is_project_scoped_and_imports_hash_checked_preview(conn,settings,tmp_path,monkeypatch):
    from PIL import Image
    project=creative.project_create(conn,settings,name='Review')
    image=tmp_path/'failure.png';Image.new('RGB',(100,60),'cyan').save(image)
    review={'diagnostics':[{'kind':'error.txt','text':'profiler panel overlaps mesh'}],
            'images':[{'path':str(image),'sha256':hashlib.sha256(image.read_bytes()).hexdigest()}]}
    calls=[]
    def call(command,**kw):
        calls.append(command)
        return {'id':'a'*32,'project_id':project['id'],'failure':{'stage':'rendered_layout','recovery':'revise_app'}} if command=='show' else review
    monkeypatch.setattr(stagecraft,'_call',call)
    with pytest.raises(ValueError,match='another creative project'):
        stagecraft.stagecraft_show(conn,settings,job_id='a'*32,project_id=project['id']+1)
    assert calls==['show']
    result=stagecraft.stagecraft_show(conn,settings,job_id='a'*32,project_id=project['id'])
    asset=creative.media_info(conn,settings,asset_id=result['review']['image_asset_ids'][0])
    assert asset['project_id']==project['id'] and asset['mime']=='image/png'
    assert 'profiler panel' in result['review']['diagnostics'][0]['text']


def test_default_project_lookup_uses_host_context_and_unknown_dataset_has_actionable_error(conn,settings):
    from wonhee.mcp_server import mcp
    from tests.test_direct_stagecraft import context
    project=creative.project_create(conn,settings,name='Current project')
    result=mcp._tool_manager.get_tool('creative_project_show').fn(ctx=context(project['id']))
    assert result['id']==project['id']
    result=mcp._tool_manager.get_tool('stagecraft_build').fn(title='Fresh data',brief='Build profiler',
        profiler_asset_id=999999,ctx=context(project['id']))
    assert result.isError
    assert 'omit profiler_asset_id' in result.structuredContent['error']['message']
    assert conn.execute('SELECT count(*) FROM creative_brief').fetchone()[0]==0
