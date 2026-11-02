"""Design: stagecraft § trusted context, completion idempotency and project isolation."""
import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest
from mcp.server.fastmcp import Context
from mcp.types import RequestParams

from wonhee import creative, stagecraft


def test_stagecraft_mcp_binds_identity_and_keeps_completion_host_only(conn,settings,monkeypatch):
    from wonhee.mcp_server import mcp
    project=creative.project_create(conn,settings,name='Stagecraft request')
    origin={'version':1,'project_id':project['id'],'request_id':'trusted-17','route':{
        'channel':'telegram','chat_id':'owner','owner_id':'owner','route_key':'telegram:owner',
        'message_id':17,'message_thread_id':None}}
    ctx=Context(request_context=SimpleNamespace(meta=RequestParams.Meta.model_validate({'wonhee':origin})))
    calls=[]
    def call(command,data=None,job_id=None):
        calls.append((command,data))
        return {'id':'a'*32,'project_id':project['id']}
    monkeypatch.setattr(stagecraft,'_call',call)
    result=mcp._tool_manager.get_tool('stagecraft_submit').fn(spec={'title':'Tiny'},project_id=999,
          request_key='forged',route={'chat_id':'forged'},ctx=ctx)
    assert result['project_id']==project['id']
    assert calls[0][1]['request_key']=='trusted-17'
    assert calls[0][1]['route']==origin['route']
    assert calls[0][1]['project_id']==project['id']
    names={t.name for t in asyncio.run(mcp.list_tools())}
    assert {'stagecraft_schema','stagecraft_submit','stagecraft_show','stagecraft_retry'} <= names
    assert 'stagecraft_complete' not in names
    # Schema calls must not receive an extra project_id injected from metadata.
    assert isinstance(mcp._tool_manager.get_tool('stagecraft_schema').fn(ctx=ctx),dict)


def test_stagecraft_completion_stages_once_and_refuses_conflicts(conn,settings,tiny_video):
    project=creative.project_create(conn,settings,name='Stagecraft output')
    digest=hashlib.sha256(tiny_video.read_bytes()).hexdigest()
    args={'job_id':'a'*32,'project_id':project['id'],'source':str(tiny_video),'request_key':'request-17',
          'title':'One JSON key','route':{},'manifest':{'capture_status':'mechanical_checks_passed',
          'mode':'simulated_story','capture':{'sha256':digest}}}
    first=stagecraft.stagecraft_complete(conn,settings,**args)
    second=stagecraft.stagecraft_complete(conn,settings,**args)
    assert first['id']==second['id']
    assert first['status']=='queued' and first['as_document']
    assert conn.execute('SELECT count(*) FROM creative_delivery').fetchone()[0]==1
    with pytest.raises(ValueError,match='conflicts'):
        stagecraft.stagecraft_complete(conn,settings,**{**args,'route':{'chat_id':'changed'}})


def test_job_show_and_retry_are_project_scoped(conn,settings,monkeypatch):
    monkeypatch.setattr(stagecraft,'_call',lambda *a,**kw:{'project_id':2})
    with pytest.raises(ValueError,match='another creative project'):
        stagecraft.stagecraft_show(conn,settings,job_id='a'*32,project_id=1)
    with pytest.raises(ValueError):
        stagecraft.stagecraft_retry(conn,settings,job_id='a'*32,project_id=1)


def test_completion_caption_is_wonhee_owned_cpu_cached_and_idempotent(conn,settings,tiny_video):
    from pathlib import Path
    from wonhee import media
    project=creative.project_create(conn,settings,name='Desktop finishing')
    digest=hashlib.sha256(tiny_video.read_bytes()).hexdigest()
    args={'job_id':'b'*32,'project_id':project['id'],'source':str(tiny_video),'request_key':'caption-1',
          'title':'Desktop','caption':'Pipes, dressed as infrastructure.','route':{},
          'manifest':{'capture_status':'mechanical_checks_passed','capture':{'sha256':digest}}}
    first=stagecraft.stagecraft_complete(conn,settings,**args)
    assert first['ref']!=digest
    probe=media.probe(Path(first['path']))
    assert (probe.width,probe.height)==(1000,1600)
    count=conn.execute('SELECT count(*) FROM run').fetchone()[0]
    again=stagecraft.stagecraft_complete(conn,settings,**args)
    assert first['id']==again['id']
    assert conn.execute('SELECT count(*) FROM run').fetchone()[0]==count
    assert conn.execute('SELECT count(*) FROM creative_delivery').fetchone()[0]==1
    with pytest.raises(ValueError,match='conflicts'):
        stagecraft.stagecraft_complete(conn,settings,**{**args,'caption':'Changed caption'})
    receipt=json.loads(conn.execute('SELECT manifest_json FROM stagecraft_delivery WHERE job_id=?',('b'*32,)).fetchone()[0])
    assert receipt['postproduction']['source_sha256']==digest
    assert receipt['postproduction']['output_sha256']==first['ref']


def test_failed_caption_can_recover_without_staging_raw_output(conn,settings,tiny_video,monkeypatch):
    from wonhee import service
    project=creative.project_create(conn,settings,name='Recover finishing')
    digest=hashlib.sha256(tiny_video.read_bytes()).hexdigest()
    args={'job_id':'c'*32,'project_id':project['id'],'source':str(tiny_video),'request_key':'caption-recovery',
          'title':'Desktop','caption':'Two transports. One paragraph.','route':{},
          'manifest':{'capture_status':'mechanical_checks_passed','capture':{'sha256':digest}}}
    actual=service.node_run
    def fail(*args,**kwargs):
        raise RuntimeError('interrupted finishing')
    monkeypatch.setattr(service,'node_run',fail)
    with pytest.raises(RuntimeError,match='interrupted finishing'):
        stagecraft.stagecraft_complete(conn,settings,**args)
    assert conn.execute('SELECT count(*) FROM creative_delivery').fetchone()[0]==0
    assert conn.execute('SELECT count(*) FROM stagecraft_delivery').fetchone()[0]==0
    monkeypatch.setattr(service,'node_run',actual)
    result=stagecraft.stagecraft_complete(conn,settings,**args)
    assert result['status']=='queued'
    assert conn.execute('SELECT count(*) FROM creative_delivery').fetchone()[0]==1
    assert conn.execute("SELECT count(*) FROM creative_intake WHERE source_key=?",('stagecraft:'+'c'*32+':desktop',)).fetchone()[0]==1


def test_prepare_only_completion_imports_raw_once_without_outbox(conn,settings,tiny_video,monkeypatch):
    project=creative.project_create(conn,settings,name='Shared source only')
    digest=hashlib.sha256(tiny_video.read_bytes()).hexdigest()
    args={'job_id':'e'*32,'project_id':project['id'],'source':str(tiny_video),'request_key':'prepare-source',
          'title':'Shared desktop','prepare_only':True,'manifest':{'capture_status':'mechanical_checks_passed',
          'capture':{'sha256':digest}}}
    first=stagecraft.stagecraft_complete(conn,settings,**args)
    second=stagecraft.stagecraft_complete(conn,settings,**args)
    assert first==second and first['prepared']
    assert conn.execute('select count(*) from creative_delivery').fetchone()[0]==0
    assert conn.execute('select count(*) from stagecraft_delivery').fetchone()[0]==0
    asset=creative.media_info(conn,settings,asset_id=first['assets']['desktop_asset_id'])
    assert asset['role']=='raw' and asset['ref']==digest
    monkeypatch.setattr(stagecraft,'_call',lambda *a,**k:{'project_id':project['id'],'prepared':first,
        'resolved_settings':{'presentation':'mesh_story'},'phase':'prepared'})
    shown=stagecraft.stagecraft_show(conn,settings,job_id='e'*32,project_id=project['id'])
    assert shown['assets']==first['assets'] and 'delivery' not in shown
    assert shown['finishing']['style']=='desktop_reel'


def test_prepared_source_rejects_changed_bytes_and_request(conn,settings,tiny_video,tmp_path):
    import shutil
    path=tmp_path/'shared.mp4';shutil.copyfile(tiny_video,path)
    project=creative.project_create(conn,settings,name='Immutable preparation')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    args={'job_id':'f'*32,'project_id':project['id'],'source':str(path),'request_key':'original',
          'title':'Source','prepare_only':True,'caption':'','manifest':{'capture_status':'mechanical_checks_passed',
          'capture':{'sha256':digest}}}
    first=stagecraft.stagecraft_complete(conn,settings,**args)
    with pytest.raises(ValueError,match='original capture/request'):
        stagecraft.stagecraft_complete(conn,settings,**{**args,'request_key':'changed'})
    path.write_bytes(path.read_bytes()+b'changed trailer')
    changed=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='original capture/request'):
        stagecraft.stagecraft_complete(conn,settings,**{**args,'manifest':{'capture_status':'mechanical_checks_passed',
            'capture':{'sha256':changed}}})
    assert creative.media_info(conn,settings,asset_id=first['assets']['desktop_asset_id'])['ref']==digest
    assert conn.execute('select count(*) from creative_delivery').fetchone()[0]==0
