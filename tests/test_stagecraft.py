"""Design: stagecraft § trusted context, completion idempotency and project isolation."""
import asyncio
import hashlib
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
