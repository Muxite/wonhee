"""Direct Moeka design §Inspecting a blank historical capture; no filesystem escape."""
import asyncio
import pytest
from wonhee import creative,stagecraft


def test_source_review_binds_active_project_and_store_without_host_image_paths(conn,settings,monkeypatch):
    from wonhee.mcp_server import mcp
    from tests.test_direct_stagecraft import context
    project=creative.project_create(conn,settings,name='Source review')
    calls=[]
    def fake(command,data=None,**kw):
        calls.append((command,data))
        if command=='show':return {'project_id':project['id'],'state':'done','phase':'staged'}
        assert command=='review-owned'
        return {'sources':[{'path':'app/src/Scene.tsx','excerpt':'code'}],'images':[{'path':'/host/path.png'}]}
    monkeypatch.setattr(stagecraft,'_call',fake)
    tools={t.name:t for t in asyncio.run(mcp.list_tools())}
    assert set(tools['stagecraft_review'].inputSchema['properties'])=={'job_id','source_file'}
    result=mcp._tool_manager.get_tool('stagecraft_review').fn(job_id='a'*32,ctx=context(project['id']))
    assert result['sources'] and 'images' not in result
    assert calls[-1][1]['project_id']==project['id']
    assert calls[-1][1]['wonhee_home']==str(settings.home.resolve())
    with pytest.raises(ValueError,match='owning'):
        stagecraft.stagecraft_review(conn,settings,job_id='a'*32,project_id=project['id']+1)


def test_review_summary_precedes_compact_source_and_selected_file_is_bounded(conn,settings,monkeypatch):
    import json
    project=creative.project_create(conn,settings,name='Fits the tool budget')
    def fake(command,data=None,**kw):
        if command=='show':return {'project_id':project['id'],'state':'done','resolved_settings':{'width':1600,'height':1000}}
        return {'initial_state':{'benchmark_present':False},'rendered_checks':{'mesh_measured':False},
                'sources':[{'path':'app/src/'+name,'excerpt':'x'*5000,'truncated':True,'sha256':'0'*64}
                           for name in ('Scene.tsx','scene.css','MeshProfiler.tsx','Graph.tsx')]}
    monkeypatch.setattr(stagecraft,'_call',fake)
    result=stagecraft.stagecraft_review(conn,settings,job_id='a'*32,project_id=project['id'])
    assert list(result).index('initial_state') < list(result).index('sources')
    assert len(json.dumps(result,indent=2))<10000
    assert all(len(f['excerpt'])<=1300 for f in result['sources'])
    detail=stagecraft.stagecraft_review(conn,settings,job_id='a'*32,project_id=project['id'],source_file='MeshProfiler.tsx')
    assert len(detail['sources'])==1 and len(detail['sources'][0]['excerpt'])==5000
    with pytest.raises(ValueError,match='listed'):
        stagecraft.stagecraft_review(conn,settings,job_id='a'*32,project_id=project['id'],source_file='../secret')
