"""Direct Moeka contract: ordinary typed inputs, permanent errors and reference pixels."""
import asyncio
from types import SimpleNamespace

import pytest
from mcp.server.fastmcp import Context
from mcp.types import RequestParams
from PIL import Image

from wonhee import creative,stagecraft


def context(project_id):
    origin={'version':1,'project_id':project_id,'request_id':'direct-request','route':{
        'channel':'telegram','chat_id':'creator','owner_id':'creator','route_key':'telegram:creator'}}
    return Context(request_context=SimpleNamespace(meta=RequestParams.Meta.model_validate({'wonhee':origin})))


def test_build_tool_has_plain_fields_and_host_only_identity(settings):
    from wonhee.mcp_server import mcp
    tools={t.name:t for t in asyncio.run(mcp.list_tools())}
    props=tools['stagecraft_build'].inputSchema['properties']
    assert {'brief','title','duration_seconds','surface','orientation','primitive','presentation','reference_asset_ids','width','height','dashboard_aspect_ratio',
            'caption_above_dashboard','forbidden_text','base_job_id','caption','prepare_only','profiler_asset_id','benchmark'}==set(props)
    assert props['brief']['type']=='string'
    assert tools['stagecraft_build'].inputSchema['required']==['brief','title']


def test_invalid_build_is_permanent_mcp_error(conn,settings):
    from wonhee.mcp_server import mcp
    project=creative.project_create(conn,settings,name='Direct')
    result=mcp._tool_manager.get_tool('stagecraft_build').fn(brief='',title='Missing brief',ctx=context(project['id']))
    assert result.isError
    assert result.structuredContent['error']['retryable'] is False
    assert result.structuredContent['error']['code']=='invalid_request'


def test_build_resolves_refs_and_preserves_brief(conn,settings,tmp_path,monkeypatch):
    project=creative.project_create(conn,settings,name='Direct')
    image=tmp_path/'reference.png';Image.new('RGB',(64,32),'cyan').save(image)
    asset=creative.inbox_add(conn,settings,source=str(image),project_id=project['id'])
    calls=[]
    monkeypatch.setattr(stagecraft,'_call',lambda command,data=None,**kwargs:calls.append((command,data)) or {'id':'a'*32})
    from wonhee.mcp_server import mcp
    fn=mcp._tool_manager.get_tool('stagecraft_build').fn
    result=fn(brief='Dense animated communication dashboard',title='Mesh',reference_asset_ids=[asset['id']],ctx=context(project['id']))
    assert result['id']=='a'*32
    data=calls[0][1]
    assert data['request_key']=='direct-request' and data['project_id']==project['id']
    assert data['references'][0]['sha256']==asset['ref']
    assert data['spec']['coding']['primitive']=='stdio_mesh'
    fn(brief='Dense animated communication dashboard',title='Mesh',reference_asset_ids=[asset['id']],ctx=context(project['id']))
    assert conn.execute('SELECT count(*) FROM creative_brief').fetchone()[0]==1
    assert creative.project_show(conn,settings,project_id=project['id'])['brief']['content'].startswith('Dense')
    other=creative.project_create(conn,settings,name='Other')
    with pytest.raises(ValueError,match='another creative project'):
        stagecraft.stagecraft_build(conn,settings,brief='Use image',title='Wrong',project_id=other['id'],request_key='x',reference_asset_ids=[asset['id']])


def test_still_image_returns_viewable_pixels(conn,settings,tmp_path):
    project=creative.project_create(conn,settings,name='Images')
    image=tmp_path/'reference.png';Image.new('RGB',(64,32),'cyan').save(image)
    asset=creative.inbox_add(conn,settings,source=str(image),project_id=project['id'])
    from wonhee.mcp_server import media_info
    result=media_info(asset['id'],analyze=True)
    assert any(item.type=='image' and item.mimeType=='image/jpeg' for item in result.content)


def test_missing_reference_is_permanent_mcp_error(settings):
    from wonhee.mcp_server import media_info
    result = media_info(999999, analyze=True)
    assert result.isError
    assert result.structuredContent['error']['retryable'] is False


def test_revision_inherits_and_explicit_dimensions_override_orientation(conn,settings,monkeypatch):
    project=creative.project_create(conn,settings,name='Revise')
    calls=[]
    parent={'settings':{'premise':'old','title':'old','duration_ms':20000,'width':1000,'height':1600,
                        'surface':'browser','coding':{'task':'old','primitive':'stdio_mesh'},
                        'dashboard_aspect_ratio':'16:10','caption_above_dashboard':True,
                        'forbidden_text':['one','two','three'], 'actions':[{'id':'late','at_ms':15000,'patch':{'stage':'complete'}}]},'references':[]}
    def call(command,data=None,**kwargs):
        calls.append((command,data))
        return parent if command=='base-input' else {'id':'b'*32}
    monkeypatch.setattr(stagecraft,'_call',call)
    stagecraft.stagecraft_build(conn,settings,brief='Improve previous app',title='Revision',
                               project_id=project['id'],request_key='r',base_job_id='a'*32)
    spec=calls[-1][1]['spec']
    assert spec['width']==1280 and spec['height']==800 and spec['duration_ms']==20000
    assert spec['forbidden_text']==['one','two','three'] and not spec['caption_above_dashboard']
    stagecraft.stagecraft_build(conn,settings,brief='Shorten previous app',title='Short',
        project_id=project['id'],request_key='short',base_job_id='a'*32,duration_seconds=5)
    assert calls[-1][1]['spec']['actions'][0]['at_ms']==3750
    stagecraft.stagecraft_build(conn,settings,brief='Exact canvas',title='Revision',project_id=project['id'],
        request_key='r2',width=1000,height=1600,orientation='landscape')
    assert calls[-1][1]['spec']['width']==1000 and calls[-1][1]['spec']['height']==1600
    with pytest.raises(ValueError,match='together'):
        stagecraft.stagecraft_build(conn,settings,brief='Exact',title='Invalid',project_id=project['id'],request_key='r3',width=1000)
    from wonhee.mcp_server import mcp
    tools={t.name:t for t in asyncio.run(mcp.list_tools())}
    assert set(tools['stagecraft_list'].inputSchema['properties'])=={'limit'}


def test_video_references_use_cached_cpu_contact_sheet(conn,settings,tmp_path,monkeypatch):
    import hashlib
    project=creative.project_create(conn,settings,name='Video')
    sheet=tmp_path/'contact.png';Image.new('RGB',(64,64),'cyan').save(sheet)
    digest=hashlib.sha256(sheet.read_bytes()).hexdigest()
    asset={'project_id':project['id'],'path':'/video','ref':'original','mime':'video/mp4'}
    calls=[]
    def media(*args,analyze=False,**kwargs):
        calls.append(analyze)
        return {**asset,**({'contact_sheet':{'path':str(sheet),'ref':digest,'run_id':42}} if analyze else {})}
    monkeypatch.setattr(creative,'media_info',media)
    submitted=[]
    monkeypatch.setattr(stagecraft,'_call',lambda command,data=None,**kwargs:submitted.append(data) or {'id':'a'*32})
    stagecraft.stagecraft_build(conn,settings,brief='Video direction',title='Video',project_id=project['id'],
                                request_key='r',reference_asset_ids=[3])
    ref=submitted[0]['references'][0]
    assert calls==[False,True] and ref['sha256']==digest
    assert ref['source_video']=={'asset_id':3,'sha256':'original','conversion':'contact_sheet'}


def test_real_video_reference_conversion_is_cached(conn,settings,tiny_video,monkeypatch):
    project=creative.project_create(conn,settings,name='Real video reference')
    asset=creative.inbox_add(conn,settings,source=str(tiny_video),project_id=project['id'])
    submissions=[]
    monkeypatch.setattr(stagecraft,'_call',lambda command,data=None,**kwargs:submissions.append(data) or {'id':'a'*32})
    for _ in range(2):
        stagecraft.stagecraft_build(conn,settings,brief='Use this video for visual direction',title='Video',
            project_id=project['id'],request_key='cached-video',reference_asset_ids=[asset['id']])
    first,second=[item['references'][0] for item in submissions]
    assert first==second
    assert first['source_video']['sha256']==asset['ref']
    assert first['mime'].startswith('image/') and Image.open(first['path']).width>0


def test_profiler_dataset_is_project_owned_and_caption_stays_out_of_scene(conn,settings,tmp_path,monkeypatch):
    import json
    project=creative.project_create(conn,settings,name='Profiler')
    path=tmp_path/'plot-data.json'
    from tests.test_profiler_data import data_fixture
    data=data_fixture()
    path.write_text(json.dumps(data))
    asset=creative.inbox_add(conn,settings,source=str(path),project_id=project['id'])
    submitted=[]
    monkeypatch.setattr(stagecraft,'_call',lambda command,data=None,**kw:submitted.append(data) or {'id':'a'*32})
    stagecraft.stagecraft_build(conn,settings,project_id=project['id'],request_key='profiler',
        title='Desktop',brief='Graph and profiler share the desktop',profiler_asset_id=asset['id'],caption='Wonhee caption')
    request=submitted[-1]
    assert request['caption']=='Wonhee caption' and 'caption' not in request['spec']
    assert request['spec']['height']==800 and request['spec']['width']==1280
    assert request['spec']['initial']['benchmark']==data
    assert request['spec']['initial']['benchmark_source']['sha256']==asset['ref']
    other=creative.project_create(conn,settings,name='Other')
    with pytest.raises(ValueError,match='another creative project'):
        stagecraft.stagecraft_build(conn,settings,project_id=other['id'],request_key='wrong',title='Other',
                                   brief='Use dataset',profiler_asset_id=asset['id'])


def test_mesh_story_clears_old_custom_app_and_uses_tall_shared_preset(conn,settings,monkeypatch):
    from tests.test_profiler_data import data_fixture
    project=creative.project_create(conn,settings,name='Built-in story')
    calls=[]
    def fake(command,data=None,**kwargs):
        calls.append((command,data))
        if command=='base-input':
            return {'settings':{'width':1600,'height':1000,'duration_ms':20000,'surface':'browser',
                    'component':'old custom app','css':'old styles','dashboard_aspect_ratio':'16:10',
                    'initial':{'benchmark':data_fixture()},'actions':[{'at_ms':0,'id':'old','patch':{'presentation':'standard'}}]},
                    'references':[]}
        return {'id':'a'*32}
    monkeypatch.setattr(stagecraft,'_call',fake)
    stagecraft.stagecraft_build(conn,settings,project_id=project['id'],request_key='story',brief='Escalate the mesh',
        title='Story',presentation='mesh_story',base_job_id='b'*32,caption='',benchmark='off')
    spec=calls[-1][1]['spec']
    assert (spec['width'],spec['height'])==(1080,1440)
    assert spec['dashboard_aspect_ratio']=='1080:1440'
    assert spec['coding'] is None and spec['component'] is None and spec['css'] is None
    assert not spec['actions'] and spec['graph_nodes']==0
    with pytest.raises(ValueError,match='measured profiler'):
        stagecraft.stagecraft_build(conn,settings,project_id=project['id'],request_key='empty',brief='Mesh',
            title='Empty',presentation='mesh_story',benchmark='off',caption='')


def test_prepare_only_suppresses_default_finishing_and_never_accepts_caption(conn,settings,monkeypatch):
    project=creative.project_create(conn,settings,name='Prepare then assemble')
    calls=[]
    monkeypatch.setattr(stagecraft,'_call',lambda command,data=None,**kw:calls.append(data) or {'id':'a'*32})
    stagecraft.stagecraft_build(conn,settings,project_id=project['id'],request_key='prepare',title='Shared',
        brief='Shared source for two captions',presentation='mesh_story',prepare_only=True)
    assert calls[-1]['prepare_only'] is True and calls[-1]['caption'] is None
    assert calls[-1]['spec']['coding'] is None and calls[-1]['benchmark']
    with pytest.raises(ValueError,match='without a finishing caption'):
        stagecraft.stagecraft_build(conn,settings,project_id=project['id'],request_key='bad',title='Shared',
            brief='Source',prepare_only=True,caption='Must not silently ignore this')
