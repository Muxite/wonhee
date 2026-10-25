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
    assert {'brief','title','duration_seconds','surface','orientation','primitive','reference_asset_ids'}==set(props)
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
