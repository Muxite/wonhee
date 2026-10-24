"""Trusted transport metadata supplies provenance without relying on model-authored arguments."""
import asyncio
from types import SimpleNamespace

from mcp.server.fastmcp import Context
from mcp.types import RequestParams

from wonhee import creative


def _context():
    return {'version': 1, 'project_id': 1, 'request_id': 'transport-request-17', 'assets': [],
            'route': {'channel': 'telegram', 'chat_id': 'owner-chat', 'owner_id': 'owner',
                      'message_id': 17, 'message_thread_id': 4, 'route_key': 'telegram:owner-chat:4:owner'}}


def test_memory_schema_guides_model_to_supported_kinds(settings):
    from wonhee.mcp_server import mcp
    tools = {t.name:t for t in asyncio.run(mcp.list_tools())}
    schema = tools['memory_remember'].inputSchema
    assert set(schema['properties']['kind']['enum']) == {
        'preference','constraint','decision','feedback','question','idea','brief','transcript','output'}
    assert 'ctx' not in schema['properties']


def test_mcp_memory_source_is_bound_to_actual_transport(conn, settings):
    from wonhee.mcp_server import mcp
    project = creative.project_create(conn, settings, name='Source test')
    context = _context()
    context['project_id'] = project['id']
    meta = RequestParams.Meta.model_validate({'wonhee': context})
    ctx = Context(request_context=SimpleNamespace(meta=meta))
    fn = mcp._tool_manager.get_tool('memory_remember').fn
    result = fn(content='Keep originals for quality', kind='feedback', ctx=ctx,
                source={'request_id':'model-forged','asset_id':1})
    assert result['project_id'] == project['id']
    assert result['source']['request_id'] == context['request_id']
    assert result['source']['message_id'] == 17
    assert result['source']['asset_id'] == 1
    assert result['source']['route_key'] == context['route']['route_key']


def test_mcp_explicit_creator_scope_remains_available_with_transport_source(conn, settings):
    from wonhee.mcp_server import mcp
    creative.project_create(conn, settings, name='Source test')
    ctx = Context(request_context=SimpleNamespace(meta=RequestParams.Meta.model_validate({'wonhee':_context()})))
    result = mcp._tool_manager.get_tool('memory_remember').fn(
        content='Use clear subtitles everywhere', project_id=None, scope='creator', ctx=ctx)
    assert result['project_id'] is None
    assert result['source']['request_id'] == 'transport-request-17'


def test_real_stdio_metadata_is_injected_without_a_model_argument(settings):
    """Exercise the actual MCP protocol metadata and FastMCP Context injection boundary."""
    import json
    import os
    import sys
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def run():
        params = StdioServerParameters(command=sys.executable, args=['-m','wonhee.mcp_server'],
            env={**os.environ, 'CUDA_VISIBLE_DEVICES':'', 'WONHEE_HOME':str(settings.home)})
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await session.call_tool('creative_project_create', {'name':'Source test'})
                reply = await session.call_tool('memory_remember', {'content':'Keep this original for quality'},
                                                 meta={'wonhee':_context()})
                assert not reply.isError
                result = json.loads(next(c.text for c in reply.content if c.type=='text'))
                assert result['source']['request_id']=='transport-request-17'
                assert result['project_id']==1
    asyncio.run(run())


def test_mcp_submission_uses_transport_route_and_request_key_over_model_arguments(conn, settings, tiny_video):
    """A creation's destination and idempotency identity come from the trusted origin, not model text."""
    from wonhee.mcp_server import mcp

    project = creative.project_create(conn, settings, name='Trusted submission')
    asset = creative.inbox_add(conn, settings, source=str(tiny_video), project_id=project['id'])
    origin = _context()
    origin['project_id'] = project['id']
    ctx = Context(request_context=SimpleNamespace(meta=RequestParams.Meta.model_validate({'wonhee': origin})))
    fn = mcp._tool_manager.get_tool('make_submit').fn
    arguments = {'project_id': project['id'], 'asset_id': asset['id'], 'request_key': 'model-chosen-request',
                 'route': {'channel': 'telegram', 'chat_id': 'model-chosen-recipient'}, 'brief': 'Keep the take clear'}

    result = fn(**arguments, ctx=ctx)
    assert isinstance(result, dict), result
    assert result['request_key'] == origin['request_id']
    assert result['route'] == origin['route']
    brief = creative.project_show(conn, settings, project_id=project['id'])['brief']
    assert brief['source']['request_key'] == origin['request_id']
    assert brief['source']['chat_id'] == 'owner-chat'

    arguments['request_key'] = 'another-model-chosen-request'
    arguments['route'] = {'channel': 'telegram', 'chat_id': 'another-recipient'}
    replay = fn(**arguments, ctx=ctx)
    assert replay['id'] == result['id']
    assert replay['route'] == origin['route']
    assert conn.execute('SELECT count(*) FROM creative_submission').fetchone()[0] == 1


def test_mcp_forget_preserves_original_source_and_records_actual_forgetting_message(conn, settings):
    """Forgetting keeps the fact's creation provenance and attributes the forgetting event independently."""
    from wonhee.mcp_server import mcp

    project = creative.project_create(conn, settings, name='Audited memory')
    old_source = {'channel': 'telegram', 'chat_id': 'owner-chat', 'message_id': 7, 'request_id': 'saved-at-seven'}
    memory = creative.memory_remember(conn, settings, content='An outdated caption preference',
                                     project_id=project['id'], source=old_source)
    origin = _context()
    origin['project_id'] = project['id']
    ctx = Context(request_context=SimpleNamespace(meta=RequestParams.Meta.model_validate({'wonhee': origin})))
    fn = mcp._tool_manager.get_tool('memory_forget').fn
    result = fn(memory_id=memory['id'], source={'request_id': 'forged-forget', 'message_id': 999}, ctx=ctx)

    assert result['status'] == 'forgotten'
    assert {key: result['source'][key] for key in old_source} == old_source
    forgotten_by = result['source']['forgotten_by']
    assert forgotten_by['request_id'] == origin['request_id']
    assert forgotten_by['message_id'] == 17
    assert forgotten_by['origin_project_id'] == project['id']
    assert creative.memory_query(conn, settings, project_id=project['id'], operation='list')['results'] == []
    export = creative.memory_export(conn, settings)
    import json
    from pathlib import Path
    record = next(row for row in json.loads(Path(export['path']).read_text())['memories'] if row['id'] == memory['id'])
    assert record['source'] == result['source']
