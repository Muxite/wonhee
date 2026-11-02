"""Direct Moeka design §Readable access to measured benchmarks: read-only, bound, evidenced."""
import asyncio
import hashlib
import json

import pytest

from wonhee import creative, stagecraft
from wonhee import media
from tests.test_profiler_data import data_fixture


@pytest.fixture(autouse=True)
def no_vectors(monkeypatch):
    monkeypatch.setattr(creative, '_docstore', lambda *a, **k: None)


def dataset(conn, settings, tmp_path, project_id, complete=True):
    directory = tmp_path / str(project_id)
    directory.mkdir(exist_ok=True)
    data = data_fixture()
    manifest = {'files':{},'summary':data['summary'],'paragraphs':2,'trials':1,'rows':4,'seed':data['seed'],
                'http':'persistent loopback HTTP/1.1, no TLS','stdio':'NDJSON child pipes'}
    assets = {}
    for name in ('plot-data.json','paragraph_echo.py','paragraphs.csv','paragraphs.jsonl','samples.csv','pairs.csv'):
        path = directory / name
        path.write_text(json.dumps(data) if name == 'plot-data.json' else name+' evidence\n')
        manifest['files'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if complete or name == 'plot-data.json':
            assets[name] = creative.inbox_add(conn, settings, source=str(path),project_id=project_id)['id']
    path = directory / 'manifest.json'
    path.write_text(json.dumps(manifest))
    assets['manifest.json'] = creative.inbox_add(conn, settings, source=str(path),project_id=project_id)['id']
    return assets, manifest


def test_discovery_returns_real_values_pairs_and_evidence_without_queue(conn, settings, tmp_path, monkeypatch):
    project = creative.project_create(conn, settings, name='measured')
    assets, _ = dataset(conn, settings, tmp_path, project['id'])
    unrelated = tmp_path / 'manifest.json'; unrelated.write_text('[]')
    creative.inbox_add(conn, settings, source=str(unrelated), project_id=project['id'])
    monkeypatch.setattr(stagecraft, '_call', lambda *a, **k: pytest.fail('discovery cannot queue or call Stagecraft'))
    got = stagecraft.stagecraft_benchmarks(conn, settings, project_id=project['id'])
    assert len(got['errors']) == 1
    assert 'hash map' in got['errors'][0]['error']
    result = got['datasets'][0]
    assert result['profiler_asset_id'] == assets['plot-data.json']
    assert result['summary'] == data_fixture()['summary']
    assert result['sample_pairs'] == data_fixture()['pairs']
    assert result['evidence_asset_ids'] == assets
    assert result['evidence_status'] == 'hash-linked source and raw samples'
    assert '21.000µs' in result['supported_claim']
    assert 'local' in result['supported_claim'].lower()
    assert conn.execute('select count(*) from creative_submission').fetchone()[0] == 0


def test_partial_evidence_is_explicit_and_cross_project_files_not_used(conn, settings, tmp_path):
    a = creative.project_create(conn, settings, name='A')['id']
    b = creative.project_create(conn, settings, name='B')['id']
    dataset(conn, settings, tmp_path, a, complete=True)
    dataset(conn, settings, tmp_path, b, complete=False)
    got = stagecraft.stagecraft_benchmarks(conn, settings, project_id=b)['datasets'][0]
    assert got['evidence_status'] == 'incomplete stored evidence'
    assert 'samples.csv' in got['missing_evidence']
    assert set(got['evidence_asset_ids']) == {'plot-data.json','manifest.json'}


def test_changed_raw_bytes_cannot_be_presented_as_verified(conn, settings, tmp_path):
    project = creative.project_create(conn, settings, name='changed')['id']
    assets, _ = dataset(conn, settings, tmp_path, project)
    row = creative.media_info(conn, settings, asset_id=assets['samples.csv'])
    media.path_for(settings.media_dir,row['ref']).write_text('tampered')
    got = stagecraft.stagecraft_benchmarks(conn, settings, project_id=project)
    assert not got['datasets']
    assert 'hash differs' in got['errors'][0]['error']


def test_malformed_manifest_reported_and_model_cannot_override_project(conn, settings, tmp_path):
    from wonhee.mcp_server import mcp
    from tests.test_direct_stagecraft import context
    a = creative.project_create(conn, settings, name='active')['id']
    b = creative.project_create(conn, settings, name='other')['id']
    dataset(conn, settings, tmp_path, b)
    path = tmp_path / 'manifest.json'; path.write_text('[]')
    creative.inbox_add(conn, settings, source=str(path), project_id=a)
    tools = {t.name:t for t in asyncio.run(mcp.list_tools())}
    assert set(tools['stagecraft_benchmarks'].inputSchema['properties']) == {'limit'}
    result = mcp._tool_manager.get_tool('stagecraft_benchmarks').fn(ctx=context(a))
    assert result['datasets'] == []
    assert 'hash map' in result['errors'][0]['error']
    with pytest.raises(ValueError, match='select'):
        stagecraft.stagecraft_benchmarks(conn, settings)


def test_malformed_evidence_hash_is_actionable(conn, settings, tmp_path):
    project = creative.project_create(conn, settings, name='bad hash')['id']
    _, manifest = dataset(conn, settings, tmp_path, project)
    manifest['files']['samples.csv'] = []
    path = tmp_path / 'manifest.json'; path.write_text(json.dumps(manifest))
    creative.inbox_add(conn, settings, source=str(path), project_id=project)
    result = stagecraft.stagecraft_benchmarks(conn, settings, project_id=project)
    assert result['datasets'] == []
    assert 'invalid SHA-256 for samples.csv' in result['errors'][0]['error']
