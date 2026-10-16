"""Run a validated chain on the fixed orchestrator: one Stage per chain node."""
from __future__ import annotations

from wonhee import media
from wonhee.db import queries, repo
from wonhee.nodes import registry
from wonhee.nodes.base import Node
from wonhee.nodes.chain import WIRE_RE, Chain, ChainNode, cache_key, validate
from wonhee.nodes.types import is_literal, parse_type, ref_ok


def stages_for_chain(chain: Chain):
    # Imported here: pipeline.registry imports scout stages, and stages_for_run imports this module.
    from wonhee.pipeline.context import Stage

    nodes = {cn.node: registry.get(cn.node) for cn in chain.nodes}
    stages = []
    for cn in validate(chain, nodes):
        node = nodes[cn.node]
        ups = tuple(dict.fromkeys(WIRE_RE.match(v).group(1) for v in cn.in_.values() if not is_literal(v)))

        def fn(ctx, cn=cn, node=node):
            return _run_node(ctx, chain, cn, node)

        stages.append(Stage(cn.id, ups, fn, dict(cn.params)))
    return stages


def _resolve_inputs(ctx, cn: ChainNode) -> dict[str, str]:
    from wonhee.pipeline.context import StageFailed

    inputs: dict[str, str] = {}
    for port, value in cn.in_.items():
        if is_literal(value):
            inputs[port] = value
            continue
        up_id, up_port = WIRE_RE.match(value).groups()
        up = queries.current_step(ctx.conn, ctx.run_id, up_id)
        out = queries.step_outputs(ctx.conn, up["id"]).get(up_port) if up is not None else None
        if out is None:
            raise StageFailed("EXCEPTION", f"{cn.id}: input '{port}': no output '{up_port}' recorded for '{up_id}'")
        inputs[port] = out["ref"]
    return inputs


def _files_present(ctx, hit: dict[str, tuple[str, str]]) -> bool:
    """A cached result is only good while the files it points at are still in the media store."""
    return all(media.path_for(ctx.settings.media_dir, ref).is_file()
               for typ, ref in hit.values() if typ != "content")


def _out_types(node: Node) -> dict[str, str]:
    return {p: parse_type(t)[0] for p, t in node.outputs.items()}


def _run_node(ctx, chain: Chain, cn: ChainNode, node: Node) -> str:
    from wonhee.pipeline.context import StageFailed

    inputs = _resolve_inputs(ctx, cn)
    params = node.params.model_validate(cn.params)
    types = _out_types(node)
    key = cache_key(node, inputs, params, node.salt(ctx.settings) if node.salt else None)
    flag = ctx.params.get("no_cache")
    override = flag.strip().lower() in ("1", "true", "yes") if isinstance(flag, str) else bool(flag)
    lookup = node.cacheable and cn.cache and not chain.no_cache and not override
    if lookup:
        hit = queries.cache_get(ctx.conn, key)
        if hit is not None and hit.keys() == types.keys() and _files_present(ctx, hit):
            repo.set_step_outputs(ctx.conn, ctx.step_id, hit)
            ctx.flag("CACHED", node.name)
            return "ok"
    outs = node.fn(ctx, inputs, params)
    if set(outs) != set(types):
        raise StageFailed("EXCEPTION", f"{node.name} returned ports {sorted(outs)}, expected {sorted(types)}")
    for port, ref in outs.items():
        if not ref_ok(types[port], ref):
            raise StageFailed("EXCEPTION", f"{node.name}.{port}: '{ref}' is not a valid {types[port]} ref")
    typed = {p: (types[p], r) for p, r in outs.items()}
    repo.set_step_outputs(ctx.conn, ctx.step_id, typed)
    if node.cacheable and not ctx.weak:
        repo.put_cache(ctx.conn, key, node.name, node.version, ctx.step_id, typed)
    return "weak" if ctx.weak else "ok"
