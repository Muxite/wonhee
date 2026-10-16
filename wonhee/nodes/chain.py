from __future__ import annotations

import hashlib
import json
import re

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wonhee.nodes.base import Node
from wonhee.nodes.types import is_literal, parse_type, ref_ok

ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
WIRE_RE = re.compile(r"^([^.\s]+)\.([^.\s]+)$")


class ChainError(ValueError):
    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("chain is invalid:\n" + "\n".join(f"- {p}" for p in problems))


class ChainNode(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    id: str
    node: str
    in_: dict[str, str] = Field(default_factory=dict, alias="in")
    params: dict = Field(default_factory=dict)
    cache: bool = True


class Chain(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nodes: list[ChainNode]
    cost_cap_usd: float | None = Field(default=None, gt=0)
    no_cache: bool = False


def parse_chain(obj: list | dict) -> Chain:
    if isinstance(obj, list):
        obj = {"nodes": obj}
    try:
        return Chain.model_validate(obj)
    except ValidationError as e:
        raise ChainError([_fmt_err(e)]) from e


def _fmt_err(e: ValidationError) -> str:
    return "; ".join(
        ".".join(str(p) for p in err["loc"]) + ": " + err["msg"] if err["loc"] else err["msg"]
        for err in e.errors()
    )


def validate(chain: Chain, nodes: dict[str, Node]) -> list[ChainNode]:
    problems: list[str] = []
    by_id: dict[str, ChainNode] = {}
    for cn in chain.nodes:
        if not ID_RE.match(cn.id):
            problems.append(f"bad id '{cn.id}' (want ^[a-z][a-z0-9_]{{0,31}}$)")
        if cn.id in by_id:
            problems.append(f"duplicate id '{cn.id}'")
        else:
            by_id[cn.id] = cn

    deps: dict[str, set[str]] = {cn.id: set() for cn in chain.nodes}
    for cn in chain.nodes:
        node = nodes.get(cn.node)
        if node is None:
            problems.append(f"{cn.id}: unknown node '{cn.node}'")
            continue
        for port in cn.in_:
            if port not in node.inputs:
                problems.append(f"{cn.id}: unknown input '{port}' for node '{cn.node}'")
        for port, t in node.inputs.items():
            base, optional = parse_type(t)
            value = cn.in_.get(port)
            if value is None:
                if not optional:
                    problems.append(f"{cn.id}: missing required input '{port}'")
                continue
            if is_literal(value):
                if not ref_ok(base, value):
                    problems.append(f"{cn.id}: input '{port}': '{value}' is not a valid {base} ref")
                continue
            m = WIRE_RE.match(value)
            if not m:
                problems.append(f"{cn.id}: input '{port}': `{value}`: no such file, ref or wire")
                continue
            src_id, src_port = m.groups()
            if src_id == cn.id:
                problems.append(f"{cn.id}: input '{port}' is wired to itself")
                continue
            src = by_id.get(src_id)
            if src is None:
                problems.append(f"{cn.id}: input '{port}': `{value}`: no such file, ref or wire (unknown node id '{src_id}')")
                continue
            deps[cn.id].add(src_id)
            src_node = nodes.get(src.node)
            if src_node is None:
                continue  # already reported against the source
            if src_port not in src_node.outputs:
                problems.append(f"{cn.id}: input '{port}': '{src_id}' ({src.node}) has no output '{src_port}'")
                continue
            out_base, _ = parse_type(src_node.outputs[src_port])
            if out_base != base:
                problems.append(
                    f"{cn.id}: type mismatch on '{port}': {value} is {out_base}, expected {base}"
                )
        try:
            node.params.model_validate(cn.params)
        except ValidationError as e:
            problems.append(f"{cn.id}: params: {_fmt_err(e)}")

    order = _toposort(chain.nodes, deps)
    if order is None:
        stuck = sorted(_cycle_ids(chain.nodes, deps))
        problems.append(f"cycle among nodes: {', '.join(stuck)}")
    if problems:
        raise ChainError(problems)
    return order  # type: ignore[return-value]


def _toposort(items: list[ChainNode], deps: dict[str, set[str]]) -> list[ChainNode] | None:
    """Kahn's algorithm; among ready nodes, file order wins."""
    remaining = {cn.id: set(deps[cn.id]) for cn in items}
    out: list[ChainNode] = []
    done: set[str] = set()
    pending = list(range(len(items)))
    while pending:
        idx = next((i for i in pending if remaining[items[i].id] <= done), None)
        if idx is None:
            return None
        pending.remove(idx)
        done.add(items[idx].id)
        out.append(items[idx])
    return out


def _cycle_ids(items: list[ChainNode], deps: dict[str, set[str]]) -> set[str]:
    """Ids left after repeatedly peeling nodes with no unresolved dependencies, then
    dropping those nothing in the remainder depends on."""
    left = {cn.id for cn in items}
    changed = True
    while changed:
        changed = False
        for i in list(left):
            if not (deps[i] & left):
                left.discard(i)
                changed = True
        for i in list(left):
            if not any(i in deps[j] for j in left):
                left.discard(i)
                changed = True
    return left


def cache_key(node: Node, inputs: dict[str, str], params: BaseModel, salt: dict | None = None) -> str:
    blob = json.dumps(
        {"node": node.name, "version": node.version, "inputs": inputs,
         "params": params.model_dump(mode="json"), "salt": salt},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()
