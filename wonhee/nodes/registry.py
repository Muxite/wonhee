from __future__ import annotations

import importlib

from wonhee.nodes.base import Node

NODES: dict[str, Node] = {}
# Node modules register themselves at import; each later task appends its module here.
MODULES: list[str] = ["wonhee.nodes.subs", "wonhee.nodes.grab", "wonhee.nodes.translate", "wonhee.nodes.screen", "wonhee.nodes.burn",
           "wonhee.nodes.look", "wonhee.nodes.text", "wonhee.nodes.carousel",
           "wonhee.nodes.teardown", "wonhee.nodes.larp", "wonhee.nodes.ideate", "wonhee.nodes.vibe", "wonhee.nodes.filter", "wonhee.nodes.filter_probe", "wonhee.nodes.poster",
           "wonhee.nodes.audio", "wonhee.nodes.shots", "wonhee.nodes.viz", "wonhee.nodes.publish", "wonhee.nodes.slop", "wonhee.nodes.compose", "wonhee.nodes.delivery", "wonhee.nodes.inspect", "wonhee.nodes.drive"]
_loaded = False


def register(node: Node) -> Node:
    NODES[node.name] = node
    return node


def unregister(name: str) -> None:
    NODES.pop(name, None)


def load_all() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    for mod in MODULES:
        importlib.import_module(mod)


def get(name: str) -> Node:
    load_all()
    try:
        return NODES[name]
    except KeyError:
        raise KeyError(f"unknown node '{name}'; known: {', '.join(sorted(NODES))}") from None


def all_nodes() -> dict[str, Node]:
    load_all()
    return dict(NODES)
