from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .palette import Palette


@dataclass
class Ctx:
    """Everything an effect may use. ``size`` is (W, H) pixels."""
    size: tuple[int, int]
    seed: int
    palette: Palette
    params: dict[str, Any] = field(default_factory=dict)
    source: np.ndarray | None = None      # optional input image, float32 HxWx3 0..1
    data: Any = None                      # optional real data (a graph, a matrix, a series)

    @property
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    def p(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)


@dataclass
class Effect:
    name: str
    fn: Callable[[Ctx], np.ndarray]
    family: str
    tags: list[str]
    params: dict[str, tuple[Any, str]]    # name -> (default, description)
    doc: str


REGISTRY: dict[str, Effect] = {}


def effect(name: str, *, family: str, tags: list[str] | None = None, params: dict[str, tuple[Any, str]] | None = None):
    def deco(fn: Callable[[Ctx], np.ndarray]):
        if name in REGISTRY and REGISTRY[name].fn.__module__ != fn.__module__:
            raise ValueError(f"effect {name!r} already registered by {REGISTRY[name].fn.__module__}")
        REGISTRY[name] = Effect(name, fn, family, list(tags or []), dict(params or {}), (fn.__doc__ or "").strip())
        return fn
    return deco


def discover() -> None:
    """Import every module under wonhee.art.effects so its @effect runs."""
    from . import effects
    for m in pkgutil.iter_modules(effects.__path__):
        importlib.import_module(f"{effects.__name__}.{m.name}")


def get(name: str) -> Effect:
    discover()
    if name not in REGISTRY:
        raise KeyError(f"unknown effect {name!r}; known: {', '.join(sorted(REGISTRY))}")
    return REGISTRY[name]


def names(family: str | None = None, tag: str | None = None) -> list[str]:
    discover()
    return sorted(n for n, e in REGISTRY.items() if (family is None or e.family == family) and (tag is None or tag in e.tags))
