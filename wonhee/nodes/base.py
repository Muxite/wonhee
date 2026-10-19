from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel

if TYPE_CHECKING:
    from wonhee.pipeline.context import Ctx


@dataclass(frozen=True)
class Node:
    name: str
    version: int
    inputs: dict[str, str]  # port -> type ("type?" = optional)
    outputs: dict[str, str]
    params: type[BaseModel]  # extra="forbid"
    fn: Callable[["Ctx", dict[str, str], BaseModel], dict[str, str]]  # returns port -> ref
    doc: str  # one line, shown by `node list`
    cacheable: bool = True
    salt: Callable[[Any], dict] | None = None  # settings -> extra cache-key facts (model names)
    # Which warm worker can run it (wonhee/worker.py): "cuda" nodes load a GPU model; "none" runs anywhere.
    # Only routing: a detached chain with a cuda node goes to a live cuda worker instead of a cold child process.
    lane: str = "none"
