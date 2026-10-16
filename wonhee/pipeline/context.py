from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from wonhee import llm
from wonhee.db import queries, repo


class StageFailed(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


@dataclass
class Stage:
    name: str
    needs: tuple[str, ...]
    fn: Callable[["Ctx"], str]


@dataclass
class Ctx:
    conn: sqlite3.Connection
    settings: Any
    deps: Any
    run_id: int
    project_id: int
    request: Any
    creator: Any
    step_id: int
    params: dict
    today: date

    def step_of(self, stage: str) -> int:
        row = queries.current_step(self.conn, self.run_id, stage)
        if row is None:
            raise StageFailed("EXCEPTION", f"stage '{stage}' has no current step")
        return row["id"]

    def param(self, name: str, default):
        return self.params.get(name, default)

    def flag(self, code: str, detail: str = "") -> None:
        repo.add_flag(self.conn, self.step_id, code, detail)

    def add_cost(self, usd: float) -> None:
        repo.add_cost(self.conn, self.step_id, usd)

    def remaining_budget(self) -> float:
        cap = queries.run_row(self.conn, self.run_id)["cost_cap_usd"]
        return cap - queries.run_cost(self.conn, self.run_id)

    def apify(self, actor: str, run_input: dict, *, max_items: int | None = None):
        """Run an actor within the run's remaining budget; cost is recorded even when the run fails."""
        left = self.remaining_budget()
        if left <= 0:
            cap = queries.run_row(self.conn, self.run_id)["cost_cap_usd"]
            raise StageFailed("COST_CAP", f"spent ${cap - left:.2f} of ${cap:.2f} during {self.stage_name()}")
        try:
            items, cost = self.deps.apify.run(actor, run_input, max_total_charge_usd=left, max_items=max_items)
        except Exception as e:
            self.add_cost(getattr(e, "cost_usd", 0.0) or 0.0)
            raise
        self.add_cost(cost)
        return items

    def stage_name(self) -> str:
        row = self.conn.execute("SELECT stage FROM step WHERE id=?", (self.step_id,)).fetchone()
        return row[0] if row else "stage"

    def save_json(self, obj, role: str) -> str:
        sha = repo.save_bytes(self.conn, self.settings.media_dir,
                              json.dumps(obj, indent=1, default=str).encode(), "application/json")
        repo.add_artifact(self.conn, self.step_id, sha, role)
        return sha

    def llm(self, model_cls, *, schema_name: str, system: str, prompt: str, images=()):
        try:
            res = llm.call_json(self.conn, self.settings, model_cls, schema_name=schema_name, system=system,
                                prompt=prompt, images=images, step_id=self.step_id)
        except llm.SchemaFail as e:
            raise StageFailed("SCHEMA_FAIL", str(e)) from e
        self.add_cost(res.cost_usd)
        if res.retried:
            self.flag("SCHEMA_RETRY", schema_name)
        return res.value
