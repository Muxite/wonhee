from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx


@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str
    age: str | None = None


class Searcher(Protocol):
    def search(self, query: str, count: int = 10) -> list[SearchResult]: ...


class BraveSearch:
    def __init__(self, api_key: str, endpoint: str, client: httpx.Client | None = None):
        self.api_key, self.endpoint = api_key, endpoint
        self.client = client or httpx.Client(timeout=30)

    def search(self, query: str, count: int = 10) -> list[SearchResult]:
        r = self.client.get(self.endpoint, params={"q": query, "count": min(count, 20)},
                            headers={"X-Subscription-Token": self.api_key, "Accept": "application/json"})
        r.raise_for_status()
        return [SearchResult(x.get("url", ""), x.get("title", ""), x.get("description", ""), x.get("age"))
                for x in (r.json().get("web") or {}).get("results", []) if x.get("url")]
