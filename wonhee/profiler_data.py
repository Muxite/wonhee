"""Bounded measured echo data for a profiler replay, validated before coding is queued."""
from __future__ import annotations

import math
from bisect import bisect_right
from collections import Counter
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

Positive = Annotated[float, Field(gt=0)]


class DataModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)


class Timing(DataModel):
    n: int = Field(ge=1, le=20000)
    median_us: Positive
    mean_us: Positive
    p95_us: Positive
    p99_us: Positive
    min_us: Positive
    max_us: Positive

    @model_validator(mode='after')
    def ordered(self):
        if not self.min_us <= self.median_us <= self.p95_us <= self.p99_us <= self.max_us:
            raise ValueError('timing quantiles must be ordered')
        if not self.min_us <= self.mean_us <= self.max_us:
            raise ValueError('mean must lie within measured bounds')
        return self


class Summary(DataModel):
    stdio: Timing
    http: Timing
    http_over_stdio_median_rtt: Positive

    @model_validator(mode='after')
    def ratio(self):
        expected = self.http.median_us / self.stdio.median_us
        if not math.isclose(self.http_over_stdio_median_rtt, expected, rel_tol=1e-6):
            raise ValueError('median RTT ratio must match measurements')
        return self


class Pair(DataModel):
    pair_id: str = Field(min_length=1, max_length=80)
    payload_bytes: int = Field(ge=1, le=16384)
    stdio_us: Positive
    http_us: Positive


class Histogram(DataModel):
    edges_us: list[Positive] = Field(min_length=2, max_length=256)
    stdio: list[Annotated[int, Field(ge=0)]] = Field(min_length=1, max_length=255)
    http: list[Annotated[int, Field(ge=0)]] = Field(min_length=1, max_length=255)

    @model_validator(mode='after')
    def bins(self):
        if any(a >= b for a, b in zip(self.edges_us, self.edges_us[1:])):
            raise ValueError('histogram edges must increase')
        if len(self.stdio) != len(self.edges_us)-1 or len(self.http) != len(self.edges_us)-1:
            raise ValueError('histogram counts must match bin edges')
        return self


class ProfilerData(DataModel):
    seed: int
    summary: Summary
    histogram: Histogram
    sample_stride: int = Field(ge=1, le=20000)
    pairs: list[Pair] = Field(min_length=1, max_length=1024)
    paragraph_previews: list[Annotated[str, Field(max_length=1000)]] = Field(min_length=1, max_length=16)
    sample_count_per_transport: int = Field(ge=1, le=20000)
    paragraph_count: int = Field(ge=1, le=2000)
    scope: str = Field(min_length=1, max_length=500)
    animation: str = Field(min_length=1, max_length=200)

    @model_validator(mode='after')
    def consistent(self):
        n = self.sample_count_per_transport
        if self.summary.stdio.n != n or self.summary.http.n != n:
            raise ValueError('summary counts must match sample_count_per_transport')
        if sum(self.histogram.stdio) != n or sum(self.histogram.http) != n:
            raise ValueError('histograms must retain every sample')
        if n < self.paragraph_count or n % self.paragraph_count or n // self.paragraph_count > 10:
            raise ValueError('sample count must represent 1..10 complete paragraph trials')
        if len(self.pairs) != math.ceil(n / self.sample_stride):
            raise ValueError('replay pair count must match sample_stride')
        for transport in ('http', 'stdio'):
            stats = getattr(self.summary, transport)
            edges = self.histogram.edges_us
            if not edges[0] <= stats.min_us <= stats.max_us < edges[-1]:
                raise ValueError('histogram edges must cover every measured timing')
            observed = Counter(bisect_right(edges, getattr(pair, transport + '_us')) - 1 for pair in self.pairs)
            counts = getattr(self.histogram, transport)
            if any(index < 0 or index >= len(counts) or count > counts[index] for index, count in observed.items()):
                raise ValueError('replay bin counts cannot exceed the full histogram')
        if len(self.pairs) > n or len({p.pair_id for p in self.pairs}) != len(self.pairs):
            raise ValueError('replay pairs must have unique identities and not exceed sample count')
        for pair in self.pairs:
            for transport in ('http', 'stdio'):
                stats = getattr(self.summary, transport)
                value = getattr(pair, transport + '_us')
                if not stats.min_us <= value <= stats.max_us:
                    raise ValueError('replay timings must lie within measured bounds')
        return self
