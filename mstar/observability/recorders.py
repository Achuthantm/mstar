"""Dependency-free accumulators for the families in :mod:`.catalog`.

A process other than the API server keeps one recorder per family and sends
``snapshot()`` on a message the API server already receives; the API server
exports the latest snapshot of each family. Snapshots are cumulative, never
deltas, so a lost or reordered message cannot corrupt a counter.

Not thread-safe: update and snapshot a recorder from one thread.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import accumulate

from mstar.observability.catalog import HISTOGRAM, Family

Labels = tuple[str, ...]


@dataclass
class MetricSnapshot:
    """One family's series, columnar so the wire codec encodes it typed."""
    name: str
    #: One row per series, in the family's label order.
    label_values: list[list[str]]
    #: Counter or gauge: one value per series.
    values: list[float] = field(default_factory=list)
    #: Histogram: per series, cumulative counts per bucket, ``+Inf`` last
    #: (so the last is the observation count), and the sum of observations.
    bucket_counts: list[list[int]] = field(default_factory=list)
    sums: list[float] = field(default_factory=list)


class Gauge:
    def __init__(self, family: Family):
        self.family = family
        self._values: dict[Labels, float] = {}

    def set(self, labels: Labels, value: float) -> None:
        self._values[labels] = value

    def snapshot(self) -> MetricSnapshot:
        return MetricSnapshot(
            self.family.name, [list(k) for k in self._values], list(self._values.values()),
        )


class Counter(Gauge):
    def inc(self, labels: Labels, amount: float = 1.0) -> None:
        self._values[labels] = self._values.get(labels, 0.0) + amount


class Histogram:
    """Fixed buckets from the family. ``observe`` bumps one bucket; the
    cumulative counts Prometheus wants are built at ``snapshot``."""

    def __init__(self, family: Family):
        self.family = family
        self._bounds = family.buckets
        self._counts: dict[Labels, list[int]] = {}
        self._sums: dict[Labels, float] = {}

    def observe(self, labels: Labels, value: float) -> None:
        counts = self._counts.get(labels)
        if counts is None:
            counts = self._counts[labels] = [0] * (len(self._bounds) + 1)
            self._sums[labels] = 0.0
        # le is inclusive: a value equal to a bound falls in that bucket.
        counts[bisect_left(self._bounds, value)] += 1
        self._sums[labels] += value

    def snapshot(self) -> MetricSnapshot:
        return MetricSnapshot(
            self.family.name,
            [list(k) for k in self._counts],
            bucket_counts=[list(accumulate(c)) for c in self._counts.values()],
            sums=list(self._sums.values()),
        )


def merge(family: Family, snapshots: Iterable[MetricSnapshot]) -> MetricSnapshot:
    """One snapshot of ``family`` summing the series of ``snapshots`` that
    share label values: several processes' views of one deployment (the
    replicas of a graph node on different workers, say)."""
    values: dict[Labels, float] = {}
    counts: dict[Labels, list[int]] = {}
    sums: dict[Labels, float] = {}
    for snap in snapshots:
        if family.kind != HISTOGRAM:
            for row, value in zip(snap.label_values, snap.values, strict=True):
                key = tuple(row)
                values[key] = values.get(key, 0.0) + value
            continue
        for row, row_counts, total in zip(
                snap.label_values, snap.bucket_counts, snap.sums, strict=True):
            key = tuple(row)
            have = counts.get(key)
            counts[key] = list(row_counts) if have is None else [
                a + b for a, b in zip(have, row_counts, strict=True)]
            sums[key] = sums.get(key, 0.0) + total
    if family.kind != HISTOGRAM:
        return MetricSnapshot(family.name, [list(k) for k in values], list(values.values()))
    return MetricSnapshot(
        family.name, [list(k) for k in counts],
        bucket_counts=list(counts.values()), sums=list(sums.values()),
    )
