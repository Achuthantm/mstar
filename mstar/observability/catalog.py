"""Metric families whose values come from a process other than the API server.

Dependency-free, so the conductor and workers can import it: they record into
:mod:`mstar.observability.recorders` by these names, and the API server builds
the Prometheus families from the same entries, so the names cannot drift.
``model_name`` is added by the API server and is not listed in ``labels``.
"""

from __future__ import annotations

from dataclasses import dataclass

COUNTER, GAUGE, HISTOGRAM = "counter", "gauge", "histogram"


@dataclass(frozen=True)
class Family:
    name: str
    kind: str
    help: str
    labels: tuple[str, ...]
    #: Histogram upper bounds, ascending, without ``+Inf``.
    buckets: tuple[float, ...] = ()


#: How often the conductor sends its status message while metrics are on.
STATUS_INTERVAL_S = 1.0

#: ``waiting``: queued at ``max_concurrent_requests``, not dispatched yet.
#: ``running``: dispatched to workers and not finished. ``draining``: finished
#: (completed, aborted or failed) and waiting for every reader to release its
#: outputs before the workers free its state; it still counts toward
#: ``max_concurrent_requests``.
CONDUCTOR_REQUEST_STATES = ("waiting", "running", "draining")

CONDUCTOR_REQUESTS = Family(
    "mstar_conductor_requests", GAUGE,
    "Requests the conductor holds, by state.",
    ("state",),
)
COMPONENT_IN_FLIGHT = Family(
    "mstar_component_in_flight", GAUGE,
    "Running requests with a pass on this graph node that the conductor "
    "dispatched and whose completion it has not received yet.",
    ("component",),
)
REQUEST_QUEUE_SECONDS = Family(
    "mstar_request_queue_seconds", HISTOGRAM,
    "Successful requests: the conductor taking the request in to admitting it "
    "for dispatch (zero unless it queued at max_concurrent_requests).",
    ("output_modalities",),
    buckets=(
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
        10.0, 30.0, 60.0, 120.0, 300.0, 600.0,
    ),
)

#: Families the conductor's status message carries, exported from the latest
#: message.
STATUS_FAMILIES: dict[str, Family] = {
    f.name: f for f in (CONDUCTOR_REQUESTS, COMPONENT_IN_FLIGHT)
}
