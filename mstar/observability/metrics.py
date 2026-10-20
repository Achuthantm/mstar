"""Prometheus metrics for the API server: ``--enable-metrics`` and ``GET /metrics``.

Requests are counted where they leave :class:`~mstar.api_server.entrypoint.APIServer`,
which ``/generate``, ``/generate/ws``, the OpenAI routes, the Rust-frontend
bridge and the Dynamo bridge all go through. Each family carries the
deployment's ``model_name``; every other label takes values from a fixed set.

Only the API server process imports ``prometheus_client`` and renders
``/metrics``. Other processes (data worker, conductor, workers) send plain
numbers on messages they already send the API server, which records them
here; no multiprocess directory, which would only aggregate one host.

``prometheus_client`` is optional (``pip install 'mstar[metrics]'``) and only
needed when metrics are on.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Histogram,
        generate_latest,
    )
    from prometheus_client.core import GaugeMetricFamily
except ModuleNotFoundError as e:
    # Only a missing package means "not installed"; a broken install must
    # still raise.
    if e.name != "prometheus_client":
        raise
    MISSING_DEPENDENCY: ModuleNotFoundError | None = e
else:
    MISSING_DEPENDENCY = None

if TYPE_CHECKING:
    from mstar.api_server.entrypoint import PendingRequest

INSTALL_HINT = "--enable-metrics needs prometheus_client: pip install 'mstar[metrics]'"

#: ``success``: the result went to the HTTP layer. ``error``: the request
#: failed (engine, preprocessing, result delivery, conductor gone).
#: ``abort``: the client went away, or its finished result was dropped
#: unread. ``timeout``: the per-request ``--timeout`` expired.
OUTCOMES = ("success", "error", "abort", "timeout")

# Outputs range from a text token in milliseconds to minutes of video, and
# --timeout defaults to 600 s.
FIRST_OUTPUT_BUCKETS = (
    0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75,
    1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0, 600.0,
)
E2E_BUCKETS = (
    0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0,
    15.0, 20.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1200.0,
)
# Preprocess and delivery: under a millisecond for a short text prompt, up to
# seconds for a long video upload or a large final payload.
STAGE_BUCKETS = (
    0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
    1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0,
)
# 16-bit PCM: the data worker emits audio as headerless s16le.
_AUDIO_BYTES_PER_SAMPLE = 2


def require_prometheus_client() -> None:
    if MISSING_DEPENDENCY is not None:
        raise RuntimeError(INSTALL_HINT) from MISSING_DEPENDENCY


def modalities_label(modalities: list[str]) -> str:
    """One label value per set of output modalities, e.g. ``audio,text``."""
    return ",".join(sorted(set(modalities)))


class _InFlight:
    """``mstar_requests_in_flight``, read when scraped rather than counted up
    and down, so a request that ends on an unusual path cannot leave it off."""

    def __init__(self, model_name: str, count: Callable[[], int]):
        self.model_name = model_name
        self.count = count

    def collect(self):
        family = GaugeMetricFamily(
            "mstar_requests_in_flight",
            "Requests the API server holds: submitted and not yet returned, "
            "failed or aborted.",
            labels=["model_name"],
        )
        family.add_metric([self.model_name], self.count())
        yield family


class _RequestSeries(NamedTuple):
    """One output-modality set's labeled series, looked up once: a
    ``labels()`` call costs about as much as the observation itself."""
    outcomes: dict[str, Any]
    e2e: Any
    first_output: Any
    preprocess: Any
    engine: Any
    delivery: Any


class ServingMetrics:
    """The API server's registry and its request metrics."""

    def __init__(self, model_name: str, in_flight: Callable[[], int]):
        require_prometheus_client()
        self.model_name = model_name
        self.content_type = CONTENT_TYPE_LATEST
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "mstar_requests",
            "Requests that left the API server, by outcome.",
            ["model_name", "output_modalities", "outcome"],
            registry=self.registry,
        )
        self.e2e_latency = Histogram(
            "mstar_e2e_request_latency_seconds",
            "Successful requests: submission to the result leaving the API server.",
            ["model_name", "output_modalities"],
            buckets=E2E_BUCKETS,
            registry=self.registry,
        )
        self.time_to_first_output = Histogram(
            "mstar_time_to_first_output_seconds",
            "Successful requests: submission to the first result chunk reaching "
            "the API server.",
            ["model_name", "output_modalities"],
            buckets=FIRST_OUTPUT_BUCKETS,
            registry=self.registry,
        )
        self.registry.register(_InFlight(model_name, in_flight))

        # Where a successful request's time went, from the timestamps it
        # carries. The conductor's perf_counter is compared with this
        # process's: both read the host's monotonic clock.
        stages = ["model_name", "output_modalities"]
        self.preprocess_latency = Histogram(
            "mstar_request_preprocess_seconds",
            "Successful requests: submission to the data worker handing the "
            "request to the conductor, including its wait in the data worker's queue.",
            stages, buckets=STAGE_BUCKETS, registry=self.registry,
        )
        self.engine_latency = Histogram(
            "mstar_request_engine_seconds",
            "Successful requests: conductor ingest to conductor finish.",
            stages, buckets=E2E_BUCKETS, registry=self.registry,
        )
        self.delivery_latency = Histogram(
            "mstar_request_delivery_seconds",
            "Successful requests: conductor finish to the result leaving the API server.",
            stages, buckets=STAGE_BUCKETS, registry=self.registry,
        )

        # Output volume, counted per chunk as it reaches the API server, for
        # every request; modality names come from the model's graph.
        self.output_chunks = Counter(
            "mstar_output_chunks",
            "Result chunks that reached the API server, by modality.",
            ["model_name", "modality"], registry=self.registry,
        )
        self.output_bytes = Counter(
            "mstar_output_bytes",
            "Bytes of result chunks that reached the API server, by modality.",
            ["model_name", "modality"], registry=self.registry,
        )
        self.output_audio_seconds = Counter(
            "mstar_output_audio_seconds",
            "Seconds of audio in the result chunks that reached the API server.",
            ["model_name"], registry=self.registry,
        ).labels(model_name)
        self.time_to_first_chunk = Histogram(
            "mstar_time_to_first_chunk_seconds",
            "Successful requests: submission to the first result chunk of each "
            "output modality reaching the API server.",
            ["model_name", "modality"], buckets=FIRST_OUTPUT_BUCKETS, registry=self.registry,
        )
        # Label lookups resolved once per modality set or modality. Two
        # threads may resolve the same key at once; labels() returns the same
        # series to both, so the race is harmless.
        self._request_series: dict[str, _RequestSeries] = {}
        self._first_chunk_series: dict[str, Any] = {}
        self._chunk_series: dict[str, tuple] = {}

    def _series(self, modalities: str) -> _RequestSeries:
        series = self._request_series.get(modalities)
        if series is None:
            labels = (self.model_name, modalities)
            series = _RequestSeries(
                # Exported at 0 from here on: rate() and increase() miss the
                # first increment of a series that appears at 1.
                outcomes={o: self.requests.labels(*labels, o) for o in OUTCOMES},
                e2e=self.e2e_latency.labels(*labels),
                first_output=self.time_to_first_output.labels(*labels),
                preprocess=self.preprocess_latency.labels(*labels),
                engine=self.engine_latency.labels(*labels),
                delivery=self.delivery_latency.labels(*labels),
            )
            self._request_series[modalities] = series
        return series

    def record(self, req: PendingRequest, outcome: str) -> None:
        """Count one request that left the API server's table."""
        now = time.perf_counter()
        series = self._series(modalities_label(req.output_modalities))
        series.outcomes[outcome].inc()
        if outcome != "success":
            return
        timing = req.profile.timing
        series.e2e.observe(now - timing.recv_time)
        if timing.first_chunk_time is not None:
            series.first_output.observe(timing.first_chunk_time - timing.recv_time)
        if timing.preprocess_finish_time is not None:
            series.preprocess.observe(timing.preprocess_finish_time - timing.recv_time)
        # Every successful request got its RequestComplete, which carries
        # both; a value from another host's clock would come out negative.
        if timing.conductor_finish_time is not None and now >= timing.conductor_finish_time:
            series.engine.observe(timing.conductor_finish_time - timing.conductor_ingest_time)
            series.delivery.observe(now - timing.conductor_finish_time)
        for modality, first in req.first_chunk_times.items():
            first_chunk = self._first_chunk_series.get(modality)
            if first_chunk is None:
                first_chunk = self.time_to_first_chunk.labels(self.model_name, modality)
                self._first_chunk_series[modality] = first_chunk
            first_chunk.observe(first - timing.recv_time)

    def record_chunk(self, modality: str, data: bytes, metadata: dict | None) -> None:
        """Count one result chunk that reached the API server (not an error chunk)."""
        series = self._chunk_series.get(modality)
        if series is None:
            series = (self.output_chunks.labels(self.model_name, modality),
                      self.output_bytes.labels(self.model_name, modality))
            self._chunk_series[modality] = series
        chunks, nbytes = series
        chunks.inc()
        nbytes.inc(len(data))
        if modality == "audio" and metadata:
            # Chunks from the result-token path carry no sample rate.
            rate = metadata.get("sample_rate")
            if rate:
                channels = metadata.get("num_channels") or 1
                self.output_audio_seconds.inc(
                    len(data) / (_AUDIO_BYTES_PER_SAMPLE * channels * rate))

    def render(self) -> bytes:
        return generate_latest(self.registry)
