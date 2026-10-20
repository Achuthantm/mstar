"""Prometheus metrics at ``/metrics`` (``--enable-metrics``).

Requests go through the FastAPI app and the API server's real message thread;
only the data worker and the conductor are stand-ins, fed the messages the
real ones send. Each test scrapes ``/metrics`` and reads the samples back.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import queue
import sys
import threading
import time
import types

import httpx
import pytest
from api_server_helpers import make_test_api_server
from fastapi import HTTPException
from fastapi.testclient import TestClient

import mstar.communication.wire_types  # noqa: F401  (registers the wire tags)
from mstar.api_server import entrypoint
from mstar.api_server.request_types import (
    APIServerMessage,
    ConductorStatus,
    DataWorkerProfile,
    RequestComplete,
    RequestFailed,
    ResultChunk,
)
from mstar.communication import wire
from mstar.observability import catalog
from mstar.observability import metrics as metrics_module
from mstar.observability.recorders import Counter, Gauge, Histogram, MetricSnapshot

TEXT = dict(input_modalities=["text"], output_modalities=["text"])


class _DataWorker:
    """What the API server calls on its data worker. Like the real one, it
    reports each request's preprocess-finish time once asked to."""

    def __init__(self):
        self.inputs = []
        self.aborted = []
        self.chunks = queue.Queue()
        self.profiles = queue.Queue()
        self.reporting = False

    def report_preprocess_finish(self):
        self.reporting = True

    def new_request(self, inp):
        self.inputs.append(inp)
        if self.reporting:
            self.profiles.put(DataWorkerProfile(
                request_id=inp.request_id, preprocess_finish_time=time.perf_counter(),
            ))

    def abort_request(self, key):
        self.aborted.append(key)

    def get_result_chunks(self):
        out = []
        while not self.chunks.empty():
            out.append(self.chunks.get())
        return out

    def get_profile_updates(self):
        out = []
        while not self.profiles.empty():
            out.append(self.profiles.get())
        return out

    def has_pending_tensors(self, key):
        return False

    def received_final_chunks(self, key, output_chunk_counts):
        return True

    def finished_reading(self, key, drained=True):
        pass


class _Conductor:
    """The API server's socket to the conductor."""

    def __init__(self):
        self.inbox = queue.Queue()

    def get_all_new_messages(self):
        """Everything queued; a queued list arrives as one batch."""
        out = []
        while not self.inbox.empty():
            item = self.inbox.get()
            out.extend(item if isinstance(item, list) else [item])
        return out

    def send(self, entity, msg):
        pass


class _Served:
    def __init__(self, server, data_worker, conductor):
        self.server, self.data_worker, self.conductor = server, data_worker, conductor

    def key(self, i=0):
        return self.data_worker.inputs[i].request_id

    def answer(self, key, data=b"hi", chunks=None, ingest=None, finish=None, **stamps):
        """The engine's result chunks (one text chunk unless given), then its
        completion, as the real data worker and conductor deliver them.
        ``ingest`` and ``finish`` are the conductor's perf_counter stamps;
        ``stamps`` are further ``RequestComplete`` fields (its arrival stamp)."""
        chunks = chunks or [ResultChunk(request_id=key, modality="text", data=data)]
        for chunk in chunks:
            self.data_worker.chunks.put(chunk)
        finish = time.perf_counter() if finish is None else finish
        fields = {"conductor_ingest_time": finish if ingest is None else ingest,
                  "conductor_finish_time": finish, **stamps}
        self.conductor.inbox.put(APIServerMessage("request_complete", RequestComplete(
            request_id=key, output_chunk_counts={"out": len(chunks)}, **fields,
        )))

    def fail(self, key, status=500):
        self.conductor.inbox.put(APIServerMessage("request_failed", RequestFailed(
            request_id=key, error_message="boom", status=status,
        )))

    def scrape(self) -> dict:
        r = TestClient(entrypoint.app).get("/metrics")
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("text/plain")
        return _parse(r.text)


def _complete(key, conductor_ingest_time=0.0, conductor_finish_time=0.0, **kwargs):
    return APIServerMessage("request_complete", RequestComplete(
        request_id=key, output_chunk_counts={"text": 1},
        conductor_ingest_time=conductor_ingest_time, conductor_finish_time=conductor_finish_time,
        **kwargs,
    ))


def _parse(text: str) -> dict:
    from prometheus_client.parser import text_string_to_metric_families

    samples = {}
    for family in text_string_to_metric_families(text):
        for s in family.samples:
            samples[(s.name, tuple(sorted(s.labels.items())))] = s.value
    return samples


@contextlib.contextmanager
def _serve(monkeypatch, tmp_path, metrics=True):
    data_worker, conductor = _DataWorker(), _Conductor()
    server = make_test_api_server(
        preprocess_worker=data_worker, communicator=conductor, upload_dir=tmp_path,
    )
    if metrics:
        server.enable_metrics()
    monkeypatch.setattr(entrypoint, "api_server", server)
    thread = threading.Thread(target=server._process_messages, daemon=True)
    thread.start()
    try:
        yield _Served(server, data_worker, conductor)
    finally:
        server.running = False
        thread.join(timeout=5)


@pytest.fixture
def served(monkeypatch, tmp_path):
    pytest.importorskip("prometheus_client")
    with _serve(monkeypatch, tmp_path) as s:
        yield s


def _labels(**kv):
    return tuple(sorted({"model_name": "dummy", **kv}.items()))


AGE = ("mstar_conductor_status_age_seconds", _labels())


def _without_age(samples: dict) -> dict:
    """Every sample but the status age, which moves between two scrapes."""
    assert AGE in samples
    return {k: v for k, v in samples.items() if k != AGE}


def _count(samples, outcome, modalities="text"):
    return samples.get(("mstar_requests_total", _labels(output_modalities=modalities, outcome=outcome)), 0)


def _wait(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline
        time.sleep(0.005)


def test_metrics_answer_404_until_enabled(monkeypatch):
    monkeypatch.setattr(entrypoint, "api_server", make_test_api_server())
    r = TestClient(entrypoint.app).get("/metrics")
    assert r.status_code == 404
    assert "--enable-metrics" in r.json()["detail"]


@pytest.mark.parametrize("streaming", [False, True])
def test_a_generate_request_moves_the_counter_and_both_histograms(served, streaming):
    form = {"text": "hi", "output_modalities": "text,audio", "streaming": str(streaming).lower()}
    labels = _labels(output_modalities="audio,text")

    async def run():
        transport = httpx.ASGITransport(app=entrypoint.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            pending = asyncio.create_task(client.post("/generate", data=form))
            while not served.data_worker.inputs:
                await asyncio.sleep(0.005)
            during = served.scrape()
            served.answer(served.key())
            return await pending, during

    r, during = asyncio.run(run())

    assert r.status_code == 200, r.text
    assert during[("mstar_requests_in_flight", _labels())] == 1
    after = served.scrape()
    assert after[("mstar_requests_in_flight", _labels())] == 0
    assert _count(after, "success", "audio,text") == 1
    for name in ("mstar_e2e_request_latency_seconds", "mstar_time_to_first_output_seconds"):
        assert after[(f"{name}_count", labels)] == 1
        observed = after[(f"{name}_sum", labels)]
        buckets = {
            float(dict(k[1])["le"]): v for k, v in after.items()
            if k[0] == f"{name}_bucket" and dict(k[1])["output_modalities"] == "audio,text"
        }
        assert len(buckets) > 2 and float("inf") in buckets
        # The one observation sits in every bucket at or above its value.
        assert buckets == {le: float(le >= observed) for le in buckets}, name
    first = after[("mstar_time_to_first_output_seconds_sum", labels)]
    e2e = after[("mstar_e2e_request_latency_seconds_sum", labels)]
    assert 0 < first <= e2e


def test_metrics_need_no_api_key_while_generate_still_does(served):
    """Scrapers carry no key, like the /health probe; the request routes keep it."""
    from mstar.api_server import auth

    entrypoint.api_key_guard.set("s3cret-key")
    try:
        client = TestClient(entrypoint.app)
        assert client.post("/generate", data={"text": "hi"}).status_code == 401
        assert client.post("/generate", data={"text": "hi"},
                           headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert not served.data_worker.inputs
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "mstar_requests_in_flight" in r.text
    finally:
        entrypoint.api_key_guard.set(auth.resolve_api_key())


class _BridgeSocket:
    """The Rust-frontend bridge's transport: each receive hands the serve loop
    the next scripted batch (a callable runs first and returns it), then
    stops the loop."""

    def __init__(self, bridge, script):
        self.bridge, self.script = bridge, list(script)

    def get_all_new_messages(self, blocking=True, timeout_s=None):
        if not self.script:
            self.bridge.running = False
            return []
        step = self.script.pop(0)
        return step() if callable(step) else step


def _serve_bridge(server, script, sent: list) -> None:
    """Run the real ``RustFrontendBridge`` serve loop over ``script``,
    appending what it sends the frontend to ``sent``."""
    from mstar.api_server.rust_frontend import RustFrontendBridge

    bridge = RustFrontendBridge.__new__(RustFrontendBridge)
    bridge.server, bridge._keys, bridge.running = server, {}, True
    bridge._send = sent.append
    bridge.comm = _BridgeSocket(bridge, script)
    asyncio.run(bridge._serve())


def test_the_rust_bridge_answers_metrics_with_what_the_python_route_serves(served):
    """``mstar-server``'s ``/metrics`` relays this reply, so both frontends
    show the same registry, and requests that came in through the bridge
    move it."""
    sent = []

    def scrape_while_in_flight():
        _wait(lambda: served.data_worker.inputs)
        return [{"t": "metrics", "rid": "metrics-a", "sub": 2}]

    def answer_then_scrape():
        served.answer(served.key())
        _wait(lambda: any(m["t"] == "done" and m["sub"] == 1 for m in sent))
        return [{"t": "metrics", "rid": "metrics-b", "sub": 3}]

    _serve_bridge(served.server, [
        [{"t": "submit", "rid": "X", "sub": 1, "text": "hi", "streaming": True, **TEXT}],
        scrape_while_in_flight,
        answer_then_scrape,
    ], sent)

    replies = {m["sub"]: m for m in sent if m["t"] == "metrics"}
    assert sorted(replies) == [2, 3]
    for reply, rid in ((replies[2], "metrics-a"), (replies[3], "metrics-b")):
        assert reply["rid"] == rid
        assert reply["metadata"] == {"content_type": served.server.metrics.content_type}
    assert _parse(replies[2]["data"].decode())[("mstar_requests_in_flight", _labels())] == 1
    after = _parse(replies[3]["data"].decode())
    assert after[("mstar_requests_in_flight", _labels())] == 0
    assert _count(after, "success") == 1
    assert _without_age(after) == _without_age(served.scrape())


def test_the_rust_bridge_answers_metrics_off_when_they_are_off(tmp_path):
    """Empty metadata is how ``mstar-server`` knows to answer 404; the
    frontend's refusal counts are ignored then."""
    sent = []
    refusals = {"route": ["/generate"], "status": [401], "count": [2]}
    _serve_bridge(make_test_api_server(upload_dir=tmp_path),
                  [[{"t": "metrics", "rid": "metrics-a", "sub": 5, "refusals": refusals}]], sent)
    assert sent == [{"t": "metrics", "rid": "metrics-a", "sub": 5, "data": b"", "metadata": {}}]


def _refusals(samples) -> dict:
    return {(dict(k[1])["route"], dict(k[1])["status"]): v
            for k, v in samples.items() if k[0] == "mstar_http_refusals_total"}


def test_the_rust_bridge_exports_the_frontend_s_refusal_counts(served):
    """``mstar-server`` sends its cumulative counts with each scrape; they
    come back in the registry's text, and a count from an older scrape that
    crossed a newer one never lowers it."""
    sent = []
    _serve_bridge(served.server, [
        [{"t": "metrics", "rid": "metrics-a", "sub": 2, "refusals": {
            "route": ["/generate", "unmatched"], "status": [401, 404], "count": [3, 1]}}],
        [{"t": "metrics", "rid": "metrics-b", "sub": 3, "refusals": {
            "route": ["/generate", "/generate"], "status": [401, 409], "count": [2, 1]}}],
    ], sent)

    first, second = (_refusals(_parse(m["data"].decode())) for m in sent)
    assert first == {("/generate", "401"): 3, ("unmatched", "404"): 1}
    assert second == {("/generate", "401"): 3, ("/generate", "409"): 1, ("unmatched", "404"): 1}
    assert _refusals(served.scrape()) == second


_KEY = {"Authorization": "Bearer s3cret-key"}
_CHAT = {"model": "dummy", "messages": [{"role": "user", "content": "hi"}]}


@pytest.fixture
def refusing(served, monkeypatch):
    """The served app with an API key and a 1 MiB body limit."""
    from mstar.api_server import auth

    monkeypatch.setenv("MSTAR_MAX_BODY_MB", "1")
    # BodyLimitMiddleware reads the limit when the stack is built.
    monkeypatch.setattr(entrypoint.app, "middleware_stack", entrypoint.app.build_middleware_stack())
    entrypoint.api_key_guard.set("s3cret-key")
    yield TestClient(entrypoint.app)
    entrypoint.api_key_guard.set(auth.resolve_api_key())


def _chunked(total: int):
    for _ in range(total // 2**16):
        yield b"x" * 2**16


def _in_flight_id(served, client):
    served.server.submit_request(text="x", request_id="X", streaming=False, **TEXT)
    return client.post("/generate", data={"text": "hi", "request_id": "X"}, headers=_KEY)


@pytest.mark.parametrize("send,status,route", [
    (lambda s, c: c.post("/generate", data={"text": "hi"}), 401, "/generate"),
    (lambda s, c: c.post("/no/such/route"), 401, "unmatched"),
    # Declared Content-Length: refused before routing; chunked: while the route reads it.
    (lambda s, c: c.post("/v1/chat/completions", content=b"x" * 2**21,
                         headers={**_KEY, "content-type": "application/json"}), 413, "/v1/chat/completions"),
    (lambda s, c: c.post("/generate", content=_chunked(2**21),
                         headers={**_KEY, "content-type": "multipart/form-data; boundary=b"}), 413, "/generate"),
    (lambda s, c: c.post("/v1/chat/completions", json={"messages": "hi"}, headers=_KEY), 422, "/v1/chat/completions"),
    (lambda s, c: c.post("/generate", data={"text": "hi", "model_kwargs": "{"}, headers=_KEY), 400, "/generate"),
    (_in_flight_id, 409, "/generate"),
    (lambda s, c: c.get("/no/such/route", headers=_KEY), 404, "unmatched"),
    # The served model has no OpenAI adapter.
    (lambda s, c: c.post("/v1/chat/completions", json=_CHAT, headers=_KEY), 404, "/v1/chat/completions"),
    (lambda s, c: c.get("/generate", headers=_KEY), 405, "/generate"),
], ids=["401", "401-unmatched", "413-declared", "413-chunked", "422", "400", "409", "404-unmatched",
        "404-no-adapter", "405"])
def test_a_request_refused_before_the_api_server_counts_by_route_template_and_status(
        served, refusing, send, status, route):
    assert send(served, refusing).status_code == status
    assert _refusals(served.scrape()) == {(route, str(status)): 1}
    assert len(served.data_worker.inputs) == (1 if status == 409 else 0)


def test_every_route_keeps_its_template_when_refused_before_routing(served, refusing):
    """The API key refuses before the router picks a route; the refusal still
    carries the route's template, for every route the app serves (a router
    included later must also be listed in ``RefusalMiddleware``'s routes)."""
    from mstar.api_server.auth import UNAUTHENTICATED_PATHS

    paths = {path: next(iter(ops)) for path, ops in entrypoint.app.openapi()["paths"].items()
             if path not in UNAUTHENTICATED_PATHS}
    assert {"/generate", "/v1/chat/completions", "/v1/images/edits"} <= set(paths)
    for path, method in paths.items():
        assert refusing.request(method, path).status_code == 401, path
    assert _refusals(served.scrape()) == {(path, "401"): 1 for path in paths}


def test_a_request_the_engine_fails_is_not_a_refusal_but_one_refused_first_is(served, refusing):
    """Both answer 400. The submitted one is counted once, as an error in
    ``mstar_requests_total``; only the other is a refusal."""

    async def run():
        transport = httpx.ASGITransport(app=entrypoint.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", headers=_KEY) as client:
            pending = asyncio.create_task(client.post("/generate", data={"text": "hi", "streaming": "false"}))
            while not served.data_worker.inputs:
                await asyncio.sleep(0.005)
            served.fail(served.key(), status=400)
            failed = await pending
            refused = await client.post("/generate", data={"text": "hi", "model_kwargs": "[1]"})
            return failed, refused

    failed, refused = asyncio.run(run())

    assert (failed.status_code, refused.status_code) == (400, 400)
    samples = served.scrape()
    assert _refusals(samples) == {("/generate", "400"): 1}
    assert _count(samples, "error") == 1


def test_a_served_request_is_not_a_refusal(served):
    async def run():
        transport = httpx.ASGITransport(app=entrypoint.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            pending = asyncio.create_task(client.post("/generate", data={"text": "hi", "streaming": "false"}))
            while not served.data_worker.inputs:
                await asyncio.sleep(0.005)
            served.answer(served.key())
            return await pending

    assert asyncio.run(run()).status_code == 200
    samples = served.scrape()
    assert "mstar_http_refusals_total" in TestClient(entrypoint.app).get("/metrics").text
    assert _refusals(samples) == {}
    assert _count(samples, "success") == 1


def test_refusals_count_from_when_metrics_are_turned_on(monkeypatch):
    from mstar.api_server import auth

    server = make_test_api_server(preprocess_worker=_DataWorker())
    monkeypatch.setattr(entrypoint, "api_server", server)
    client = TestClient(entrypoint.app)
    entrypoint.api_key_guard.set("s3cret-key")
    try:
        assert client.post("/generate", data={"text": "hi"}).status_code == 401
        server.enable_metrics()
        assert client.post("/generate", data={"text": "hi"}).status_code == 401
    finally:
        entrypoint.api_key_guard.set(auth.resolve_api_key())
    assert _refusals(_parse(client.get("/metrics").text)) == {("/generate", "401"): 1}


def _engine_fails(served):
    async def run():
        served.server.submit_request(text="x", request_id="r", streaming=False, **TEXT)
        served.fail(served.key())
        with pytest.raises(HTTPException) as e:
            await served.server.collect_results("r")
        assert e.value.status_code == 500
    asyncio.run(run())


def _stream_is_dropped_after_its_first_chunk(served):
    async def run():
        served.server.submit_request(text="x", request_id="r", **TEXT)
        stream = served.server.iter_result_chunks("r")
        served.data_worker.chunks.put(ResultChunk(request_id=served.key(), modality="text", data=b"a"))
        assert (await stream.__anext__()).data == b"a"
        await stream.aclose()
    asyncio.run(run())


def _non_streaming_times_out(served):
    served.server.timeout_seconds = 0.05

    async def run():
        served.server.submit_request(text="x", request_id="r", streaming=False, **TEXT)
        with pytest.raises(HTTPException, match="timed out"):
            await served.server.collect_results("r")
    asyncio.run(run())


def _stream_times_out(served):
    served.server.timeout_seconds = 0.05

    async def run():
        served.server.submit_request(text="x", request_id="r", **TEXT)
        with pytest.raises(HTTPException, match="timed out"):
            async for _ in served.server.iter_result_chunks("r"):
                pass
    asyncio.run(run())


def _finished_result_is_released_unread(served):
    served.server.submit_request(text="x", request_id="r", streaming=False, **TEXT)
    served.answer(served.key())
    _wait(served.server.pending_requests[served.key()].event.is_set)
    served.server.release_request("r")


@pytest.mark.parametrize("end,outcome", [
    (_engine_fails, "error"),
    (_stream_is_dropped_after_its_first_chunk, "abort"),
    (_non_streaming_times_out, "timeout"),
    (_stream_times_out, "timeout"),
    (_finished_result_is_released_unread, "abort"),
], ids=lambda v: v if isinstance(v, str) else v.__name__.strip("_"))
def test_each_way_a_request_ends_counts_once_under_its_outcome(served, end, outcome):
    end(served)

    samples = served.scrape()
    counted = {o: _count(samples, o) for o in metrics_module.OUTCOMES if _count(samples, o)}
    assert counted == {outcome: 1}
    # Latency histograms hold successful requests only.
    assert samples.get(("mstar_e2e_request_latency_seconds_count", _labels(output_modalities="text")), 0) == 0
    assert samples[("mstar_requests_in_flight", _labels())] == 0


def test_every_outcome_is_exported_at_zero_once_its_modalities_are_seen(served):
    """A series that first appears at 1 loses that increment to rate() and
    increase(), so the first error must find its series already at 0."""
    _finished_result_is_released_unread(served)
    samples = served.scrape()
    exported = {o: samples[("mstar_requests_total", _labels(output_modalities="text", outcome=o))]
                for o in metrics_module.OUTCOMES
                if ("mstar_requests_total", _labels(output_modalities="text", outcome=o)) in samples}
    assert exported == {"success": 0, "error": 0, "abort": 1, "timeout": 0}
    assert ("mstar_requests_total", _labels(output_modalities="audio", outcome="error")) not in samples


def _hist(samples, name, **labels):
    """(count, sum) of one histogram series."""
    key = _labels(**labels)
    return samples.get((f"{name}_count", key), 0), samples.get((f"{name}_sum", key), 0.0)


def test_a_request_s_stages_are_observed_from_its_timestamps(served):
    """Preprocess ends at the data worker's handoff; engine and delivery split
    at the conductor's finish stamp on RequestComplete."""
    stamps = {}

    async def run():
        transport = httpx.ASGITransport(app=entrypoint.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            pending = asyncio.create_task(client.post("/generate", data={"text": "hi"}))
            while not served.data_worker.inputs:
                await asyncio.sleep(0.005)
            stamps["ingest"] = time.perf_counter()
            await asyncio.sleep(0.05)
            stamps["finish"] = time.perf_counter()
            served.answer(served.key(), ingest=stamps["ingest"], finish=stamps["finish"])
            return await pending

    assert asyncio.run(run()).status_code == 200

    s = served.scrape()
    e2e_n, e2e = _hist(s, "mstar_e2e_request_latency_seconds", output_modalities="text")
    pre_n, pre = _hist(s, "mstar_request_preprocess_seconds", output_modalities="text")
    eng_n, eng = _hist(s, "mstar_request_engine_seconds", output_modalities="text")
    dlv_n, dlv = _hist(s, "mstar_request_delivery_seconds", output_modalities="text")
    assert e2e_n == pre_n == eng_n == dlv_n == 1
    assert eng == pytest.approx(stamps["finish"] - stamps["ingest"])
    assert pre > 0 and dlv > 0
    # The three stages do not overlap, so together they fit in the request.
    assert pre + eng + dlv <= e2e


def test_output_is_counted_per_modality_and_each_first_chunk_is_timed(served):
    pcm = b"\0\0" * 2400  # 0.1 s of 24 kHz mono 16-bit PCM
    served.server.submit_request(text="x", request_id="r", streaming=False,
                                 input_modalities=["text"], output_modalities=["text", "audio"])
    key = served.key()
    served.answer(key, chunks=[
        ResultChunk(request_id=key, modality="text", data=b"hello"),
        ResultChunk(request_id=key, modality="audio", data=pcm,
                    metadata={"sample_rate": 24000, "num_channels": 1}),
        ResultChunk(request_id=key, modality="audio", data=pcm * 2,
                    metadata={"sample_rate": 24000, "num_channels": 2}),
        ResultChunk(request_id=key, modality="text", data=b"!"),
    ])
    assert len(asyncio.run(served.server.collect_results("r"))) == 4
    # A dropped stream's chunk is output too, but only successful requests
    # are timed; an error chunk is neither.
    async def drop():
        served.server.submit_request(text="x", request_id="d", **TEXT)
        stream = served.server.iter_result_chunks("d")
        served.data_worker.chunks.put(ResultChunk(request_id=served.key(1), modality="text", data=b"a"))
        assert (await stream.__anext__()).data == b"a"
        await stream.aclose()

    asyncio.run(drop())
    served.server.submit_request(text="y", request_id="e", streaming=False, **TEXT)
    served.data_worker.chunks.put(ResultChunk(
        request_id=served.key(2), modality="error", data=b"bad image", metadata={"status": 400}))
    with pytest.raises(HTTPException):
        asyncio.run(served.server.collect_results("e"))

    s = served.scrape()
    assert s[("mstar_output_chunks_total", _labels(modality="text"))] == 3
    assert s[("mstar_output_bytes_total", _labels(modality="text"))] == len(b"hello!a")
    assert s[("mstar_output_chunks_total", _labels(modality="audio"))] == 2
    assert s[("mstar_output_bytes_total", _labels(modality="audio"))] == 3 * len(pcm)
    assert s[("mstar_output_audio_seconds_total", _labels())] == pytest.approx(0.2)
    assert not any(k[0].startswith("mstar_output") and ("modality", "error") in k[1] for k in s)
    text_n, text_first = _hist(s, "mstar_time_to_first_chunk_seconds", modality="text")
    audio_n, audio_first = _hist(s, "mstar_time_to_first_chunk_seconds", modality="audio")
    assert (text_n, audio_n) == (1, 1)
    assert 0 < text_first <= audio_first


def test_with_metrics_off_no_chunk_is_timed(tmp_path):
    data_worker, conductor = _DataWorker(), _Conductor()
    server = make_test_api_server(
        preprocess_worker=data_worker, communicator=conductor, upload_dir=tmp_path,
    )
    thread = threading.Thread(target=server._process_messages, daemon=True)
    thread.start()
    try:
        server.submit_request(text="x", request_id="r", streaming=False, **TEXT)
        key = data_worker.inputs[0].request_id
        data_worker.chunks.put(ResultChunk(request_id=key, modality="text", data=b"a"))
        _wait(lambda: server.pending_requests[key].chunks)
        assert server.pending_requests[key].first_chunk_times == {}
        assert not data_worker.reporting
    finally:
        server.running = False
        thread.join(timeout=5)


def test_the_data_worker_reports_preprocess_finish_once_asked(tmp_path):
    """The real data worker thread stamps the handoff without ``--log-stats``
    once ``report_preprocess_finish`` is called, and skips the input summary."""
    from mstar.api_server.data_worker import PreprocessWorker, PreprocessWorkerThread
    from mstar.api_server.request_types import PreprocessInput

    class _Model:
        def process_prompt(self, *args, **kwargs):
            return None

    class _TensorManager:
        def store_and_return_tensor_info(self, request_id, tensors):
            return {}

        def register_for_send(self, request_id, tensor_infos):
            pass

        def get_ready_tensors(self):
            return {}

        def has_inflight_reads(self, request_id):
            return False

        def force_cleanup_request(self, request_id):
            pass

    sent = []
    facade = PreprocessWorker.__new__(PreprocessWorker)
    facade.timing_requested = threading.Event()
    facade.profile_queue = queue.Queue()
    stop = threading.Event()
    worker = PreprocessWorkerThread(
        in_queue=queue.Queue(), result_tensor_queue=queue.Queue(), out_queue=queue.Queue(),
        profile_queue=facade.profile_queue, cleanup_request_queue=queue.Queue(),
        abort_request_queue=queue.Queue(), reads_done_queue=queue.Queue(),
        discard_tensor_queue=queue.Queue(), stop_event=stop,
        communicator=types.SimpleNamespace(get_all_new_messages=lambda: [],
                                           send=lambda entity, msg: sent.append(msg)),
        tensor_manager=_TensorManager(), model=_Model(),
        timing_requested=facade.timing_requested,
    )
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()

    def preprocess(key):
        before = time.perf_counter()
        worker.in_queue.put(PreprocessInput(
            request_id=key, client_request_id=key, text="hi", file_paths=None,
            model_kwargs={}, **TEXT))
        _wait(lambda: len(sent) == int(key[-1]))
        return before

    try:
        preprocess("k1")
        facade.report_preprocess_finish()
        before = preprocess("k2")
        updates = facade.get_profile_updates()
    finally:
        stop.set()
        thread.join(timeout=5)

    assert [u.request_id for u in updates] == ["k2"]
    assert before < updates[0].preprocess_finish_time < time.perf_counter()
    assert updates[0].inputs == []


# ── conductor status ─────────────────────────────────────────────────────────

def _status(waiting=0, running=0, draining=0, **in_flight):
    """A status message as the conductor builds it, after the wire codec."""
    states, components = Gauge(catalog.CONDUCTOR_REQUESTS), Gauge(catalog.COMPONENT_IN_FLIGHT)
    for state, n in (("waiting", waiting), ("running", running), ("draining", draining)):
        states.set((state,), n)
    for node, n in in_flight.items():
        components.set((node,), n)
    message = APIServerMessage("conductor_status", ConductorStatus(
        [states.snapshot(), components.snapshot()]))
    return wire.decode(wire.encode(message))


def _conductor_samples(samples):
    states = {dict(k[1])["state"]: v for k, v in samples.items() if k[0] == "mstar_conductor_requests"}
    in_flight = {dict(k[1])["component"]: v for k, v in samples.items()
                 if k[0] == "mstar_component_in_flight"}
    return states, in_flight


def _scrape_until(served, pred):
    """Scrape until ``pred(samples)``: the message thread applies what is queued
    on its next pass."""
    _wait(lambda: pred(served.scrape()))
    return served.scrape()


def test_the_conductor_gauges_follow_its_latest_status(served):
    before = served.scrape()
    assert _conductor_samples(before) == ({}, {})
    assert before[AGE] >= 0

    served.conductor.inbox.put(_status(waiting=2, running=1, enc=1, llm=0))
    after = _scrape_until(served, lambda s: _conductor_samples(s)[0])
    assert _conductor_samples(after) == (
        {"waiting": 2, "running": 1, "draining": 0}, {"enc": 1, "llm": 0})

    served.conductor.inbox.put(_status(running=3, draining=1, enc=2, llm=3))
    after = _scrape_until(served, lambda s: _conductor_samples(s)[0]["running"] == 3)
    assert _conductor_samples(after) == (
        {"waiting": 0, "running": 3, "draining": 1}, {"enc": 2, "llm": 3})


def test_status_age_counts_from_enabling_metrics_then_from_each_status(served):
    first = served.scrape()[AGE]
    time.sleep(0.1)
    waited = served.scrape()[AGE]
    assert waited >= first + 0.1

    served.conductor.inbox.put(_status())
    fresh = _scrape_until(served, lambda s: s[AGE] < waited)[AGE]
    time.sleep(0.1)
    assert served.scrape()[AGE] >= fresh + 0.1


def test_admission_wait_is_observed_on_a_completed_request(served):
    labels = _labels(output_modalities="text")
    family = "mstar_request_queue_seconds"

    async def run(rid, **times):
        served.server.submit_request(text="x", request_id=rid, streaming=False, **TEXT)
        served.answer(served.key(len(served.data_worker.inputs) - 1), **times)
        await served.server.collect_results(rid)

    asyncio.run(run("queued", conductor_arrival_time=10.0, conductor_ingest_time=10.25))
    # no arrival stamp: a conductor running without metrics
    asyncio.run(run("unstamped", conductor_ingest_time=3.0))

    samples = served.scrape()
    assert _count(samples, "success") == 2
    assert samples[(f"{family}_count", labels)] == 1
    assert samples[(f"{family}_sum", labels)] == 0.25
    assert samples[(f"{family}_bucket", _labels(output_modalities="text", le="0.1"))] == 0
    assert samples[(f"{family}_bucket", _labels(output_modalities="text", le="0.25"))] == 1


@pytest.mark.parametrize("metrics", [True, False], ids=["metrics-on", "metrics-off"])
def test_a_status_message_never_costs_the_messages_after_it(monkeypatch, tmp_path, metrics):
    """Before enable_metrics() (warmups) a status is dropped; either way the
    rest of its batch is still applied."""
    pytest.importorskip("prometheus_client")
    with _serve(monkeypatch, tmp_path, metrics=metrics) as served:
        served.server.timeout_seconds = 2.0

        async def run():
            served.server.submit_request(text="x", request_id="r", streaming=False, **TEXT)
            served.data_worker.chunks.put(ResultChunk(request_id=served.key(), modality="text", data=b"hi"))
            served.conductor.inbox.put([_status(running=1), _complete(served.key())])
            return await served.server.collect_results("r")

        assert [c.data for c in asyncio.run(run())] == [b"hi"]
        if metrics:
            assert _conductor_samples(served.scrape())[0]["running"] == 1


def test_a_malformed_snapshot_is_dropped_and_the_rest_still_exported(served):
    good = _status(waiting=1, enc=1).body.metrics
    served.conductor.inbox.put(APIServerMessage("conductor_status", ConductorStatus([
        good[0],
        MetricSnapshot("mstar_component_in_flight", [["enc", "extra"]], [1.0]),
        MetricSnapshot("mstar_not_in_the_catalog", [[]], [1.0]),
    ])))
    after = _scrape_until(served, lambda s: _conductor_samples(s)[0])
    assert _conductor_samples(after) == ({"waiting": 1, "running": 0, "draining": 0}, {})


def test_counter_and_histogram_snapshots_render_as_prometheus_client_would(monkeypatch):
    """The kinds workers and transports will send: rendered from a snapshot,
    they match prometheus_client's own Counter and Histogram fed the same
    values."""
    prometheus_client = pytest.importorskip("prometheus_client")
    counter = catalog.Family("mstar_test_things_total", catalog.COUNTER, "Things.", ("component",))
    histogram = catalog.Family("mstar_test_step_seconds", catalog.HISTOGRAM, "Steps.",
                               ("component",), buckets=(0.1, 1.0))
    monkeypatch.setattr(catalog, "STATUS_FAMILIES", {f.name: f for f in (counter, histogram)})
    observed = {"a": [0.1, 0.5, 5.0], "b": [0.05]}

    things, steps = Counter(counter), Histogram(histogram)
    reference = prometheus_client.CollectorRegistry()
    ref_things = prometheus_client.Counter(
        "mstar_test_things", "Things.", ["model_name", "component"], registry=reference)
    ref_steps = prometheus_client.Histogram(
        "mstar_test_step_seconds", "Steps.", ["model_name", "component"], buckets=(0.1, 1.0),
        registry=reference)
    for component, values in observed.items():
        for v in values:
            things.inc((component,))
            steps.observe((component,), v)
            ref_things.labels("dummy", component).inc()
            ref_steps.labels("dummy", component).observe(v)

    server = make_test_api_server(preprocess_worker=_DataWorker())
    server.enable_metrics()
    server.metrics.conductor_status(ConductorStatus([things.snapshot(), steps.snapshot()]))
    ours = {k: v for k, v in _parse(server.metrics.render().decode()).items()
            if k[0].startswith("mstar_test_")}
    theirs = {k: v for k, v in _parse(prometheus_client.generate_latest(reference).decode()).items()
              if not k[0].endswith("_created")}
    assert ours == theirs
    assert ours[("mstar_test_step_seconds_bucket", _labels(component="a", le="0.1"))] == 1


def test_the_rust_bridge_serves_the_conductor_families(served):
    served.conductor.inbox.put(_status(waiting=1, running=2, enc=2))
    python_route = _scrape_until(served, lambda s: _conductor_samples(s)[0])
    sent = []
    _serve_bridge(served.server, [[{"t": "metrics", "rid": "metrics-a", "sub": 1}]], sent)

    (reply,) = sent
    bridged = _parse(reply["data"].decode())
    assert _conductor_samples(bridged) == ({"waiting": 1, "running": 2, "draining": 0}, {"enc": 2})
    assert _without_age(bridged) == _without_age(python_route)


class _StopError(Exception):
    pass


def _config(monkeypatch, tmp_path, text="model: fake\n") -> str:
    """A config for a model that needs nothing to build."""
    from mstar.model import registry

    class _Model:
        PREPROCESS_TORCH_THREADS = None

        def __init__(self, **kwargs):
            pass

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(text)
    monkeypatch.setattr(registry, "get_model_class", lambda name: _Model)
    monkeypatch.setattr(entrypoint, "model_init_kwargs", lambda name: {})
    return str(cfg)


def _main_until_api_server(monkeypatch, tmp_path, *argv):
    """Run ``mstar-serve`` up to building the API server; True if it got there."""
    reached = []

    def fake_api_server(**kwargs):
        reached.append(kwargs)
        raise _StopError

    cfg = _config(monkeypatch, tmp_path)
    monkeypatch.setattr(entrypoint, "APIServer", fake_api_server)
    with pytest.raises((_StopError, SystemExit)):
        entrypoint.main(["--config", cfg, *argv])
    return bool(reached)


def test_metrics_start_after_the_warmup_requests_and_before_the_bind(monkeypatch, tmp_path):
    """Warmups compile first shapes; counting them would fill the latency
    histograms with startup time."""
    events = []

    class _Server:
        fatal_error = None

        def __init__(self, **kwargs):
            pass

        def finalize_setup(self):
            events.append("setup")

        def enable_metrics(self):
            events.append("metrics")

        def set_on_fatal(self, callback):
            pass

        def cleanup(self):
            pass

    class _Conductor:
        pid = 1

        def start(self):
            pass

    class _Uvicorn:
        started = True

        def __init__(self, config):
            pass

        def run(self):
            events.append("bind")

    cfg = _config(monkeypatch, tmp_path, "model: fake\nwarmup_requests:\n  - text: hi\n")
    # main() assigns the module's api_server; restore it for the tests after this one.
    monkeypatch.setattr(entrypoint, "api_server", entrypoint.api_server)
    monkeypatch.setattr(entrypoint, "MISSING_DEPENDENCY", None)
    monkeypatch.setattr(entrypoint, "APIServer", _Server)
    monkeypatch.setattr(entrypoint.mp, "get_context",
                        lambda method: types.SimpleNamespace(Process=lambda **kw: _Conductor()))
    monkeypatch.setattr(entrypoint, "_run_warmup_requests", lambda server, specs: events.append("warmup"))
    monkeypatch.setattr(entrypoint.uvicorn, "Server", _Uvicorn)
    monkeypatch.setattr(entrypoint, "_shutdown_conductor_process", lambda proc: None)

    entrypoint.main(["--config", cfg, "--enable-metrics"])

    assert events == ["setup", "warmup", "metrics", "bind"]


def _conductor_spawn_kwargs(monkeypatch, tmp_path, *argv) -> dict:
    """The keyword arguments ``mstar-serve`` spawns the conductor process with."""
    spawned = []

    class _Server:
        fatal_error = None

        def __init__(self, **kwargs):
            pass

        def finalize_setup(self):
            raise _StopError

        def cleanup(self):
            pass

    cfg = _config(monkeypatch, tmp_path)
    monkeypatch.setattr(entrypoint, "api_server", entrypoint.api_server)
    monkeypatch.setattr(entrypoint, "MISSING_DEPENDENCY", None)
    monkeypatch.setattr(entrypoint, "APIServer", _Server)
    monkeypatch.setattr(entrypoint.mp, "get_context", lambda method: types.SimpleNamespace(
        Process=lambda **kw: spawned.append(kw) or types.SimpleNamespace(pid=1, start=lambda: None)))
    monkeypatch.setattr(entrypoint, "_shutdown_conductor_process", lambda proc: None)
    with pytest.raises(_StopError):
        entrypoint.main(["--config", cfg, *argv])
    (kw,) = spawned
    return kw.get("kwargs", {})


def test_the_conductor_gets_a_status_interval_only_with_metrics_on(monkeypatch, tmp_path):
    assert _conductor_spawn_kwargs(monkeypatch, tmp_path, "--enable-metrics") == {"status_interval_s": 1.0}
    assert _conductor_spawn_kwargs(monkeypatch, tmp_path).get("status_interval_s") is None


def test_the_conductor_process_hands_the_status_interval_to_the_conductor(monkeypatch, tmp_path):
    import mstar.conductor.conductor as conductor_mod

    built = []

    class _Conductor:
        def __init__(self, **kwargs):
            built.append(kwargs)

        def run(self):
            pass

        def shutdown(self):
            pass

    cfg = _config(monkeypatch, tmp_path)
    monkeypatch.setattr(conductor_mod, "Conductor", _Conductor)
    monkeypatch.setattr(entrypoint, "watch_parent", lambda *a, **k: None)
    monkeypatch.setattr(entrypoint.logging, "basicConfig", lambda **k: None)
    monkeypatch.setattr(entrypoint.signal, "signal", lambda *a: None)
    entrypoint._conductor_process_target("fake", cfg, str(tmp_path), status_interval_s=0.5)
    assert [b["status_interval_s"] for b in built] == [0.5]


def test_enable_metrics_without_the_package_stops_before_anything_starts(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(entrypoint, "MISSING_DEPENDENCY", ModuleNotFoundError("prometheus_client"))
    assert not _main_until_api_server(monkeypatch, tmp_path, "--enable-metrics")
    assert "mstar[metrics]" in capsys.readouterr().err
    assert _main_until_api_server(monkeypatch, tmp_path)


def test_enable_metrics_with_the_rust_frontend_starts(monkeypatch, tmp_path):
    monkeypatch.setattr(entrypoint, "MISSING_DEPENDENCY", None)
    assert _main_until_api_server(monkeypatch, tmp_path, "--enable-metrics", "--rust-frontend")


def test_mstar_serve_forwards_enable_metrics(monkeypatch):
    from mstar.cli import main as cli

    seen = []
    monkeypatch.setitem(sys.modules, "mstar.api_server.entrypoint", types.SimpleNamespace(main=seen.append))
    cli.main(["serve", "kokoro", "--enable-metrics"])
    cli.main(["serve", "kokoro"])
    assert "--enable-metrics" in seen[0]
    assert "--enable-metrics" not in seen[1]


def test_a_missing_package_is_reported_and_a_broken_one_raises(monkeypatch):
    pytest.importorskip("prometheus_client")
    try:
        with monkeypatch.context() as m:
            m.setitem(sys.modules, "prometheus_client", None)
            importlib.reload(metrics_module)
            assert metrics_module.MISSING_DEPENDENCY is not None
            with pytest.raises(RuntimeError, match=r"mstar\[metrics\]"):
                metrics_module.ServingMetrics("dummy", in_flight=int)
        with monkeypatch.context() as m:
            m.setitem(sys.modules, "prometheus_client.core", None)
            with pytest.raises(ModuleNotFoundError) as e:
                importlib.reload(metrics_module)
            assert e.value.name == "prometheus_client.core"
    finally:
        importlib.reload(metrics_module)
    assert metrics_module.MISSING_DEPENDENCY is None
