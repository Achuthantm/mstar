"""Prometheus metrics at ``/metrics`` (``--enable-metrics``).

Requests go through the FastAPI app and the API server's real message thread;
only the data worker and the conductor are stand-ins, fed the messages the
real ones send. Each test scrapes ``/metrics`` and reads the samples back.
"""

from __future__ import annotations

import asyncio
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

from mstar.api_server import entrypoint
from mstar.api_server.request_types import (
    APIServerMessage,
    DataWorkerProfile,
    RequestComplete,
    RequestFailed,
    ResultChunk,
)
from mstar.observability import metrics as metrics_module

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
        out = []
        while not self.inbox.empty():
            out.append(self.inbox.get())
        return out

    def send(self, entity, msg):
        pass


class _Served:
    def __init__(self, server, data_worker, conductor):
        self.server, self.data_worker, self.conductor = server, data_worker, conductor

    def key(self, i=0):
        return self.data_worker.inputs[i].request_id

    def answer(self, key, data=b"hi", chunks=None, ingest=None, finish=None):
        """The engine's result chunks (one text chunk unless given), then its
        completion, as the real data worker and conductor deliver them.
        ``ingest`` and ``finish`` are the conductor's perf_counter stamps."""
        chunks = chunks or [ResultChunk(request_id=key, modality="text", data=data)]
        for chunk in chunks:
            self.data_worker.chunks.put(chunk)
        finish = time.perf_counter() if finish is None else finish
        self.conductor.inbox.put(APIServerMessage("request_complete", RequestComplete(
            request_id=key, output_chunk_counts={"out": len(chunks)},
            conductor_ingest_time=finish if ingest is None else ingest,
            conductor_finish_time=finish,
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


def _parse(text: str) -> dict:
    from prometheus_client.parser import text_string_to_metric_families

    samples = {}
    for family in text_string_to_metric_families(text):
        for s in family.samples:
            samples[(s.name, tuple(sorted(s.labels.items())))] = s.value
    return samples


@pytest.fixture
def served(monkeypatch, tmp_path):
    pytest.importorskip("prometheus_client")
    data_worker, conductor = _DataWorker(), _Conductor()
    server = make_test_api_server(
        preprocess_worker=data_worker, communicator=conductor, upload_dir=tmp_path,
    )
    server.enable_metrics()
    monkeypatch.setattr(entrypoint, "api_server", server)
    thread = threading.Thread(target=server._process_messages, daemon=True)
    thread.start()
    yield _Served(server, data_worker, conductor)
    server.running = False
    thread.join(timeout=5)


def _labels(**kv):
    return tuple(sorted({"model_name": "dummy", **kv}.items()))


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
    assert replies[3]["data"] == TestClient(entrypoint.app).get("/metrics").content


def test_the_rust_bridge_answers_metrics_off_when_they_are_off(tmp_path):
    """Empty metadata is how ``mstar-server`` knows to answer 404."""
    sent = []
    _serve_bridge(make_test_api_server(upload_dir=tmp_path),
                  [[{"t": "metrics", "rid": "metrics-a", "sub": 5}]], sent)
    assert sent == [{"t": "metrics", "rid": "metrics-a", "sub": 5, "data": b"", "metadata": {}}]


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
