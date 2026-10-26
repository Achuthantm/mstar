"""Deterministic CPU reproduction of the extra-read completion race.

Only the native transfer engine is substituted: the async reader, transport
buffers, conductor completion barrier, worker drain and Worker._remove_request
are real. The extra read models an erroneous output still being read at
completion; this does not establish that a particular production model emits
such an output.
"""

import ctypes
from collections import deque
from threading import Event
from types import SimpleNamespace

import pytest
import torch

from mstar.communication import tensors
from mstar.communication.communicator import CommProtocol
from mstar.conductor.conductor import Conductor, RequestData
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge
from mstar.utils.containers import RecentSet
from mstar.utils.ipc_format import ConductorMessageType, ReadsDone, WorkerMessageType
from mstar.worker.rid_table import RidTable
from mstar.worker.worker import Worker


class _CheckedCopyEngine(tensors.LocalTransferEngine):
    """Copy registered host memory, rejecting a read after either endpoint frees.

    Checking before memmove makes the pre-fix failure safe to reproduce rather
    than dereferencing freed pointers. Events select the ordering without sleeps.
    """

    def __init__(self, *, peers, **kwargs):
        super().__init__(f"cpu:{len(peers)}")
        peers[self.get_session_id()] = self
        self.peers = peers
        self._engine = self
        self.regions = {}
        self.started = Event()
        self.resume = Event()
        self.invalid_endpoints = []
        self.copied = None

    def register_memory(self, ptr, nbytes):
        self.regions[ptr] = nbytes
        return 0

    def unregister_memory(self, ptr):
        self.regions.pop(ptr)
        return 0

    def batch_transfer_sync_read(self, session, local_ptrs, remote_ptrs, lengths):
        self.started.set()
        if not self.resume.wait(10):
            raise TimeoutError("test did not release the read")
        peer = self.peers[session]
        for local, remote, nbytes in zip(local_ptrs, remote_ptrs, lengths, strict=True):
            if peer.regions.get(remote, 0) < nbytes:
                self.invalid_endpoints.append("source")
            if self.regions.get(local, 0) < nbytes:
                self.invalid_endpoints.append("destination")
        if self.invalid_endpoints:
            return -1
        for local, remote, nbytes in zip(local_ptrs, remote_ptrs, lengths, strict=True):
            ctypes.memmove(local, remote, nbytes)
        self.copied = ctypes.string_at(local_ptrs[0], lengths[0])
        return 0


class _Recorder:
    def __init__(self):
        self.sent = []

    def send(self, entity_id, msg):
        self.sent.append((entity_id, msg))


def _worker(entity_id, manager, rid_table):
    w = Worker.__new__(Worker)
    w.worker_id = entity_id
    w.is_tp_follower = False
    w._in_flight_rids = set()
    w._removes_awaiting_step = {}
    w._pending_removes = {}
    w._draining_rids = set()
    w._pending_drains = set()
    w._reads_done_sent = set()
    w._removed_rids = RecentSet(16)
    w._unprocessed_messages = {}
    w._stream_buffer_rids = set()
    w._last_active = {}
    w.streaming_buffers = {}
    w.communicator = manager.communicator
    w.tensor_manager = manager
    w._graph_runtime = SimpleNamespace(
        get_rid_handle=rid_table.handle, get_rid_string=rid_table.name,
        get_sharding_config=lambda handle: SimpleNamespace(groups=[]),
        remove_request=rid_table.release,
    )
    w.engine_manager = SimpleNamespace(remove_request=lambda handle: None, evictable_nodes=lambda: [])
    w.request_state = SimpleNamespace(remove_request=lambda handle: None)
    w.profile_info = SimpleNamespace(pop_request=lambda handle: None)
    w.scheduler = SimpleNamespace(
        clear_rid=lambda handle, wire_id: None, fail_rids=lambda handles: None,
        pending_tp_follow_count={},
    )
    return w


def _conductor():
    c = Conductor.__new__(Conductor)
    c.communicator = _Recorder()
    c.requests = {"request": RequestData(
        persist_signals={}, persist_signal_ref_cnt={},
        worker_graph_to_workers={0: ["w0", "w1"]}, all_worker_graph_ids={0},
        max_output_tokens=1, random_seed=0, resource_configs={},
    )}
    c.draining = {}
    c._early_reads_done = {}
    c._drain_ttl_s = 120
    c._draining_deadlines = deque()
    c._try_admit_waiting = lambda: None
    return c


@pytest.mark.parametrize("teardown_order", [("w0", "w1"), ("w1", "w0")])
@pytest.mark.parametrize("read_before_completion", [False, True], ids=["stray-read", "normal-read"])
def test_successful_cleanup_does_not_invalidate_a_read(monkeypatch, teardown_order, read_before_completion):
    peers = {}
    monkeypatch.setattr(tensors, "MooncakeTransferEngine", lambda **kw: _CheckedCopyEngine(peers=peers, **kw))
    managers, workers, tables = {}, {}, {}
    for entity_id in ("w0", "w1"):
        table = tables[entity_id] = RidTable()
        handle = table.intern("request")
        manager = managers[entity_id] = tensors.MooncakeCommunicationManager(
            my_entity_id=entity_id, hostname="localhost", device="cpu",
            communicator=_Recorder(), protocol=CommProtocol.TCP,
        )
        manager.rid_to_str = table.name
        config = ShardingConfig(tp_enabled_nodes=set(), groups=[], shard_dim={})
        config.setup({})
        manager.register_request(handle, config)
        workers[entity_id] = _worker(entity_id, manager, table)

    producer, receiver = managers["w0"], managers["w1"]
    payload = torch.arange(16, dtype=torch.float32)
    [info] = producer.store_and_return_tensor_info(0, {"extra": [payload]})["extra"]
    producer.increment_ref(info.uuid)
    producer.register_for_send(0, [info])
    engine = receiver.transfer_engine
    [future] = receiver.start_read_tensors(0, [GraphEdge(
        name="extra", next_node="already_finished_node", tensor_info=[info],
        _total_fanin=1, _shard_dim=None,
    )])
    try:
        assert engine.started.wait(5), "background read did not start"
        assert not future.done()
        if read_before_completion:
            engine.resume.set()
            future.result(timeout=5)
            assert receiver.get_ready_tensors()
            for _, ack in receiver.communicator.sent:
                workers["w0"]._handle_tensor_received(ack.body)
            receiver.communicator.sent.clear()

        c = _conductor()
        c._process_request_done("request")
        c._handle_reads_done(ReadsDone(request_id="request", entity_id=c.PREPROCESS_WORKER))
        drains = {
            entity: msg.body for entity, msg in c.communicator.sent
            if msg.message_type == WorkerMessageType.DRAIN_REQUEST
        }
        for entity in teardown_order:
            if entity in drains:
                workers[entity]._drain_request(drains[entity])

        def deliver_reads_done():
            for manager in managers.values():
                for _, msg in manager.communicator.sent:
                    if msg.message_type == ConductorMessageType.READS_DONE:
                        c._handle_reads_done(msg.body)
                manager.communicator.sent = [
                    (e, m) for e, m in manager.communicator.sent
                    if m.message_type != ConductorMessageType.READS_DONE
                ]

        deliver_reads_done()
        if "request" in c.draining:
            # The receiver holds its READS_DONE, and so the removal, until its
            # in-flight read has finished.
            assert not read_before_completion
            assert c.draining["request"].expected_acks == {"w1"}
            engine.resume.set()
            future.result(timeout=5)
            assert receiver.get_ready_tensors()
            workers["w1"]._apply_pending_drains(set())
            deliver_reads_done()

        removals = {
            entity: msg.body for entity, msg in c.communicator.sent
            if msg.message_type == WorkerMessageType.REMOVE_REQUEST
        }
        assert set(removals) == {"w0", "w1", c.PREPROCESS_WORKER}
        assert "request" not in c.requests
        for entity in teardown_order:
            workers[entity]._remove_request(removals[entity])
            assert tables[entity].handle("request") is None

        engine.resume.set()
        try:
            future.result(timeout=5)
        except RuntimeError as exc:
            pytest.fail(f"Race reproduced: cleanup invalidated {engine.invalid_endpoints} before read finished: {exc}")
        assert engine.copied == payload.numpy().tobytes()
        assert engine.invalid_endpoints == []
        assert receiver.get_ready_tensors() == {}
        assert not producer.tensor_store.check_uuid_presence(info.uuid)
        assert not receiver.tensor_store.check_uuid_presence(info.uuid)
    finally:
        engine.resume.set()
        for manager in managers.values():
            # A failed future must not prevent the pool from being joined.
            manager._async_reader._executor.shutdown(wait=True)
