"""Pass closure through real worker admission, Rust completion and wire frames."""

from collections import deque
from types import SimpleNamespace

import pytest

pytest.importorskip("mstar_rust")

from rust_runtime import Inboxes, build_runtime

from mstar.conductor.conductor import Conductor, RequestData
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    CurrentForwardPassInfo,
    PartitionState,
)
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge, GraphNode
from mstar.graph.runtime.base import PostprocessNeed, PostprocessRequest, RouteInput, RouteTensors, SendInput
from mstar.graph.special_destinations import EMPTY_DESTINATION
from mstar.model.base import ForwardPassArgs, WorkerGraph
from mstar.utils.containers import ParallelList, RecentSet
from mstar.utils.ipc_format import (
    InputSignals,
    NewRequest,
    ReadsDone,
    WorkerGraphsDone,
    WorkerMessageType,
)
from mstar.worker.node_manager_utils import RequestStateManager
from mstar.worker.worker import Worker


def _info(epoch=0, request="r", partition="default", walk="run"):
    info = CurrentForwardPassInfo(
        request_id=request, graph_walk=walk, fwd_index=0,
        random_seed=0, max_tokens=8, partition_name=partition,
    )
    # Also usable against the unmodified PR2 as a negative control.
    info.input_epoch = epoch
    return info


class _Case:
    def __init__(self, path, count=1, epoch=0, partitions=None):
        self.boxes = Inboxes("w0", path)
        partitions = partitions or dict.fromkeys(range(count), "default")
        graphs = [
            WorkerGraph(
                section=GraphNode(
                    name=f"n{i}", input_names={"input"},
                    outputs=[GraphEdge(name="out", next_node=EMPTY_DESTINATION)],
                ),
                graph_walks={"run"}, ranks=[0], worker_graph_id=i,
            )
            for i in range(count)
        ]
        self.rt = build_runtime(
            inboxes=self.boxes, my_worker_id="w0", my_worker_graphs=graphs,
            all_wg_ids_to_graph_walks={i: {"run"} for i in range(count)},
            all_wg_ids_to_dyn_loops={i: set() for i in range(count)},
            all_wg_ids_to_nodes={i: {f"n{i}"} for i in range(count)},
            node_to_partition={f"n{i}": partitions[i] for i in range(count)},
            sharding_config=ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}),
        )
        w = self.worker = Worker.__new__(Worker)
        w.worker_id = "w0"
        w._graph_runtime = self.rt
        w.request_state = RequestStateManager()
        w._reads_done_sent = set()
        w._draining_rids = set()
        w._removed_rids = RecentSet(16)
        w._unprocessed_messages = {}
        w.enable_nvtx = False
        w.scheduler = SimpleNamespace(note_arrival=lambda rid: None, failed_rids=set())
        w.engine_manager = SimpleNamespace(add_request=lambda *a: None)
        w._skip_unshared_publish = lambda *a: None
        w.streams = SimpleNamespace(add_request=lambda *a: None)
        w.wakeup_event = SimpleNamespace(register_futures=lambda futures: None)
        self.reads, self.unread = [], []
        w.tensor_manager = SimpleNamespace(
            register_request=lambda *a: None,
            start_read_tensors=lambda rid, edges, **kw: self.reads.extend(edges) or [],
            ack_unread_tensors=self.unread.extend,
        )
        ids = [i for i in range(count) if partitions[i] == "default"]
        self.new = NewRequest(
            request_id="r", partition_worker_graph_ids=ids,
            worker_graph_to_workers={i: ["w0"] for i in range(count)},
            initial_inputs=[GraphEdge(name="input", next_node=f"n{i}") for i in ids],
            request_info=_info(epoch), loop_caps={},
        )
        w._add_new_request(self.new)
        self.rid = self.rt.get_rid_handle("r")
        self.reads.clear()

    def finish(self, node=0, *, other_work=(), reads=False, native=False, reported_epoch=None):
        assert self.rt.pop_rids(f"n{node}", "run", [self.rid]) is not None
        if native:
            step = self.rt.begin_postprocess(PostprocessRequest(
                partition="default", graph_walk="run", node_name=f"n{node}",
                output_signals=["out"], wg_ids=ParallelList([self.rid], [node]),
            ))
            assert step.need is PostprocessNeed.STORE
            step = self.rt.resume_postprocess(stored=RouteTensors([], [0]))
            assert step.need is PostprocessNeed.SEND
            route = step.route_output
        else:
            route = self.rt.complete_and_route_batch(RouteInput(
                partition="default", graph_walk="run", node_name=f"n{node}",
                output_signals=["out"], wg_ids=ParallelList([self.rid], [node]),
                tensors=[], num_tensors=[0],
            ))
        info = self.worker.request_state.get_fwd_info(self.rid, "default")
        kwargs = {}
        # Keep the reproduction runnable against PR2, where the certificate
        # does not exist; its real report leaves admission open.
        if "input_closure_rids" in SendInput._fields:
            kwargs["input_closure_rids"] = self.worker._input_closure_candidates(
                [self.rid], [info], 0, other_work,
            )
            if reported_epoch is not None:
                kwargs["input_epochs"] = [reported_epoch]
        send = SendInput(
            completion_id=route.completion_id,
            per_request_info=ParallelList([self.rid], [info]),
            new_token_counts=ParallelList([], []),
            inflight_read_rids=frozenset({self.rid}) if reads else frozenset(),
            **kwargs,
        )
        if native:
            assert self.rt.resume_postprocess(send=send).need is PostprocessNeed.DONE
        else:
            self.rt.send_outputs(send)
        [report] = [msg.body for peer, msg in self.boxes.take() if peer == "conductor"]
        return report

    def input(self, epoch, *, node=0, tensor=False, partition="default", walk="run"):
        edge = GraphEdge(
            name="input", next_node=f"n{node}",
            tensor_info=[object()] if tensor else [],
        )
        self.worker._process_new_inputs(InputSignals(
            request_id="r", inputs=[edge], request_info=_info(epoch, partition=partition, walk=walk),
            partition_name=partition,
        ))
        return edge


@pytest.mark.parametrize("epoch", [0, -1])
@pytest.mark.parametrize("native", [False, True], ids=["split", "native"])
def test_late_descriptor_is_rejected_before_transport_read(tmp_path, epoch, native):
    case = _Case(tmp_path)
    report = case.finish(native=native)
    assert not report.has_inflight_reads
    edge = case.input(epoch, tensor=True)
    assert case.reads == [], "a late descriptor started a read after the clear completion report"
    assert case.unread == [edge]
    assert report.closed_input_epoch == 0


def test_new_epoch_reopens_same_walk_and_old_epoch_stays_closed(tmp_path):
    case = _Case(tmp_path)
    case.finish()
    case.input(1)
    assert len(case.reads) == 1
    case.reads.clear()
    report = case.finish()
    assert report.closed_input_epoch == 1
    case.input(0, tensor=True)
    case.input(1, tensor=True)
    assert not case.reads
    assert len(case.unread) == 2


def test_already_read_edge_cannot_restart_a_sealed_graph(tmp_path):
    case = _Case(tmp_path)
    case.finish()
    w = case.worker
    w.wakeup_event.drain = lambda: None
    w.tensor_manager.take_failed_reads = lambda: []
    w.tensor_manager.get_ready_tensors = lambda: {
        case.rid: [GraphEdge(name="input", next_node="n0")],
    }
    w._check_ready_tensors()
    assert case.rt.pop_rids("n0", "run", [case.rid], check_ready=True) is None
    case.input(1)
    assert case.finish().closed_input_epoch == 1


def test_partitions_sharing_a_walk_keep_independent_closure(tmp_path):
    case = _Case(tmp_path, count=2, partitions={0: "default", 1: "other"})
    case.worker._add_new_request(NewRequest(
        request_id="r", partition_worker_graph_ids=[1],
        worker_graph_to_workers={0: ["w0"], 1: ["w0"]},
        initial_inputs=[GraphEdge(name="input", next_node="n1")],
        request_info=_info(partition="other"), loop_caps={},
    ))
    assert case.finish().closed_input_epoch == 0
    case.input(1)
    # Reopening default must not add other's unfinished graph to its pending
    # set merely because both partitions use the same walk name.
    assert case.finish().closed_input_epoch == 1
    assert case.rt.pop_rids("n1", "run", [case.rid], check_ready=True) is not None


def test_all_local_graphs_must_complete_before_sealing(tmp_path):
    case = _Case(tmp_path, count=2)
    first = case.finish(0)
    assert first.closed_input_epoch == -1
    last = case.finish(1)
    assert last.closed_input_epoch == 0
    case.input(0, tensor=True)
    assert not case.reads


def test_queued_work_in_an_already_completed_graph_prevents_sealing(tmp_path):
    case = _Case(tmp_path, count=2)
    case.finish(0)
    # A late input arrived while the other local graph was still running.
    # Its read is complete, but its queued work could itself start more reads.
    case.input(0, node=0)
    assert case.finish(1).closed_input_epoch == -1
    assert case.finish(0).closed_input_epoch == 0


def test_new_epoch_cannot_discard_unfinished_local_graph(tmp_path):
    case = _Case(tmp_path, count=2)
    case.finish(0)
    case.input(1, tensor=True)
    assert not case.reads
    assert case.finish(1).closed_input_epoch == 0


def test_retiring_old_step_cannot_certify_newer_epoch(tmp_path):
    case = _Case(tmp_path)
    assert case.finish(other_work={case.rid}).closed_input_epoch == -1
    case.input(1)
    # An old speculative step retiring after new metadata arrived must never
    # certify the newer epoch, even if its graph and walk names are identical.
    report = case.finish(reported_epoch=0)
    assert report.closed_input_epoch == -1
    case.input(1)
    assert case.finish().closed_input_epoch == 1


@pytest.mark.parametrize("kind", ["next_step", "other_lane", "side_lane"])
def test_outstanding_work_requires_barrier(tmp_path, kind):
    case = _Case(tmp_path)
    other_work = ()
    if kind == "next_step":
        other_work = {case.rid}
    elif kind == "other_lane":
        case.worker._rows_by_lane = {0: frozenset({case.rid}), 1: frozenset({case.rid})}
    else:
        case.worker._side_steps = object()
    report = case.finish(other_work=other_work)
    assert report.closed_input_epoch == -1


@pytest.mark.parametrize("epoch", [0, -1, 1])
def test_duplicate_new_request_does_not_reopen_sealed_pass(tmp_path, epoch):
    case = _Case(tmp_path)
    case.finish()
    case.new.request_info = _info(epoch)
    case.worker._add_new_request(case.new)
    case.input(0, tensor=True)
    assert not case.reads


def test_recycled_handle_does_not_inherit_seal(tmp_path):
    case = _Case(tmp_path)
    case.finish()
    case.rt.remove_request(case.rid)
    case.worker.request_state.remove_request(case.rid)
    case.new.request_id = "second"
    case.new.request_info = _info(request="second")
    case.worker._add_new_request(case.new)
    assert case.rt.get_rid_handle("second") == case.rid
    assert len(case.reads) == 1


def test_unsupported_request_keeps_admission_open_for_drain(tmp_path):
    case = _Case(tmp_path, epoch=-1)
    assert case.finish().closed_input_epoch == -1
    case.input(-1, tensor=True)
    assert len(case.reads) == 1


def _conductor():
    c = Conductor.__new__(Conductor)
    c.sent = []
    c.communicator = SimpleNamespace(send=lambda peer, msg: c.sent.append((peer, msg)))
    p = PartitionState(
        partition_name="default", metadata=CurrentForwardConductorMetadata("run", False),
        current_worker_graph_ids={0}, input_workers={"w0", "w1"},
    )
    c.requests = {"r": RequestData(
        persist_signals={}, persist_signal_ref_cnt={},
        worker_graph_to_workers={0: ["w0"], 1: ["w1"]}, all_worker_graph_ids={0, 1},
        max_output_tokens=8, random_seed=0, resource_configs={},
        input_closure_enabled=True, partition_states={"default": p},
    )}
    c._client_streams = {}
    c.producer_triggered_partitions = set()
    c.enable_prof = False
    c.draining = {}
    c._early_reads_done = {}
    c._early_reads_done_deadlines = deque()
    c._draining_deadlines = deque()
    c._drain_ttl_s = 120
    return c


@pytest.mark.parametrize("missing", ["none", "worker", "old_epoch", "positive_read", "unsupported"])
def test_conductor_requires_every_participants_current_seal(missing):
    c = _conductor()
    if missing == "unsupported":
        c.requests["r"].input_closure_enabled = False
    for index, worker in enumerate(("w0", "w1")):
        if missing == "worker" and index == 1:
            continue
        c._process_worker_graphs_done(WorkerGraphsDone(
            request_id="r", worker_graph_ids=[index], is_first_tp_rank=True,
            worker_id=worker,
            closed_input_epoch=-1 if missing == "old_epoch" and index == 1 else 0,
            has_inflight_reads=missing == "positive_read" and index == 1,
        ))
    c._process_request_done("r")
    drains = {peer for peer, msg in c.sent if msg.message_type == WorkerMessageType.DRAIN_REQUEST}
    assert drains == (set() if missing == "none" else {"w0", "w1"})
    assert c.draining["r"].expected_acks == drains | {c.PREPROCESS_WORKER}
    # Even a sealed compute side still waits for output delivery.
    assert not any(msg.message_type == WorkerMessageType.REMOVE_REQUEST for _, msg in c.sent)


def test_preprocess_ack_cannot_bypass_missing_worker_seal():
    c = _conductor()
    c._handle_reads_done(ReadsDone(request_id="r", entity_id=c.PREPROCESS_WORKER))
    c._process_request_done("r")
    assert c.draining["r"].expected_acks == {"w0", "w1"}
    assert "r" in c.requests


def test_next_conductor_pass_invalidates_only_participating_workers_seals():
    c = _conductor()
    c._stream_fed_partitions = set()
    c.worker_graphs = {
        0: SimpleNamespace(graph_walks={"run", "next"}),
        1: SimpleNamespace(graph_walks={"run"}),
    }
    c.model = SimpleNamespace(get_partition_forward_pass_args=lambda **kw: ForwardPassArgs(
        full_metadata=CurrentForwardConductorMetadata("next", False),
        inputs=[GraphEdge(name="input", next_node="n0")], unpersist_tensors=[],
    ))
    p = c.requests["r"].partition_states["default"]
    p.input_closed_workers = {"w0", "w1"}
    sent = []
    c._send_partition_inputs = lambda *a: sent.append((p.input_epoch, p.fwd_pass_number))
    assert not c._process_done_forward("r", "default")
    assert sent == [(1, 0)], "the closure epoch must advance before the next inputs are sent"
    assert p.input_closed_workers == {"w1"}
    c._process_worker_graphs_done(WorkerGraphsDone(
        request_id="r", worker_graph_ids=[0], is_first_tp_rank=True,
        worker_id="w0", closed_input_epoch=0,
    ))
    assert p.input_closed_workers == {"w1"}, "an old seal must not certify the new pass"
