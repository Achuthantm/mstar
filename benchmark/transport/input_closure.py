"""CPU admission/completion/teardown cost over real Rust graphs and ZMQ.

One worker process per graph, with signal-only inputs and no model execution,
engine resources, GPU scheduling or tensor payloads. Each request runs a chosen
number of conductor-driven passes. The API output acknowledgement is local.
Compare the same script under each checkout's PYTHONPATH and freshly built
extension; --force-drain measures the unconditional barrier on the same base.
"""

import argparse
import hashlib
import inspect
import json
import logging
import multiprocessing as mp
import os
import platform
import statistics
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import mstar_rust

from mstar.communication.communicator import make_communicator
from mstar.communication.tensor_store import RustTensorBookkeeping, TensorStore
from mstar.communication.tensors import SharedMemoryCommunicationManager
from mstar.conductor.conductor import Conductor, RequestData
from mstar.conductor.request_info import CurrentForwardConductorMetadata, CurrentForwardPassInfo, PartitionState
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge, GraphNode, NodeAndGraphWalk
from mstar.graph.runtime.base import PostprocessNeed, PostprocessRequest, RouteTensors, SendInput
from mstar.graph.runtime.rust import RustGraphRuntime
from mstar.graph.special_destinations import EMPTY_DESTINATION
from mstar.model.base import ForwardPassArgs, WorkerGraph
from mstar.utils.containers import RecentSet
from mstar.utils.ipc_format import ConductorMessageType, NewRequest, ReadsDone, WorkerMessage, WorkerMessageType
from mstar.worker.node_manager_utils import RequestStateManager
from mstar.worker.stream_buffers import StreamBufferManager
from mstar.worker.worker import Worker

_CLOSURE = "input_closure_rids" in SendInput._fields


def _graphs(workers):
    return [
        WorkerGraph(
            section=GraphNode(
                name=f"n{i}", input_names={"input"},
                outputs=[GraphEdge(name="out", next_node=EMPTY_DESTINATION)],
            ),
            graph_walks={"run"}, ranks=[i], worker_graph_id=i,
        )
        for i in range(len(workers))
    ]


def _receive(comm):
    messages = comm.get_all_new_messages(blocking=True, timeout_s=30)
    if not messages:
        raise TimeoutError("input-closure benchmark peer did not respond")
    return messages


def _worker(index, workers, prefix, pending_step):
    logging.disable(logging.INFO)
    worker_id, node = workers[index], f"n{index}"
    comm = make_communicator(worker_id, ["conductor"], ipc_socket_path_prefix=prefix)
    graphs = _graphs(workers)
    book = RustTensorBookkeeping()
    rt = RustGraphRuntime(
        my_worker_id=worker_id, my_worker_graphs=[graphs[index]],
        all_wg_ids_to_graph_walks={i: {"run"} for i in range(len(workers))},
        all_wg_ids_to_dyn_loops={i: set() for i in range(len(workers))},
        all_wg_ids_to_nodes={i: {f"n{i}"} for i in range(len(workers))},
        node_to_partition={f"n{i}": "default" for i in range(len(workers))},
        sharding_config=ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}),
        bookkeeping=book, communicator=comm,
    )
    w = Worker.__new__(Worker)
    w.worker_id, w.communicator, w._graph_runtime = worker_id, comm, rt
    w.enable_nvtx = w.enable_prof = w.is_tp_follower = False
    w._in_flight_rids, w._draining_rids, w._pending_drains, w._reads_done_sent = set(), set(), set(), set()
    w._pending_removes, w._removes_awaiting_step, w._unprocessed_messages = {}, {}, {}
    w._removed_rids = RecentSet(4096)
    w.streaming_buffers = {}
    w._all_wg_nodes = {}
    failed = set()
    w.scheduler = SimpleNamespace(
        note_arrival=lambda rid: None, failed_rids=failed, fail_rids=failed.update,
        clear_rid=lambda rid, wire: failed.discard(rid), pending_tp_follow_count={},
    )
    w.engine_manager = SimpleNamespace(
        engine=SimpleNamespace(publisher_nodes={}),
        add_request=lambda *a: None, remove_request=lambda rid: None,
    )
    w.request_state = RequestStateManager()
    w.profile_info = SimpleNamespace(pop_request=lambda rid: None)
    w.wakeup_event = SimpleNamespace(register_futures=lambda futures: None)
    w.tensor_manager = SharedMemoryCommunicationManager(
        my_entity_id=worker_id, hostname="localhost", device="cpu", communicator=comm, shm_dir=prefix,
    )
    w.tensor_manager.tensor_store = TensorStore(bookkeeping=book)
    w.tensor_manager.rid_to_str = rt.get_rid_string
    w.streams = StreamBufferManager(w.request_state, w.tensor_manager, [])
    remaining = reports = closed = 0
    started = 0.0
    while True:
        ready = []
        for msg in _receive(comm):
            if isinstance(msg, dict):
                if msg["op"] == "stop":
                    return
                remaining, reports, closed = len(msg["rids"]), 0, 0
                comm.send("conductor", {"op": "ready"})
                started = time.process_time()
                continue
            body = msg.body
            if msg.message_type == WorkerMessageType.NEW_REQUEST:
                w._add_new_request(body)
                ready.append(rt.get_rid_handle(body.request_id))
            elif msg.message_type == WorkerMessageType.INPUT_SIGNALS:
                w._process_new_inputs(body)
                ready.append(rt.get_rid_handle(body.request_id))
            elif msg.message_type == WorkerMessageType.DRAIN_REQUEST:
                w._drain_request(body)
            elif msg.message_type == WorkerMessageType.REMOVE_REQUEST:
                w._remove_request(body)
                remaining -= 1
                if remaining == 0:
                    cpu_us = (time.process_time() - started) * 1e6
                    assert not failed and not w.request_state.per_request_info and not w._draining_rids
                    comm.send("conductor", {"op": "removed", "cpu_us": cpu_us, "reports": reports, "closed": closed})
            else:
                raise AssertionError(msg)
        if ready:
            rows = list(dict.fromkeys(ready))
            popped = rt.pop_rids(node, "run", rows, check_ready=True)
            assert popped is not None and len(popped.wg_ids.keys) == len(rows)
            step = rt.begin_postprocess(PostprocessRequest(
                partition="default", node_name=node, graph_walk="run", output_signals=["out"],
                wg_ids=popped.wg_ids,
            ))
            assert step.need is PostprocessNeed.STORE
            step = rt.resume_postprocess(stored=RouteTensors([], [0] * len(rows)))
            assert step.need is PostprocessNeed.SEND
            infos = [w.request_state.get_fwd_info(rid, "default") for rid in rows]
            extra = {}
            if _CLOSURE:
                extra["input_closure_rids"] = w._input_closure_candidates(
                    rows, infos, 0, rows if pending_step else (),
                )
                extra["input_epochs"] = [info.input_epoch for info in infos]
                closed += len(extra["input_closure_rids"])
            send = SendInput(
                completion_id=step.route_output.completion_id,
                per_request_info=type(popped.wg_ids)(rows, infos),
                new_token_counts=type(popped.wg_ids)([], []),
                inflight_read_rids=w.tensor_manager.inflight_read_rids(rows),
                **extra,
            )
            done = rt.resume_postprocess(send=send)
            assert done.need is PostprocessNeed.DONE and set(done.completed_rids) == set(rows)
            reports += len(rows)
        w._apply_pending_drains(set())


class _Model:
    def __init__(self, passes, workers):
        self.passes = passes
        self.inputs = [GraphEdge(name="input", next_node=f"n{i}") for i in range(len(workers))]

    def get_partition_forward_pass_args(self, *, partition_metadata, **kwargs):
        completed = partition_metadata.kwargs.get("completed", 0) + 1
        done = completed >= self.passes
        return ForwardPassArgs(
            full_metadata=CurrentForwardConductorMetadata("run", False, kwargs={"completed": completed}),
            inputs=[] if done else self.inputs, unpersist_tensors=[], request_done=done,
        )


class _ApiLoopback:
    def __init__(self, comm):
        self.comm, self.completed, self.drains = comm, [], 0

    def send(self, entity_id, msg):
        if entity_id == "api_server":
            assert msg.message_type == "request_complete"
            self.completed.append(msg.body.request_id)
        elif entity_id != Conductor.PREPROCESS_WORKER:
            self.drains += msg.message_type == WorkerMessageType.DRAIN_REQUEST
            self.comm.send(entity_id, msg)


def _conductor(comm, workers, passes, force_drain):
    c = Conductor.__new__(Conductor)
    c.communicator = _ApiLoopback(comm)
    c.model = _Model(passes, workers)
    c.worker_graphs = {g.worker_graph_id: g for g in _graphs(workers)}
    c.requests, c.draining, c._early_reads_done = {}, {}, {}
    c._early_reads_done_deadlines, c._draining_deadlines = deque(), deque()
    c._drain_ttl_s = 120
    c._client_streams = {"forced": SimpleNamespace(partition="__forced__")} if force_drain else {}
    c._stream_fed_partitions, c.producer_triggered_partitions, c._chunk_ticks = set(), set(), {}
    c.enable_prof = False
    c._try_admit_waiting = lambda: None
    return c


@dataclass
class Wave:
    wall_us: float
    conductor_cpu_us: float
    worker_cpu_us: float
    total_cpu_us: float
    drain_messages: int
    completion_reports: int
    certified_reports: int


def _wave(c, comm, workers, concurrency, sequence, passes, force_drain, pending_step):
    names = [f"r{sequence}_{i}" for i in range(concurrency)]
    for worker in workers:
        comm.send(worker, {"op": "prepare", "rids": names})
    ready = 0
    while ready < len(workers):
        for msg in _receive(comm):
            assert msg["op"] == "ready"
            ready += 1
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    c.communicator.drains = 0
    assignments = {i: [w] for i, w in enumerate(workers)}
    config = ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={})
    config.setup({NodeAndGraphWalk(f"n{i}", "run"): [worker] for i, worker in enumerate(workers)})
    for name in names:
        p = PartitionState(
            partition_name="default", metadata=CurrentForwardConductorMetadata("run", False),
            current_worker_graph_ids=set(assignments),
        )
        extra = {}
        if _CLOSURE:
            extra["input_closure_enabled"] = not force_drain
            p.input_workers = set(workers)
        c.requests[name] = RequestData(
            persist_signals={}, persist_signal_ref_cnt={}, worker_graph_to_workers=assignments,
            all_worker_graph_ids=set(assignments), max_output_tokens=1 << 30, random_seed=0,
            resource_configs={}, partition_states={"default": p}, sharding_config=config, **extra,
        )
        for index, worker in enumerate(workers):
            info = CurrentForwardPassInfo(name, "run", 0, 0, 1 << 30)
            if _CLOSURE:
                info.input_epoch = -1 if force_drain else 0
            comm.send(worker, WorkerMessage(WorkerMessageType.NEW_REQUEST, NewRequest(
                request_id=name, partition_worker_graph_ids=list(assignments),
                worker_graph_to_workers=assignments,
                initial_inputs=[GraphEdge(name="input", next_node=f"n{index}")],
                request_info=info, loop_caps={},
            )))
    removed, worker_cpu, reports, certified = 0, 0.0, 0, 0
    while removed < len(workers):
        for msg in _receive(comm):
            if isinstance(msg, dict):
                assert msg["op"] == "removed"
                removed += 1
                worker_cpu += msg["cpu_us"]
                reports += msg["reports"]
            elif msg.message_type == ConductorMessageType.WORKER_GRAPHS_DONE:
                certified += getattr(msg.body, "closed_input_epoch", -1) >= 0
                for partition in c._process_worker_graphs_done(msg.body):
                    if c._process_done_forward(msg.body.request_id, partition):
                        c._process_request_done(msg.body.request_id)
            elif msg.message_type == ConductorMessageType.READS_DONE:
                c._handle_reads_done(msg.body)
            else:
                raise AssertionError(msg)
        for name in c.communicator.completed:
            c._handle_reads_done(ReadsDone(name, c.PREPROCESS_WORKER))
        c.communicator.completed.clear()
    wall_us = (time.perf_counter() - wall_start) * 1e6 / concurrency
    cpu_us = (time.process_time() - cpu_start) * 1e6 / concurrency
    assert not c.requests and not c.draining
    assert reports == len(workers) * concurrency * passes
    expected_drains = len(workers) * concurrency if force_drain or (_CLOSURE and pending_step) else 0
    assert c.communicator.drains == expected_drains, "unexpected fallback or unsafe fast path"
    assert certified == (reports if _CLOSURE and not force_drain and not pending_step else 0)
    return Wave(wall_us, cpu_us, worker_cpu / concurrency, cpu_us + worker_cpu / concurrency,
                c.communicator.drains, reports, certified)


def _sources():
    root = Path(inspect.getfile(Worker)).resolve().parents[2]
    paths = [
        root / name for name in (
            "mstar/worker/worker.py", "mstar/conductor/conductor.py", "mstar/conductor/request_info.py",
            "mstar/utils/ipc_format.py", "mstar/graph/runtime/rust.py", "mstar/graph/runtime/base.py",
        )
    ] + [Path(mstar_rust.__file__), Path(__file__)]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 32, 64])
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--waves", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--force-drain", action="store_true")
    parser.add_argument("--pending-step", action="store_true", help=(
        "Pass every row as having another submitted step, exercising closure's fallback; "
        "no GPU work or wait is simulated."
    ))
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if min(args.workers, args.passes, args.waves, *args.concurrency) < 1 or args.warmup < 0:
        parser.error("workers, passes, waves and concurrency must be positive")
    logging.disable(logging.INFO)
    before = _sources()
    ctx = mp.get_context("spawn")
    workers = [f"worker_{i}" for i in range(args.workers)]
    with tempfile.TemporaryDirectory(prefix="ic_", dir="/tmp") as prefix:
        comm = make_communicator("conductor", workers, ipc_socket_path_prefix=prefix)
        processes = [
            ctx.Process(target=_worker, args=(i, workers, prefix, args.pending_step))
            for i in range(len(workers))
        ]
        for process in processes:
            process.start()
        try:
            c = _conductor(comm, workers, args.passes, args.force_drain)
            results = {
                "platform": platform.platform(), "loadavg": os.getloadavg(),
                "workers": args.workers, "passes": args.passes, "waves": args.waves,
                "warmup": args.warmup, "force_drain": args.force_drain, "closure": _CLOSURE,
                "pending_step": args.pending_step,
                "sources": before, "results": [],
            }
            sequence = 0
            for concurrency in args.concurrency:
                samples = []
                for wave in range(args.warmup + args.waves):
                    row = _wave(
                        c, comm, workers, concurrency, sequence, args.passes, args.force_drain, args.pending_step,
                    )
                    sequence += 1
                    if wave >= args.warmup:
                        samples.append(vars(row))
                result = {"concurrency": concurrency, "samples": samples}
                result.update({key: statistics.median(s[key] for s in samples) for key in samples[0]})
                results["results"].append(result)
                print(json.dumps({k: v for k, v in result.items() if k != "samples"}), flush=True)
            assert before == _sources(), "source or native build changed during measurement"
            args.json.write_text(json.dumps(results, indent=2) + "\n")
        finally:
            for worker in workers:
                comm.send(worker, {"op": "stop"})
            for process in processes:
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
            assert all(p.exitcode == 0 for p in processes), "a benchmark worker failed"


if __name__ == "__main__":
    main()
