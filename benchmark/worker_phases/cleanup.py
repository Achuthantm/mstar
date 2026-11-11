"""CPU cost of input-closure and speculative teardown filtering.

Export the baseline step_executor.py or request_lifecycle.py, then run:
    python -m benchmark.worker_phases.cleanup --component closure --baseline-source /tmp/step_executor.py
    python -m benchmark.worker_phases.cleanup --component teardown --baseline-source /tmp/request_lifecycle.py

Arms alternate within one process. Both must pass the same behavior checks,
including backlog exclusion. This isolates CPU filtering, not serving throughput.
"""
import argparse
import hashlib
import inspect
import json
import platform
import runpy
import statistics
import time
import timeit
from pathlib import Path
from types import SimpleNamespace

from mstar.communication.tensor_store import RustTensorBookkeeping
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.base import ShardingConfig
from mstar.graph.runtime.rust import RustGraphRuntime
from mstar.graph.runtime.utils import require_rust_extension
from mstar.utils.containers import ParallelList
from mstar.utils.ipc_format import DrainRequest
from mstar.worker.scheduled_batch import ScheduledBatch
from mstar.worker.services import WorkerLayout


def _closure_case(cls, rows, state):
    method = cls._input_closure_candidates
    rids = list(range(rows)) if state != "no_completion" else []
    infos = [CurrentForwardPassInfo(
        request_id=f"r{rid}", graph_walk="run", fwd_index=0,
        random_seed=0, max_tokens=8, partition_name="default", input_epoch=0,
    ) for rid in rids]
    backlog = {}
    if state == "unrelated_backlog":
        backlog = {
            (f"n{i}", "run"): ScheduledBatch(f"n{i}", "run", dict.fromkeys(range(rows, rows + 4096), i))
            for i in range(4)
        }
    elif state == "overlapping_backlog":
        backlog[("n0", "run")] = ScheduledBatch("n0", "run", dict.fromkeys(rids[::2], 0))
    executor = cls.__new__(cls)
    executor._layout = WorkerLayout()
    executor._scheduler = SimpleNamespace(backlog=backlog)

    args = (executor, rids, infos)
    args += (0, (), {}) if "lane" in inspect.signature(method).parameters else ((),)

    def run():
        return method(*args)

    expected = rids if state != "overlapping_backlog" else rids[1::2]
    assert run() == frozenset(expected)
    return run


def _teardown_case(lifecycle_type, rows, state):
    runtime = RustGraphRuntime(
        my_worker_id="worker", my_worker_graphs=[],
        all_wg_ids_to_graph_walks={}, all_wg_ids_to_dyn_loops={},
        all_wg_ids_to_nodes={}, node_to_partition={},
        sharding_config=ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}),
        bookkeeping=RustTensorBookkeeping(),
    )
    lifecycle = lifecycle_type(WorkerLayout(), runtime)
    rids, names = [], []
    for i in range(rows + 1):
        wire = f"00000000-0000-4000-8000-{i:012d}"
        rid = runtime.add_request(wire, "default", "decode", [], ParallelList([], []), {})
        lifecycle.add(rid, wire)
        rids.append(rid)
        names.append(wire)
    if state in ("draining", "mixed"):
        lifecycle.begin_drain(names[-1])
    if state in ("pending", "mixed"):
        lifecycle.drain(DrainRequest(request_id=names[0]))
    rids = rids[:-1]
    batch = ScheduledBatch("node", "decode", dict.fromkeys(rids, 0))

    def run():
        return batch.get_continuing_rows(
            "node", rids,
            lifecycle.is_tearing_down if lifecycle.teardowns_pending() else None,
            (),
        )

    expected = rids[1:] if state in ("pending", "mixed") else rids
    assert run().loop_back == expected
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("closure", "teardown"), required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if min(*args.rows, args.iterations, args.repeats) < 1:
        parser.error("rows, iterations and repeats must be positive")
    require_rust_extension()
    if args.component == "closure":
        from mstar.worker.step_executor import StepExecutor as Candidate
        case, states = _closure_case, ("no_completion", "no_backlog", "unrelated_backlog", "overlapping_backlog")
    else:
        from mstar.worker.request_lifecycle import RequestLifecycle as Candidate
        case, states = _teardown_case, ("idle", "draining", "pending", "mixed")
    baseline = runpy.run_path(str(args.baseline_source))[Candidate.__name__]
    sources = {"baseline": args.baseline_source, "candidate": Path(inspect.getfile(Candidate))}
    print(json.dumps({
        "platform": platform.platform(), "python": platform.python_version(),
        "iterations": args.iterations, "repeats": args.repeats,
        "sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()},
        "component": args.component, "metric": "thread CPU microseconds per batch filter",
    }))
    for rows in args.rows:
        for state in states:
            runs = [case(cls, rows, state) for cls in (baseline, Candidate)]
            timers = [timeit.Timer(run, timer=time.thread_time) for run in runs]
            for timer in timers:
                timer.timeit(number=1000)
            samples = [[], []]
            for repeat in range(args.repeats):
                for arm in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                    samples[arm].append(timers[arm].timeit(number=args.iterations) * 1e6 / args.iterations)
            print(json.dumps({
                "rows": rows, "state": state,
                "baseline_us": samples[0], "candidate_us": samples[1],
                "baseline_median_us": statistics.median(samples[0]),
                "candidate_median_us": statistics.median(samples[1]),
            }), flush=True)


if __name__ == "__main__":
    main()
