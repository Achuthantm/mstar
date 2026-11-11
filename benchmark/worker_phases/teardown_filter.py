"""CPU cost of speculation's teardown filter, using the real Rust handle table.

Export a baseline request_lifecycle.py, then run from the candidate checkout:
    python -m benchmark.worker_phases.teardown_filter --baseline-source /tmp/baseline.py

Arms alternate within one process. This times row filtering, not GPU serving.
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

from mstar.communication.tensor_store import RustTensorBookkeeping
from mstar.distributed.base import ShardingConfig
from mstar.graph.runtime.rust import RustGraphRuntime
from mstar.graph.runtime.utils import require_rust_extension
from mstar.utils.containers import ParallelList
from mstar.utils.ipc_format import DrainRequest
from mstar.worker.request_lifecycle import RequestLifecycle
from mstar.worker.scheduled_batch import ScheduledBatch
from mstar.worker.services import WorkerLayout


def _case(lifecycle_type, rows, state):
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
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if min(*args.rows, args.iterations, args.repeats) < 1:
        parser.error("rows, iterations and repeats must be positive")
    require_rust_extension()
    baseline = runpy.run_path(str(args.baseline_source))["RequestLifecycle"]
    sources = {"baseline": args.baseline_source, "candidate": Path(inspect.getfile(RequestLifecycle))}
    print(json.dumps({
        "platform": platform.platform(), "python": platform.python_version(),
        "iterations": args.iterations, "repeats": args.repeats,
        "sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()},
        "metric": "thread CPU microseconds per batch filter",
    }))
    for rows in args.rows:
        for state in ("idle", "draining", "pending", "mixed"):
            runs = [_case(cls, rows, state) for cls in (baseline, RequestLifecycle)]
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
