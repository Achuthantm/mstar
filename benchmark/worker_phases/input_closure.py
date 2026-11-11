"""CPU cost of the worker's input-closure candidate filter.

Export a baseline step_executor.py, then run from the candidate checkout:
    python -m benchmark.worker_phases.input_closure --baseline-source /tmp/baseline.py

Arms alternate in one process. This isolates the filter, not serving throughput.
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

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.worker.scheduled_batch import ScheduledBatch
from mstar.worker.services import WorkerLayout
from mstar.worker.step_executor import StepExecutor


def _case(method, rows, state, *, baseline):
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
    executor = StepExecutor.__new__(StepExecutor)
    executor._layout = WorkerLayout()
    executor._scheduler = SimpleNamespace(backlog=backlog)

    def run():
        return method(executor, rids, infos, 0, (), {})

    expected = rids if baseline or state != "overlapping_backlog" else rids[1::2]
    assert run() == frozenset(expected)
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
    baseline = runpy.run_path(str(args.baseline_source))["StepExecutor"]
    sources = {"baseline": args.baseline_source, "candidate": Path(inspect.getfile(StepExecutor))}
    print(json.dumps({
        "platform": platform.platform(), "python": platform.python_version(),
        "iterations": args.iterations, "repeats": args.repeats,
        "sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()},
        "metric": "thread CPU microseconds per closure filter",
    }))
    for rows in args.rows:
        for state in ("no_completion", "no_backlog", "unrelated_backlog", "overlapping_backlog"):
            runs = [_case(cls._input_closure_candidates, rows, state, baseline=i == 0)
                    for i, cls in enumerate((baseline, StepExecutor))]
            timers = [timeit.Timer(run, timer=time.thread_time) for run in runs]
            for timer in timers:
                timer.timeit(number=1000)
            samples = [[], []]
            for repeat in range(args.repeats):
                for arm in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                    samples[arm].append(timers[arm].timeit(number=args.iterations) * 1e6 / args.iterations)
            print(json.dumps({
                "rows": rows, "state": state, "baseline_us": samples[0], "candidate_us": samples[1],
                "baseline_median_us": statistics.median(samples[0]),
                "candidate_median_us": statistics.median(samples[1]),
            }), flush=True)


if __name__ == "__main__":
    main()
