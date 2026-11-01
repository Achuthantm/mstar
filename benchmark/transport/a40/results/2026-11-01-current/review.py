"""Reproduce the saved-run cleanup audit: PYTHONPATH=. python <this file>."""

import gzip
import hashlib
import itertools
import json
import re
from collections import Counter
from pathlib import Path

from benchmark.worker_phases.phases import Segment, parse_log


def review(root):
    summary = json.loads((root / "summary.json").read_text())
    rows = []
    for run in summary["server_runs"]:
        scenario = run["scenario"]
        group = "fix-async" if scenario == "tp-async" else "main"
        case = root / "raw" / group / scenario / Path(run["case"]).name
        source = case / "server.log.gz"
        text = gzip.decompress(source.read_bytes()).decode()
        done = Counter(re.findall(r"\[conductor\].*?: Request (\S+) done\b", text))
        removed = Counter(re.findall(r"Tore down request (\S+);", text))
        workers = {
            worker: Counter(re.findall(
                rf"\[{worker}\].*?Request cleanup complete: (\S+)", text,
            )) for worker in ("worker_0", "worker_1")
        }
        cells = [json.loads(p.read_text()) for p in sorted(case.glob("clients/*/results.json"))]
        expected = sum(c["num_requests"] + c["num_warmup"] * c["aggregate"]["max_concurrency"]
                       for c in cells)
        if not cells or not done or len(done) != expected:
            raise ValueError(f"{case}: expected {expected} unique completions, found {len(done)}")
        if any(c["failed"] or c["completed"] != c["num_requests"] for c in cells):
            raise ValueError(f"{case}: incomplete client results")
        if any(c != done for c in (removed, *workers.values())) or set(done.values()) != {1}:
            raise ValueError(f"{case}: duplicate or missing cleanup records")
        errors = [line for line in text.splitlines()
                  if re.search(r"\bERROR\b|Traceback \(most recent|forcing teardown", line)]
        if errors:
            raise ValueError(f"{case}: errors: {errors}")
        row = {
            "case": str(case.relative_to(root)), "commit": run["commit"],
            "server_log_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "measured_requests": sum(c["completed"] for c in cells),
            "requests_including_warmup": expected,
            "exactly_once_conductor_and_both_worker_cleanup": True,
            "errors_or_forced_teardowns": len(errors),
            "cuda_ipc_shutdown_warnings": text.count("Producer process has been terminated"),
        }
        if scenario == "streaming":
            records = parse_log(text)
            seq = [r for r in records if r.worker == "worker_1"]
            runs = [list(g) for match, g in itertools.groupby(seq, lambda r: 60 <= r.bs <= 64)
                    if match]
            seg = Segment("worker_1", max(runs, key=len))
            row["worker_1_steady_batch"] = {
                "iterations": seg.iters, "records": len(seg.records),
                "batch_mean": seg.mean_bs, "batch_range": seg.bs_range,
                "phases": {k: {f: v[f] for f in ("mean_ms", "samples")}
                           for k, v in seg.summary().items() if k in (
                               "iter_total", "worker.main.teardown.cpu",
                               "worker.main.process_messages.cpu", "worker.postprocess.event_sync",
                           )},
            }
            row["audio_duration_mean_s"] = cells[0]["aggregate"]["audio_duration_mean_s"]
        rows.append(row)
    return {
        "scope": "Seven completed runs; excludes intentionally cancelled baseline async. "
                 "Log records prove cleanup methods reached their final log, not allocator emptiness. "
                 "Client IDs are ordinal and cannot be joined to server UUIDs; server sets are "
                 "matched exactly and their sizes checked against measured plus warmup counts.",
        "phase_selection": "Longest consecutive worker_1 run with bs 60..64; includes warmup. "
                           "Means weighted by sample count; spans not added together. "
                           "Worker_0 has no comparable sustained bs 60..64 segment.",
        "rows": rows,
    }


if __name__ == "__main__":
    print(json.dumps(review(Path(__file__).resolve().parent), indent=2))
