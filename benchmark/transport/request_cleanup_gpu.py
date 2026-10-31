"""Run the existing GPU benchmark clients against an already-running server.

See request_cleanup_gpu.md for the base/fix worktrees and server commands.
--dry-run prints the sweep without starting clients or probing hardware.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

CLIENTS = {
    "qwen": "qwen3_5_t2t.sh",
    "orpheus": "orpheus_t2s.sh",
    "omni": "qwen3omni_t2s.sh",
}


def read_result(path: Path, expected: int, concurrency: int) -> dict:
    result = json.loads(path.read_text())
    if result["failed"] or result["completed"] != expected:
        raise ValueError(f"{path}: expected {expected} completed and no failures")
    if result["aggregate"]["max_concurrency"] != concurrency:
        raise ValueError(f"{path}: wrong concurrency")
    if result["request_throughput"] <= 0:
        raise ValueError(f"{path}: nonpositive throughput")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", required=True, type=Path, help="new directory for this server instance")
    parser.add_argument("--url", default="http://127.0.0.1:8100")
    parser.add_argument("--model", choices=CLIENTS, default="qwen")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 32, 64])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if min(args.concurrency) < 1:
        parser.error("concurrency must be positive")
    checkout, out = args.checkout.resolve(), args.out.resolve()
    client = checkout / "benchmark/worker_phases/clients" / CLIENTS[args.model]
    if not client.is_file():
        parser.error(f"client not found: {client}")
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=False)
        with urlopen(args.url.rstrip("/") + "/health", timeout=5) as response:
            if response.status != 200:
                raise RuntimeError("server is not healthy")
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=checkout, text=True,
        ).strip()
        if dirty:
            raise RuntimeError("benchmark checkout has uncommitted tracked changes")
        metadata = {
            "checkout": str(checkout),
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=checkout, text=True,
            ).strip(),
            "runtime": {k: os.environ.get(k) for k in (
                "MSTAR_RUST_GRAPH", "MSTAR_RUST_ZMQ", "MSTAR_SHM_ARENA", "MSTAR_TP_ASYNC_SCHED",
            )},
        }
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        with (out / "gpu.txt").open("w") as log:
            subprocess.run([
                "nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv",
            ], stdout=log, stderr=subprocess.STDOUT, check=True)
        with (out / "packages.txt").open("w") as log:
            subprocess.run([sys.executable, "-m", "pip", "list", "--format=freeze"],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    summaries = []
    for concurrency in args.concurrency:
        # Short completions expose teardown cost; longer ones exercise decode.
        for tokens in ([16, 512] if args.model == "qwen" else [None]):
            requests = max(64, 8 * concurrency)
            label = f"c{concurrency}" + (f"-tokens{tokens}" if tokens is not None else "")
            case = out / label
            overrides = {
                "URL": args.url, "CONC": str(concurrency), "N": str(requests), "WARMUP": "5",
            }
            if tokens is not None:
                overrides.update(OUTMIN=str(tokens), OUTMAX=str(tokens))
            command = ["bash", str(client), "--output-dir", str(case)]
            if args.model == "qwen":
                command.extend(["--dataset", "text"])
            print(shlex.join(["env", *(f"{k}={v}" for k, v in overrides.items()), *command]), flush=True)
            if args.dry_run:
                continue
            case.mkdir()
            env = dict(os.environ, **overrides)
            env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
            with (case / "client.log").open("w") as log:
                subprocess.run(command, cwd=checkout, env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            result = read_result(case / "results.json", requests, concurrency)
            summary = {
                "case": label, "completed": result["completed"],
                "request_throughput": result["request_throughput"],
                "jct_median_ms": result["jct_median_ms"],
                "aggregate": result["aggregate"],
            }
            summaries.append(summary)
            (out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
            print(f"PASS {label}: {requests} completed; logs: {case}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
