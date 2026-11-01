# CPU evidence for request cleanup

The returned A40 archive contains CPU phase timings. The teardown poll itself is
small; it does **not** account for the whole cleanup protocol. Drain handling
also runs in message processing, and the conductor pays for an extra drain/ACK
exchange with every worker on successful completion.

## CPU phases from the returned GPU runs

Source: `exports/mstar-a40-evidence-analysis-2026-11-01.tar.gz`, SHA256
`2f11dce9104bb13ebf6d4c418136b305abdad7caed38e7836bbf688bdb009e09`.
The machine-readable [phase summary](cpu-phase-summary.jsonl) records the archive
hash, source paths, commits, worker, iteration bounds, batch ranges and sample
counts. Verify the hash against that file when copying this report.

These are the original A40 versions: base `9f84f18c`, fix `f7e95a03`. They are
historical evidence for that pair, rather than measurements after the main merge.
Selection uses the longest consecutive segment with batch size exactly 1
(discarding its first five timing records), or batch size 60–64. This includes
client warmup; it is a steady-batch comparison rather than a precisely isolated
client measurement interval. The repository phase parser recombines each mean
using its sample count. Worker 0 has no sustained comparable high-batch segment
and is omitted at concurrency 64.

Worker 1's autoregressive stage, in base/fix/fix/base order:

| Metric | Base 1 | Fix 1 | Fix 2 | Base 2 |
|---|---:|---:|---:|---:|
| c1 teardown CPU, µs/sample | 4.139 | 4.311 | 3.584 | 4.461 |
| c1 message processing CPU, µs/sample | 68.661 | 77.145 | 61.806 | 71.741 |
| c1 iteration wall time, ms | 26.265 | 26.233 | 26.234 | 26.232 |
| c64 mean actual batch size | 63.227 | 63.102 | 63.118 | 63.234 |
| c64 teardown CPU, µs/sample | 6.160 | 7.486 | 7.080 | 6.736 |
| c64 message processing CPU, µs/sample | 1,143.263 | 1,423.153 | 1,363.822 | 1,202.971 |
| c64 iteration wall time, ms | 43.220 | 44.315 | 43.512 | 42.926 |
| c64 GPU event wait, ms/sample | 20.582 | 17.028 | 17.607 | 19.342 |

The c64 teardown-poll increase is about 0.84 µs/sample when averaging the two
means per arm. Message processing increases by about 220 µs/sample, but it also
handles inputs and other control traffic; the data does not isolate drain
messages as the cause. GPU-thread and plan-thread CPU spans change too. These
figures cannot attribute the entire throughput difference to cleanup.

Do not add nested spans or waits to obtain a CPU work total. In particular,
`event_sync` is a GPU wait, whereas `await_gpu` waits for the GPU thread's host
work. A thread CPU clock may include active polling in a native GPU wait; it
does not identify time spent executing Python. The nonzero event wait establishes
a GPU-bound component, not GPU utilization or a purely GPU-bound workload.

The final base c64 client completed 319/320 requests. The approximate 2.7%
streaming-throughput difference remains provisional, and these phase timings
do not repair that incomplete baseline. See [the GPU report](REPORT.md).

## Direct teardown CPU measurement on the A40 host

The [raw x86 results](cpu-x86/metadata.json) were measured with the existing
`benchmark/transport/request_cleanup.py`, Python 3.12.16 and torch 2.10.0+cu128.
They use the **same old base/fix pair as the returned GPU runs**; the native
extension was rebuilt for the refreshed main only after this measurement.
The two A40 GPUs remained idle, with no inference server running. The
[hardware inventory](cpu-x86/hardware.txt) is host-visible, rather than a claim
of exclusive CPU allocation.

Each backend/worker-count cell used base/fix/fix/base order, 20 warmup waves and
200 measured waves at concurrency 1/32/64. No request had an outstanding read.
Real ZMQ sockets and the real conductor/worker cleanup handlers were used;
engine resources and tensor payloads were absent. The API ACK was local to the
conductor. Results include source-file paths/hashes, runtime type, codec and
host load. This isolates protocol cost, not serving throughput.

For concurrency 64, averaging the two run medians per arm:

| Mesh | Workers | Added CPU/request, µs | Added amortized wall/request, µs |
|---|---:|---:|---:|
| Python | 1 | 76.15 | 14.76 |
| Python | 2 | 75.09 | 2.43 |
| Rust | 1 | 130.34 | 35.74 |
| Rust | 2 | 43.04 | 0.99 |

CPU is the sum of conductor and all worker process CPU times. Wall/request is
the wave wall time divided by concurrency, **not per-client latency**. Same-code
variation is substantial: with two Python workers, base c1 wall/request is
1,530/518 µs and fix is 1,370/2,840 µs; with Rust it is 221/343 µs versus
670/687 µs. Across all measured cells, the added c1 amortized wall cost is about
0.40–2.04 ms. Those results do not support a universal overhead number or a
claim that normal completion is free. Every completion adds a drain/ACK
exchange per worker; ACKed drains avoid repeated polling scans.

## Current CPU validation

On the refreshed main (`3e98682a`) plus the cleanup fix, the full ARM Linux CPU
selection with torch 2.10.0+cpu, a rebuilt extension and `OMP_NUM_THREADS=2`
completed with **9,084 passed, 1,012 skipped, 44 failed**, plus 104 passing
subtests. The runner disables oneDNN and uses the four documented ARM
exclusions; this is not the full x86 CI job.

A full run on clean baseline completed with **9,041 passed, 1,012 skipped,
44 failed**, plus 104 passing subtests. The failure IDs match **exactly**:
32 loop-output tests access Python bookkeeping's private `_ref_info` while the
installed extension selects Rust bookkeeping, 11 numerical model cases fail
in this environment, and a transcription test fails only in the full-suite
context on both versions (it passes in isolation on both). There are 43
additional passing cases on the fix and no new failures. Neither run is green.
The [comparison](cpu-tests-summary.jsonl) records the exact failure IDs,
runtime source hashes and XML hashes; full JUnit results are retained for
[base](cpu-base-junit.xml.gz) and [fix](cpu-fix-junit.xml.gz).

The subsequent focused cleanup/race/barrier/worker-drain/TP-order/TP-simulation
and benchmark-tooling check passed **115 tests**, including the isolated
transcription retry. Repository-wide Ruff and 80 CI-selection/snapshot tests
passed. The fresh native extension on the x86 GPU host also passed **81**
race/barrier/worker-drain/TP-order tests before starting the new GPU checks.

Earlier GPU measurements and the x86 protocol benchmark are evidence for the
old pinned versions. Current CPU test results are reported separately; they do
not establish current-version GPU performance.
