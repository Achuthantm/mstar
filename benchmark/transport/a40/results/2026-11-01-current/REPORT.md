# Current-main A40 cleanup validation

Base: `3e98682a8fff4b506136a53541d868b3ceb804eb`. Fix: `eabb5f56`
(the refreshed cleanup implementation; subsequent commits change tooling/evidence).

Seven completed servers passed: four Orpheus disaggregated streaming ABBA
runs at c64, TP-sync base/fix at c1 and c64, and TP-async fix at c1 and c64.
**710 measured requests completed, zero failed.** Streaming used 128 measured
requests and two warmup waves per restart. TP used two measured c1 requests
and 64 measured c64 requests, with one warmup wave per cell.

TP used `NCCL_P2P_DISABLE=1` on both arms. The native extension was rebuilt
from the common Rust tree before testing; native-build.json records its hash.
Fresh Inductor/Triton caches were used per boot. Qwen was not run.

Streaming audio seconds/s in base/fix/fix/base order:
18.4501 / 19.0343 / 19.5731 / 23.4800. Baseline spread is large; these runs
show no clear regression but cannot exclude a small one. TP is functional
evidence, not an accepted performance comparison. Output volumes, latency
metrics, per-request results and phase data are retained for review.

The first controller reached its 25-minute cap before starting async. The
optional baseline-async control was then cancelled during startup to save
rental time. The final fix-only async run completed both cells. The cancelled
control is not a pass or a model failure; original logs/checkpoints remain.

The fresh x86 native cleanup test run passed 81 tests. Full ARM CPU results
are in ../2026-11-01/CPU_REVIEW.md: 9084 passed / 1012 skipped / 44 failed on
the fix, with exactly the same failure IDs on baseline (9041 passed).

The inherited forced-drain deadline is still a safety limit for indefinitely
stalled transfers. No cross-host/RDMA, Qwen, audio-quality or universal
minimal-overhead claim follows from these checks.

All benchmark controllers and model servers exited, port 8100 was released,
and both GPUs reported 0 MiB allocated at collection. The logs, metadata,
results and checksums are preserved here. Compile caches, weights, credentials
and generated WAVs are excluded. No more rental tests are planned.

## Offline cleanup and overhead review

`PYTHONPATH=. python3 benchmark/transport/a40/results/2026-11-01-current/review.py`
reproduces [review.json](review.json) from the saved server logs and client
results, without a GPU. The audit checks request identities, not just counts:
all **1,417 requests including warmup** have exactly one conductor completion,
one conductor teardown, and one cleanup record on **each** worker. No completed
run logged an ERROR, traceback, or forced drain timeout. The client results use
ordinal IDs rather than server UUIDs, so the client-to-server check is the total
expected count; the conductor-to-worker check matches the UUID sets exactly.

Each streaming server logs two CUDA IPC producer-shutdown warnings, on both
base and fix. These occur during server shutdown. The cleanup records show
that the removal methods reached their final log, not that every allocator or
CUDA IPC reference was empty. The cancelled baseline-async startup is excluded:
its orphan-worker errors followed the intentional controller cancellation.

Current-main worker 1 phases, in base/fix/fix/base order:

| Metric | Base 1 | Fix 1 | Fix 2 | Base 2 |
|---|---:|---:|---:|---:|
| Selected timing records | 37 | 59 | 59 | 33 |
| Mean actual batch size | 63.584 | 63.320 | 63.003 | 63.527 |
| Teardown CPU, µs/sample | 9.568 | 9.881 | 8.949 | 8.152 |
| Message processing CPU, µs/sample | 1,704.297 | 1,833.966 | 1,727.847 | 1,417.303 |
| Iteration wall time, ms | 30.256 | 32.607 | 31.868 | 27.432 |
| GPU event wait, ms/sample | 0.190 | 0.177 | 0.251 | 0.303 |
| Mean output audio duration, s/request | 4.481 | 4.423 | 4.463 | 4.533 |

Selection is the longest consecutive worker-1 segment with batch size 60–64,
including warmup, not an isolated client measurement interval. Worker 0 has
no comparable sustained segment at that batch size. Means are weighted by
sample count; the spans are not summed. These are current-main observations;
the earlier CPU_REVIEW phase table measures a different pinned source pair.

The average teardown-poll difference is about **0.56 µs/sample**. This does not
measure the whole protocol: message processing also handles the drain exchange
and other traffic. Its average difference is about **220 µs/sample**, with a
baseline-to-baseline change of about **287 µs/sample**. Neither is an isolated
cost for DRAIN/READS_DONE. Mean audio duration varies by about 2.5% across runs;
that does not by itself explain the roughly 27% baseline throughput spread.
Kernel selection, scheduling, or other restart effects remain hypotheses; the
saved evidence does not isolate their causes. Small GPU event waits alone do
not exclude host synchronization charged to other spans.

## Audit against the requested conditions

1. **Minimal normal-case overhead is not established.** The fix adds a
   DRAIN/READS_DONE exchange per compute participant even when no read remains.
   `_process_request_done` sends client completion before initiating this
   exchange, but deferred resource reuse and shared CPU work can still affect
   serving performance. The CPU microbenchmark measures real nonzero cost;
   the GPU measurements cannot resolve a small regression. There is no proved
   safe zero-exchange fast path in this implementation.
2. **The runtime change reuses the existing barrier and state.** It adds no
   message fields, per-step request objects, or persistent containers.
   `_register_draining` no longer takes a redundant expected-ACK argument.
   `_apply_pending_drains` skips already acknowledged requests using the two
   existing sets; `_mark_removed` removes from both, and ACK insertion occurs
   only for a draining request. Direct conductor drains to TP followers are
   ignored by the existing follower guard; leaders forward drains in order.
   Removing that duplicate transport traffic would require proving routing
   coverage and failure behavior, and has not been implemented or measured.
3. **The ordinary barrier path has substantial correctness evidence, with
   explicit limits.** The controlled race tests fail on baseline and pass on
   the fix; the current native tests and completed GPU runs cover asynchronous
   reads and TP sync/async execution. `_complete_drain_if_ready` waits for
   inflight reads and committed TP follows, then seals future input through
   `_takes_input`. `_drain_request` defers scheduler mutation behind GPU work.
   The inherited `_sweep_expiry` still forces removal after its deadline even
   without all ACKs. Thus indefinite stalls do not have an unconditional
   no-read-after-free guarantee. No claim of universal safety or negligible
   overhead is justified by these runs.

A simple “no reads in flight” snapshot at graph completion is insufficient:
it does not prevent a later descriptor from starting a read. Eliminating the
normal drain exchange would require a durable read seal at a point where no
valid later walk or committed TP input can need that request. That invariant
has not been established, so adding such a shortcut would weaken the fix.
