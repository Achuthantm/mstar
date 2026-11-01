# Request cleanup A40 validation — 2026-11-01

Base: `9f84f18cd7e1b4f3851d0bde85b8da45cabec06f`.
Fix: `f7e95a03c6f0753ef27e29ccb4eaeba5eec680cc`.

## Conclusion and limits

The deterministic cleanup race reproduces on the base and passes with the fix.
The focused CPU suite passes. Single-request streaming throughput is essentially
unchanged. High-concurrency results suggest a small overhead, but are provisional
because one baseline request disconnected. TP serving was not validated: default
NCCL timed out before model startup. A separate NCCL test passed with
`NCCL_P2P_DISABLE=1`; subsequent TP model runs were cancelled at the user's request
to end GPU spending. This is evidence for further development, not full acceptance.
The inherited forced-drain timeout remains a safety limit; these tests do not
prove safety for an indefinitely stalled transfer. No cross-host/RDMA claim.

## Environment and method

Two A40 GPUs, CUDA toolkit 12.8, driver 595.91.07, PyTorch 2.10.0+cu128,
Python 3.12, release Rust extension built from the pinned fix (Rust unchanged
between arms). GPU topology PXB. Host-visible Intel Xeon Gold 6342, 96 logical
CPUs and 503 GiB RAM; these are host-visible values, not dedicated allocations.
Full details are in preflight.txt, packages.txt and each case's metadata.
Unexplained GPU load existed before benchmarking; the user elected to proceed.
Both GPUs passed arithmetic, matrix correctness and 1 GiB memory checks without
CUDA errors; ECC counters were zero.

Measurements used unchanged detached base/fix trees, identical configs from each
arm, one shared environment/cache, -O2, Rust graph/ZMQ and SHM arena. Sequence:
base/fix/fix/base. Concurrency 1: 5 warmup requests, 16 measured. Concurrency 64:
320 warmup requests (5 waves), 320 measured. Qwen was excluded at user request.

## Correctness and transport

- Focused fix cleanup/race/drain/teardown/TP tests: 111 passed; conductor status:
  6 passed (117 total). Race module alone: 12 passed.
- Baseline race module: 8 expected failures, 4 passes. It substitutes the native
  transfer engine and is not a native GPU/RDMA race reproduction.
- Streaming measured requests: base 671/672, fix 672/672. Final base c64 case
  completed 319/320; client reported `Server disconnected`. No cause established.
- Python runtime SHM streaming and TCP: both versions passed, each with one
  warmup and 2 measured requests per check.
- Rust TCP: both versions passed after installing libibverbs1/librdmacm1;
  each with one warmup and 2 measured requests. Untimed functional checks.
- TP sync and async: initial base attempts both failed with NCCL timeout on the
  first one-element all-reduce. Standalone default NCCL reproduced the timeout;
  standalone with P2P disabled passed on both ranks. Model retries not executed.

## Streaming measurements

Per fresh server, in base/fix/fix/base order:

| Metric | Base 1 | Fix 1 | Fix 2 | Base 2 |
|---|---:|---:|---:|---:|
| c1 audio sec/s | 0.457813 | 0.458254 | 0.458356 | 0.458106 |
| c1 median JCT, seconds | 9.3435 | 9.1493 | 9.1426 | 9.1287 |
| c64 audio sec/s | 15.7514 | 15.5493 | 15.4496 | 16.0927* |
| c64 median JCT, seconds | 16.2554 | 16.7263 | 16.5568 | 16.0363* |

*One failed request; these aggregates use successful requests. They must not be
silently treated as an accepted complete baseline run.

Median-of-repeats differences: c1 throughput +0.075%, JCT -0.976%; c64 throughput
-2.654%, JCT +3.070%. Base c64 throughput spread is 2.167%; fix spread is 0.645%.
Mean audio duration differs by about +0.804% for the fix at c64. These descriptive
numbers are not a causal performance finding. TTFT/RTF, per-request results,
output audio and worker phases are preserved for offline review. Memory-growth
and steady-phase analysis are still pending; no leak-free conclusion is claimed.

## Export and continuation

`validation-checkpoint.json` records each final retry, exact command, PID, exit
status and cancellation. Failed and partial attempts remain intact. A partial
streaming retry was stopped after Rust TCP completed. No GPU jobs remain.

The analysis archive contains all logs, JSON results, phase tables, provenance,
checkpoints, scripts and checksums; the full archive adds generated WAV outputs.
Model weights, environment files and Hugging Face credentials are excluded.
Download an archive before deleting the rental. Archive presence on this instance
is not an off-instance backup. Work remaining offline: investigate the disconnect,
review phase/memory evidence and decide whether the possible c64 overhead warrants
another targeted allocation. TP model acceptance remains an explicit open item.
