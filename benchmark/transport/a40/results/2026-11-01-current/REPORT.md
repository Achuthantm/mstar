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
