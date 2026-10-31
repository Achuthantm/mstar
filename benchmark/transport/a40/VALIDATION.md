# Validation before the GPU handoff

The GPU runs are prepared, not executed. Local checks on the final scripts:

- Repository-wide `ruff check .`: passed.
- `bash -n` on setup.sh and all shell blocks in the GPU/agent handoffs: passed.
- Full orchestrator dry-run: 18 server runs, no server/GPU access or output creation.
- Script unit tests on macOS: 20 passed. These cover dry-run isolation, each arm's
  config paths, original client checks, startup/client failure cleanup, monitor
  termination, TCP request settings and preserving repeated measurements.
- Focused Linux aarch64 tests with torch 2.10.0+cpu, a freshly built native
  extension, `MSTAR_RUST_GRAPH=1`, `MSTAR_RUST_ZMQ=1`, `OMP_NUM_THREADS=2`:
  **117 passed**. Command:

```bash
python -m pytest -q \
  test/modular/test_request_cleanup_benchmark.py \
  test/modular/test_request_cleanup_race.py \
  test/modular/test_teardown_barrier.py test/modular/test_worker_drain.py \
  test/modular/test_conductor_status.py \
  test/modular/test_tp_async_teardown_order.py \
  test/modular/test_tp_async_lockstep_sim.py
```

Earlier validation of the same production changes: 8,624 CPU tests passed,
1,012 skipped and nine numerical failures exactly matched the pinned baseline;
558 native tests passed (108 skipped), and 77 Cargo release tests passed.
The late-read race has 8 failures/4 passes on the baseline and 12 passes with the
fix, across Python/Rust bookkeeping and read timing/removal order variants.

The repository CPU teardown harness uses real ZMQ and actual teardown handlers,
without GPU model execution. For no outstanding reads, the measured added
amortized wall cost at concurrency 64 was approximately 11–21 microseconds/request
across Python/Rust meshes and 1/2/4 workers, on a Linux ARM Docker VM on Mac16,1.
These results do not establish A40 serving throughput or optimal overhead.

A40 package installation, CUDA kernels, topology, real model traffic, TCP smoke,
and TP CUDA execution still need to be checked on the rental. CPU-only test
passes and dry runs cannot substitute for those checks. The inherited drain
expiry can force removal without every ACK; safety for a reader that remains
active after that deadline is not established by this change.
