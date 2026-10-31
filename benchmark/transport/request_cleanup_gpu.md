# GPU validation for #244

These steps are prepared for later execution; GPU serving results have **not**
been measured on the development Mac. The CPU `request_cleanup` benchmark
measures the control exchange, not inference throughput or GPU overlap.

## Pin the two versions

On a CUDA host, use one Python environment with the model dependencies and a
fresh build of `rust/`. Base and fix must use the same torch/CUDA build,
weights, native extension, configs and runtime settings. This fix changes no
Rust source, so one extension built from this base serves both arms.

After reviewing and committing the fix, create detached worktrees. Replace
`FIX_COMMIT` with that commit's SHA; the current development HEAD is the base
until the fix is committed. Do not benchmark two names for the same commit.
If the fix is rebased later, use its new upstream base and rebuild the native
extension before comparing.

For transfer from the development Mac without a pushed branch, the local
validation bundle includes `.ci-results/issue244.patch`. Copy that patch to
the GPU host, apply it to this base, then review and commit it there:

```bash
git switch -c fix-244-gpu 9f84f18cd7e1b4f3851d0bde85b8da45cabec06f
git apply /path/to/issue244.patch
```

```bash
BASE=9f84f18cd7e1b4f3851d0bde85b8da45cabec06f
FIX=FIX_COMMIT
git worktree add --detach ../mstar-244-base "$BASE"
git worktree add --detach ../mstar-244-fix "$FIX"
```

Cache the model checkpoints and startup inputs before the measured runs. Use
an existing GPU allocation; keep both arms on the same GPUs, run them serially,
and record any competing GPU/CPU load. Use at least two server restarts per
arm, in `base, fix, fix, base` order. Repeat that sequence if the result is
close to the variation between the two baseline runs.

## Sensitive normal case: Qwen3.5 0.8B

From the selected arm's worktree, terminal 1 (activate the same environment in
both terminals). Give every server instance a new absolute results directory.

```bash
ARM=$PWD
PY=$(command -v python)
RUN=/absolute/path/to/results/qwen-base-1
mkdir -p "$RUN"
export PYTHONPATH="$ARM"
export MSTAR_RUST_GRAPH=1 MSTAR_RUST_ZMQ=1 MSTAR_SHM_ARENA=1

python -c 'import mstar, torch; print(mstar.__file__, torch.__version__, torch.version.cuda)' \
  > "$RUN/source.txt"
COMMAND="$PY -m mstar.cli.main serve qwen3_5_0.8b --config configs/qwen3_5_0.8b.yaml --gpus 0 --port 8100 -O2"
printf '%s\n' "$COMMAND" > "$RUN/command.txt"
python -m benchmark.worker_phases.server \
  --command "$COMMAND" \
  --period 10 --health-timeout 3600 \
  --server-log "$RUN/server.log" --out "$RUN/phases.txt"
```

Wait for `SERVER READY`. In terminal 2, set the same `ARM`, `PYTHONPATH`,
runtime variables and `RUN`, then run the driver from the **fix** checkout:

```bash
python /absolute/path/to/mstar-244-fix/benchmark/transport/request_cleanup_gpu.py \
  --checkout "$ARM" --out "$RUN/clients" --model qwen
```

The driver uses the repo's existing client and runner. It runs concurrency
1/32/64 with 16 and 512 generated tokens, five warmup waves per cell, greedy
sampling, and at least eight measured waves (at least 64 requests). It forces
the text dataset and fixed requested decode lengths through the existing
Qwen client. Short outputs expose frequent teardown; longer outputs exercise
steady decode. `--dry-run` prints all commands without contacting a server.

Stop terminal 1 with Ctrl-C after the client finishes. The wrapper stops its
server's process group; wait for it to exit and confirm GPU memory is released
before starting the next arm. Do not stop only the wrapper with SIGKILL or use
`pkill -f`: that can leave workers behind. The wrapper refuses to start over
an already-healthy port.

## Streaming and multiple workers

Repeat the same four-arm sequence with Orpheus, which exercises actual tensor
traffic between workers. Change the server command to:

```bash
COMMAND="$PY -m mstar.cli.main serve orpheus --config configs/orpheus_pd_disaggregated.yaml --gpus 0,1 --port 8100 -O2"
printf '%s\n' "$COMMAND" > "$RUN/command.txt"
python -m benchmark.worker_phases.server \
  --command "$COMMAND" \
  --period 10 --health-timeout 3600 \
  --server-log "$RUN/server.log" --out "$RUN/phases.txt"
```

Use a new `RUN` for each server and `--model orpheus` in the client driver.
It uses Seed-TTS and concurrency 1/32/64. Compare audio seconds/s and RTF, as
well as completion counts; requests/s alone can reward shorter audio.

Also check lockstep scheduling using `configs/orpheus_tp2.yaml`, once with
`MSTAR_TP_ASYNC_SCHED=0` and once with `MSTAR_TP_ASYNC_SCHED=1`. Keep that value
identical between base and fix. Run the project's integration tests relevant
to the deployment's other models and normal mixed workloads too.

SHM arena is the primary same-host measurement. Do an untimed functional
smoke with `--tensor-comm-protocol TCP` and the same streaming config to
exercise Mooncake. TCP performance is too variable for acceptance numbers.
Use RDMA for a separate cross-host performance check if that is the deployment.

## Checks and artifacts

The driver rejects missing/failed requests, wrong concurrency, nonpositive
throughput and reuse of a results directory. Keep each run's `source.txt`,
`command.txt`, `server.log`, `phases.txt`, `clients/metadata.json`, `gpu.txt`, `packages.txt`,
`summary.json`, and every cell's `client.log` and `results.json`.
Tracked changes in a benchmark checkout are refused so the recorded commit
actually identifies the code being compared.

- All expected requests complete; no drain timeouts, read failures, missing
  tensors, hung TP ranks or growing transport memory across repeated waves.
- Compare the actual `aggregate.total_text_tokens` / audio output volume
  between arms and the same-code repeats. A content/length change invalidates
  a requests/s comparison even when requested token limits are the same.
- Compare text tokens/s, TTFT, ITL and median JCT; for speech compare audio
  seconds/s and RTF. Report the individual repeats, not only their average.
- In worker phases, compare the longest steady segments at the same actual
  batch size. Inspect `worker.main.teardown`, message processing, `iter_total`
  and their CPU spans. Only `worker.postprocess.event_sync` is the GPU wait;
  do not count `await_gpu` or other waits as CPU work or GPU execution time.
- A regression larger than the same-code variation leaves performance
  acceptance unresolved. Investigate it rather than declaring the CPU
  microbenchmark proof of unchanged serving performance.

Separately, in the fix checkout run the teardown regression and lockstep tests
with a freshly built extension:

```bash
OMP_NUM_THREADS=2 python -m pytest -q \
  test/modular/test_request_cleanup_race.py test/modular/test_teardown_barrier.py \
  test/modular/test_worker_drain.py test/modular/test_tp_async_teardown_order.py \
  test/modular/test_tp_async_lockstep_sim.py
```

The new race module is present only in the fix checkout. From the base
checkout, demonstrate the original failure using its absolute path:

```bash
PYTHONPATH="$PWD" OMP_NUM_THREADS=2 python -m pytest -q \
  /absolute/path/to/mstar-244-fix/test/modular/test_request_cleanup_race.py
```

The race test controls read timing with events and substitutes the native
transfer engine; it is not a real-weights or native RDMA reproduction.

## Automated two-A40 handoff

For the fork-only rental workflow, environment setup, serial A/B orchestration
and instructions for the agent on the GPU host, start with
[`a40/AGENT.md`](a40/AGENT.md). `a40/run.py --dry-run` prints the entire schedule
without starting a server or probing a GPU. These tools have been checked locally;
the actual A40 CUDA runs remain unexecuted.
