# Two-A40 benchmark handoff

Task: validate request cleanup correctness and normal-case serving overhead. Do
not claim completion from the CPU microbenchmark alone. The GPU runs have not
been executed on the development Mac.

## Repository and notification boundary

Use only `https://github.com/Achuthantm/mstar.git`, branch
`request-cleanup-a40`. Do not open a PR, create/comment on issues, mention GitHub
users, or push to another repository. Do not publish results without the user's
instruction. Read the checkout's AGENTS.md and benchmarking skill before running.
The two-GPU Pod rental is managed by the user; do not provision paid resources.

## Pod prerequisites

Use two full A40 48GB GPUs in one Linux x86_64 Pod, ideally 16+ vCPUs and 64+ GB
RAM. Allocate 150–200GB persistent workspace and at least 16GB `/dev/shm`.
Use a CUDA **12.8 development** image, including `nvcc`, with a compatible NVIDIA
driver. Inspect `nvidia-smi`, `nvidia-smi topo -m`, `nvcc --version`, `df -h
/dev/shm`, `free -h` and `lscpu` first. A40 hardware and kernels are not yet
validated by these scripts: fail visibly and diagnose incompatibilities rather
than silently disabling optimization or lowering concurrency on one arm.

Install missing OS prerequisites (Debian/Ubuntu, root shell):

```bash
apt-get update
apt-get install -y git curl build-essential cmake pkg-config libssl-dev \
  libsndfile1 ffmpeg iproute2 procps util-linux tmux
```

Install `uv` and `rustup` using their official installers if absent. The repository
pins its Rust toolchain; allow rustup to install that version. Setup checks the
CUDA toolkit before downloading Python dependencies. It installs Python 3.12,
CUDA-12.8 PyTorch 2.10 and the selected model extras into one shared environment,
and builds the current Rust extension in release mode. Do not upgrade packages
between arms. Keep the environment and caches on persistent storage.

## Prepare the two versions

```bash
cd /workspace
git clone --branch request-cleanup-a40 --single-branch https://github.com/Achuthantm/mstar.git mstar-bench
cd mstar-bench
bash benchmark/transport/a40/setup.sh /workspace/cleanup-bench
export ROOT=/workspace/cleanup-bench
source "$ROOT/venv/bin/activate"
cd "$ROOT/fix"
export HF_HOME="$ROOT/cache/huggingface"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/inductor"
export TRITON_CACHE_DIR="$ROOT/cache/triton"
```

Setup creates detached `base` and `fix` worktrees. It refuses existing worktrees
or an existing environment; after a failed setup, inspect the failure and use a
new root or explicitly repair the partial setup. It never deletes prior results.
The base SHA is pinned in setup.sh; fix.sha identifies the checked-out commit.
Authenticate to Hugging Face if a checkpoint or dataset requires access. Cache
these checkpoints before timing (this may require accepting their model terms):

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ('Qwen/Qwen3.5-0.8B', 'canopylabs/orpheus-3b-0.1-pretrained', 'hubertsiuzdak/snac_24khz'):
    snapshot_download(repo)
PY
```

The clients load their datasets and the server config's startup inputs on demand.
The five warmup waves per cell must absorb compilation/dataset cold starts.
Check both arms become stable; repeat if there is a first-run trend.

## Check before spending time on the full suite

```bash
python -m benchmark.transport.a40.run --root "$ROOT" --out "$ROOT/results/full-1" --dry-run
OMP_NUM_THREADS=2 python -m pytest -q \
  test/modular/test_request_cleanup_race.py test/modular/test_teardown_barrier.py \
  test/modular/test_worker_drain.py test/modular/test_tp_async_teardown_order.py \
  test/modular/test_tp_async_lockstep_sim.py test/modular/test_request_cleanup_benchmark.py
python -c 'import torch, mstar_rust; print(torch.__version__, torch.version.cuda, mstar_rust.__file__); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'
```

Run the race module from the base directory with `PYTHONPATH="$ROOT/base"` and
its absolute file path in the fix checkout to demonstrate the original failure.
It substitutes the transfer engine, so passing it is not a native GPU transfer
check. Save test stdout/stderr and exit statuses, including the expected baseline
failures. Do not modify either committed worktree during measurements.

## Execute serial A/B runs

Use tmux so an SSH interruption does not terminate a long run. Run one suite at
a time on otherwise idle GPUs. Start with Qwen; inspect logs before proceeding.
All commands below are from `$ROOT/fix`, with the venv active.

```bash
python -m benchmark.transport.a40.run --root "$ROOT" --out "$ROOT/results/qwen-1" --suite qwen
python -m benchmark.transport.a40.run --root "$ROOT" --out "$ROOT/results/audio-1" --suite streaming
python -m benchmark.transport.a40.run --root "$ROOT" --out "$ROOT/results/tp-1" --suite tp-sync tp-async
python -m benchmark.transport.a40.run --root "$ROOT" --out "$ROOT/results/tcp-1" --suite tcp
```

Alternatively, `--suite all` runs all five scenarios. Each timing scenario uses
fresh servers in base/fix/fix/base order with concurrency 1/32/64; Qwen requests
16 and 512 output tokens. TCP runs just two requests on each arm and is untimed.
The runner uses the existing server.sh lifecycle helper, health/port ownership
checks, a 60-minute startup timeout, and stop in a finally block. Ctrl-C/TERM
enters cleanup. SIGKILL cannot run cleanup: inspect the saved server-state and
use server.sh to stop only this run's server, then check GPU memory and port.
Do not use `pkill -f`, start two arms simultaneously, or reuse a results directory.

The default uses Rust graph/ZMQ and the SHM arena. Also check Python graph/ZMQ
behavior (SHM arena still enabled) in a separate, untimed functional run:

```bash
python -m benchmark.transport.a40.run --root "$ROOT" --out "$ROOT/results/python-smoke-1" \
  --suite streaming tcp --runtime python --concurrency 1
```

This Python streaming command still runs the full existing client workload at
concurrency 1; it is a functional check, not a replacement for the Rust sweep.
If TCP or TP fails because the Pod's GPU topology/container lacks support,
report that test as blocked with the exact error. Do not count a skip as a pass.
No RDMA/cross-host claim follows from this single-host rental.

## Review results and return evidence

Every run saves lifecycle/server/client logs, phase tables, GPU utilization,
commit/source metadata and client result JSON. `comparison.json` retains every
repeat plus request-throughput/JCT median deltas. It is descriptive and does not
automatically decide acceptance. TCP has no timing comparison. `PASS` means the
clients completed; it does not certify the logs or performance.

Check completion counts and failures, drain timeouts, missing/read-error tensors,
TP hangs, and GPU/host/transport memory growth over successive waves. Save
`nvidia-smi` before and after the suite. Inspect the full client.log metric tables:
text tokens/s, TTFT/ITL and audio seconds/s/RTF, alongside output token/audio
volumes. Requests/s alone is insufficient for audio. Compare same-code repeats
before interpreting a change. A regression larger than same-code variation must
be investigated. Do not accept a smaller concurrency as evidence for 32/64.

Inspect steady worker-phase segments at the same actual batch size. Only
`worker.postprocess.event_sync` is the GPU wait; do not sum waits as CPU work.
Preserve stdout/stderr on failures and partial output directories. Diagnose
without changing both versions or adding speculative runtime mechanisms.
The inherited forced-drain timeout is a known safety limit; do not claim the
change proves safety for an indefinitely stalled native transfer.

Return: exact commits, GPU/CPU/RAM/topology, driver/toolkit/torch/native versions,
commands and exit statuses, tests run/passed/failed/skipped, per-repeat metrics,
log errors, and all result artifacts. Before deleting the Pod, copy the results,
packages.txt and base.sha/fix.sha to the user's machine. The Pod volume is lost
when the Pod is deleted. Do not push or open a PR as part of benchmarking.
