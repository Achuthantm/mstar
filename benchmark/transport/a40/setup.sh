#!/usr/bin/env bash
# Run from a clean, committed fork checkout on the rented Linux host.
set -euo pipefail
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
ROOT=${1:?Usage: bash setup.sh /workspace/cleanup-bench}
mkdir -p "$ROOT"
ROOT=$(realpath "$ROOT")
BASE=9f84f18cd7e1b4f3851d0bde85b8da45cabec06f
FIX=$(git -C "$REPO" rev-parse HEAD)
for tool in uv rustup nvcc nvidia-smi ss setsid curl cmake pkg-config; do
    command -v "$tool" >/dev/null || { echo "Missing prerequisite: $tool" >&2; exit 1; }
done
[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ] || {
    echo 'Commit tracked changes before setup.' >&2; exit 1;
}
[ "$FIX" != "$BASE" ] || { echo 'Checkout the committed fix first.' >&2; exit 1; }
nvcc --version | grep -q 'release 12.8' || { echo 'This setup targets CUDA toolkit 12.8.' >&2; exit 1; }
# Refuse reuse: preserve the exact checkouts/environment of every experiment.
for target in base fix venv; do
    [ ! -e "$ROOT/$target" ] || { echo "Already exists: $ROOT/$target" >&2; exit 1; }
done
git -C "$REPO" worktree add --detach "$ROOT/base" "$BASE"
git -C "$REPO" worktree add --detach "$ROOT/fix" "$FIX"
uv venv --python 3.12 --seed "$ROOT/venv"
source "$ROOT/venv/bin/activate"
export HF_HOME="$ROOT/cache/huggingface"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/inductor"
export TRITON_CACHE_DIR="$ROOT/cache/triton"
mkdir -p "$HF_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
uv pip install --torch-backend=cu128 'torch==2.10.*' 'torchvision==0.25.*' 'torchaudio==2.10.*' \
    -e "$ROOT/fix[orpheus,audio,dev]" ninja Pillow av maturin
(cd "$ROOT/fix" && maturin develop --release -m rust/Cargo.toml)
python -m pip freeze > "$ROOT/packages.txt"
printf '%s\n' "$BASE" > "$ROOT/base.sha"
printf '%s\n' "$FIX" > "$ROOT/fix.sha"
echo "Setup complete. Activate $ROOT/venv/bin/activate and follow AGENT.md."
