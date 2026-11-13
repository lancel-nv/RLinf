#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: bash examples/embodiment/reproduce_expo/train.sh RUN_DIR" >&2
    exit 2
fi
REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
RUN_DIR=$(realpath -- "$1")
[[ -f "$RUN_DIR/train.yaml" ]] || { echo "Run prepare.py first" >&2; exit 2; }
[[ ! -e "$RUN_DIR/train.exitcode" && ! -d "$RUN_DIR/results" ]] || {
    echo "Use a fresh run directory; this command starts a new experiment" >&2
    exit 2
}
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export EMBODIED_PATH="$REPO_DIR/examples/embodiment"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$RUN_DIR/cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export RLINF_NODE_RANK=0 RAY_USAGE_STATS_ENABLED=0
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/rlinf-expo-$$}"
export JAX_PLATFORMS=cpu TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
unset SAPIEN_DISABLE_RAY_TRACING RAY_ADDRESS
cd "$REPO_DIR"
trap 'result=$?; printf "%s\n" "$result" > "$RUN_DIR/train.exitcode"' EXIT
python examples/embodiment/train_async.py --config-path "$RUN_DIR" --config-name train
