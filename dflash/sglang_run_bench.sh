#!/usr/bin/env bash
set -eo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash sglang_run_bench.sh --config-yaml PATH [--dry-run]
  bash sglang_run_bench.sh PATH [--dry-run]

The YAML file should contain a jobs list. Example:

jobs:
  - gpu: 0
    target_model: /path/to/target
    draft_model: /path/to/draft
    output_md: /path/to/result.md
    dataset_name: ["aime24:30", "math500:128"]
    concurrency_num: 1

  - gpu: 1
    target_model: /path/to/target
    output_md: /path/to/baseline.md
    dataset_name: "gsm8k:128"
    concurrency_num: "1,2,4"
EOF
}

# 直接在程序运行最后直接指定

# CONFIG_YAML=""
# DRY_RUN_ARGS=()

# while [[ $# -gt 0 ]]; do
#   case "$1" in
#     --config-yaml|--config)
#       if [[ $# -lt 2 ]]; then
#         echo "Error: $1 requires a path." >&2
#         exit 1
#       fi
#       CONFIG_YAML="$2"
#       shift 2
#       ;;
#     --dry-run)
#       DRY_RUN_ARGS=(--dry-run)
#       shift
#       ;;
#     -h|--help)
#       usage
#       exit 0
#       ;;
#     --*)
#       echo "Error: unknown option: $1" >&2
#       usage >&2
#       exit 1
#       ;;
#     *)
#       if [[ -n "${CONFIG_YAML}" ]]; then
#         echo "Error: multiple config paths provided: ${CONFIG_YAML} and $1" >&2
#         exit 1
#       fi
#       CONFIG_YAML="$1"
#       shift
#       ;;
#   esac
# done

# if [[ -z "${CONFIG_YAML}" ]]; then
#   echo "Error: missing --config-yaml PATH." >&2
#   usage >&2
#   exit 1
# fi

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export HF_HOME="/mnt/shared-storage-user/p1-shared/leihaodi/pretrain/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

source /mnt/shared-storage-user/p1-shared/leihaodi/miniconda3/bin/activate dflash-sglang-eval
EXPECTED_CONDA_ENV="dflash-sglang-eval"
if [[ "${CONDA_DEFAULT_ENV:-}" != "${EXPECTED_CONDA_ENV}" ]]; then
  echo "Error: expected conda env '${EXPECTED_CONDA_ENV}', but got '${CONDA_DEFAULT_ENV:-<none>}'." >&2
  exit 1
fi

python launch_sglang_bench_jobs.py \
  --config-yaml "/mnt/shared-storage-user/leihaodi/diffusion/dflash/eval_config/four_gpu_final.yaml"

