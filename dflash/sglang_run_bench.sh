export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export HF_HOME="/mnt/shared-storage-user/p1-shared/leihaodi/pretrain/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

cd /mnt/shared-storage-user/leihaodi/diffusion/dflash
source /mnt/shared-storage-user/p1-shared/leihaodi/miniconda3/bin/activate dflash-sglang-eval
EXPECTED_CONDA_ENV="dflash-sglang-eval"
if [[ "${CONDA_DEFAULT_ENV:-}" != "${EXPECTED_CONDA_ENV}" ]]; then
  echo "Error: expected conda env '${EXPECTED_CONDA_ENV}', but got '${CONDA_DEFAULT_ENV:-<none>}'." >&2
  exit 1
fi
# /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-30B-A3B-Thinking-2507
TARGET_MODEL="/mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B"

DRAFT_MODELS=(
  # "/mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-30b-a3b-dflash/epoch_4_step_200000"
  # "/mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash_data/epoch_4_step_240000"
  "/mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash_data/epoch_5_step_295000"
  "/mnt/shared-storage-gpfs2/p1-shared-2/leihaodi/opd-draft/qwen3-4b/16k_global_step_5000_draft"
  "/mnt/shared-storage-user/leihaodi/opd/verl/checkpoints/verl-dflash-opd/fsdp-draftmodel-0_lr-3e-4-decay-True-random_anchor-False/student-teacher-05-08/all-forward-kl-k3/train-apos-4000_code-5000_math-5000_gsm8k-2000_user_prompt-update-accumulation-steps/global_step_5000/draft_model"
)

# Optional: set to "--skip-base" to skip baseline and only benchmark DFLASH
SKIP_BASE="--skip-base"
# Optional: set to "--enable-think" for Qwen-style thinking prompts
ENABLE_THINK="--enable-think"

# math500:16 gsm8k:128 mbpp:128 humaneval:128
for i in "${!DRAFT_MODELS[@]}"; do
  DRAFT_MODEL="${DRAFT_MODELS[$i]}"

  # OUTPUT_CASE="./sglang_dflash_cases_a3b_16000_draft_${i}.jsonl"
  OUTPUT_MD="sglang_results_4b_8192_draft_${i}.md"

  echo "Running benchmark with draft model: ${DRAFT_MODEL}"

  python benchmark_sglang.py \
    --target-model "${TARGET_MODEL}" \
    --draft-model "${DRAFT_MODEL}" \
    --concurrencies 1 \
    --dataset-name aime24:30 math500:128 gsm8k:128 mbpp:128 humaneval:128 aime25:30 alpaca:128 mt-bench:80 mgsm_zh:32 swe-bench:128 \
    --attention-backends fa3 \
    --tp-size 4 \
    --max-new-tokens 8192 \
    ${ENABLE_THINK} \
    ${SKIP_BASE} \
    --output-md "${OUTPUT_MD}" \
    --mem-fraction-static 0.85
done

# OUTPUT_CASE="./sglang_dflash_cases_3.jsonl"
# python benchmark_sglang.py \
#   --target-model /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B \
#   --draft-model /mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash_data-instruct-bs4/epoch_10_step_246520 \
#   --concurrencies 1 \
#   --dataset-name math500 \
#   --attention-backends fa3 \
#   --tp-size 1 \
#   --max-new-tokens 4096 \
#   ${SKIP_BASE} \
#   ${OUTPUT_CASE:+--output-case "$OUTPUT_CASE"} \
#   ${ENABLE_THINK} \
#   --output-md sglang_results_3.md

#   --mamba-scheduler-strategy extra_buffer \