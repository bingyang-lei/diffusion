export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export HF_HOME="/mnt/shared-storage-user/p1-shared/leihaodi/pretrain/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

# /mnt/shared-storage-user/p1-shared/Qwen/
cd /mnt/shared-storage-user/leihaodi/diffusion/dflash

# Optional: set to "--skip-base" to skip baseline and only benchmark DFLASH
SKIP_BASE="--skip-base"
# Optional: path to write DFLASH completions (JSONL); leave empty to disable
OUTPUT_CASE="./sglang_dflash_cases_3.5_16000.jsonl"
# Optional: set to "--enable-think" for Qwen-style thinking prompts
ENABLE_THINK="--enable-think"
# /mnt/shared-storage-gpfs2/p1-shared-2/leihaodi/draft-model/dflash-qwen3.5-4b
# /mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash_data/epoch_5_step_295000
python benchmark_sglang.py \
  --target-model /mnt/shared-storage-user/p1-shared/Qwen/Qwen3.5-4B \
  --draft-model /mnt/shared-storage-gpfs2/p1-shared-2/leihaodi/draft-model/dflash-qwen3.5-4b \
  --concurrencies 1 \
  --dataset-name math500:7 \
  --attention-backends fa3 \
  --tp-size 1 \
  --max-new-tokens 16000 \
  ${OUTPUT_CASE:+--output-case "$OUTPUT_CASE"} \
  ${ENABLE_THINK} \
  ${SKIP_BASE} \
  --mamba-scheduler-strategy extra_buffer \
  --output-md sglang_results_3.5_16000.md

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