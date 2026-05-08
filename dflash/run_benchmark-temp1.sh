#!/usr/bin/env bash
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Optional: ./run_benchmark.sh --log-dir /path/to/logs
# If omitted, uses the default log_dir below.
source /mnt/shared-storage-user/p1-shared/leihaodi/miniconda3/bin/activate python312
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export HF_HOME="/mnt/shared-storage-user/p1-shared/leihaodi/pretrain/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  num_gpu=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
else
  num_gpu=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
cd /mnt/shared-storage-user/leihaodi/diffusion/dflash

log_dir="/mnt/shared-storage-user/leihaodi/diffusion/logs/specforge-filter"
while [ $# -gt 0 ]; do
  case "$1" in
    --log-dir)
      if [ -z "${2:-}" ]; then
        echo "usage: $0 [--log-dir PATH]" >&2
        exit 1
      fi
      log_dir="$2"
      shift 2
      ;;
    *)
      echo "unknown option: $1" >&2
      echo "usage: $0 [--log-dir PATH]" >&2
      exit 1
      ;;
  esac
done

TASKS=(
  # "gsm8k:128"
  # "aime24:30"
  "aime25:30"
  "math500:128"
  "mbpp:128"
  "humaneval:164"
  # "mt-bench:80"

  # "mgsm_zh:32"
  # "acp_app_bool:32"
  # "acp_app_gen:32"
  # "swe-bench:128"

  # "alpaca:128"
  # "livecodebench:128"
)

print_case=false
mkdir -p "$log_dir"
DRAFT_MODELS=(
  # "/mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash-eagle3loss-right/epoch_4_step_240000"
  # "/mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash-eagle3loss-right/epoch_5_step_260000"
  # "/mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash-eagle3loss-right/epoch_5_step_280000"
  # "/mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash_ultra/epoch_7_step_360000"

  "/mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash-filter/epoch_6_step_160000"
  # "/mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash-filter/epoch_5_step_140000"
  "/mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash-filter/epoch_9_step_240000"
  # "/mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash_data/epoch_4_step_200000"

  # "/mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-test_dataset/epoch_50_step_4000"
  # "/mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash_data/epoch_5_step_295000"
  
  # "/mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash-eagle3loss-right/epoch_8_step_420000"
  # "/mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash-eagle3loss-right/epoch_8_step_400000"
  # "/mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash_data/epoch_5_step_280000"


  # "/mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash_data-instruct-bs4/epoch_10_step_246520"
  # "/mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash_data-instruct-bs4/epoch_9_step_240000"
  # "/mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash_data-instruct-bs4/epoch_6_step_150000"
  # "/mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash_data-instruct-bs4/epoch_7_step_190000"


  # "/mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash_data_plus-instruct-bs4/epoch_9_step_400000"
  # "/mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash_data_plus-instruct-bs4/epoch_4_step_200000"

  # "/mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash_data-instruct-bs4/epoch_9_step_240000"
  # "/mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash_data-instruct-bs4/epoch_9_step_230000"
  # "/mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash_data-instruct-bs4/epoch_8_step_220000"

  # "/mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash-gemma/epoch_9_step_480000"
  # "/mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash-gemma/epoch_10_step_493310"
  # "/mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash-gemma/epoch_5_step_280000"
  # "/mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash-gemma/epoch_3_step_160000"
)


for draft_path in "${DRAFT_MODELS[@]}"; do
  draft_group=$(basename "$(dirname "$draft_path")")
  draft_group=${draft_group#qwen3-4b-}
  draft_step=$(basename "$draft_path")
  draft_tag="${draft_group}_${draft_step}"
  # for thinking_mode in "off" "on"; do
  for thinking_mode in "on"; do
    if [ "$thinking_mode" = "on" ]; then
      THINKING_ARGS=(--enable-thinking)
    else
      THINKING_ARGS=()
    fi
    if [ "$print_case" = "true" ]; then
      CASE_ARGS=(--case)
    else
      CASE_ARGS=()
    fi

    echo "CASE_ARGS: ${CASE_ARGS[@]}"
    echo "THINKING_ARGS: ${THINKING_ARGS[@]}"
    sleep 1

    log_name="3-${draft_tag}-${thinking_mode}.log"
    log_file="${log_dir}/${log_name}"
    # : > "$log_file"

    echo "############################################################"
    echo "Draft model: $draft_path"
    echo "Thinking mode: $thinking_mode"
    echo "Log file: $log_file"
    echo "############################################################"

    for task in "${TASKS[@]}"; do
      IFS=':' read -r DATASET_NAME MAX_SAMPLES <<< "$task"

      echo "========================================================"
      echo "Running Benchmark: $DATASET_NAME with $MAX_SAMPLES samples (draft: $draft_tag, thinking: $thinking_mode)"
      echo "========================================================"

      torchrun \
        --nproc_per_node="${num_gpu}" \
        --master_port=29600 \
        ./benchmark.py \
        --dataset "$DATASET_NAME" \
        --max-samples "$MAX_SAMPLES" \
        --model-name-or-path /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B \
        --draft-name-or-path "$draft_path" \
        --max-new-tokens 2048 \
        --block-size 16 \
        --temperature 0.0 \
        --skip-base \
        "${CASE_ARGS[@]}" \
        "${THINKING_ARGS[@]}" \
        | tee -a "$log_file"
        # --save-acc-len "${log_dir}/8192-${draft_tag}-${DATASET_NAME}.csv" | tee -a "$log_file"
        

    done
  done
done

# for draft_path in "${DRAFT_MODELS[@]}"; do
#   draft_group=$(basename "$(dirname "$draft_path")")
#   draft_group=${draft_group#qwen3-4b-}
#   draft_step=$(basename "$draft_path")
#   draft_tag="${draft_group}_${draft_step}"
#   # for thinking_mode in "off" "on"; do
#   for thinking_mode in "on"; do
#     if [ "$thinking_mode" = "on" ]; then
#       THINKING_ARGS=(--enable-thinking)
#     else
#       THINKING_ARGS=()
#     fi
#     if [ "$print_case" = "true" ]; then
#       CASE_ARGS=(--case)
#     else
#       CASE_ARGS=()
#     fi

#     echo "CASE_ARGS: ${CASE_ARGS[@]}"
#     echo "THINKING_ARGS: ${THINKING_ARGS[@]}"
#     sleep 1

#     log_name="2-${draft_tag}-${thinking_mode}.log"
#     log_file="${log_dir}/${log_name}"
#     # : > "$log_file"

#     echo "############################################################"
#     echo "Draft model: $draft_path"
#     echo "Thinking mode: $thinking_mode"
#     echo "Log file: $log_file"
#     echo "############################################################"

#     for task in "${TASKS[@]}"; do
#       IFS=':' read -r DATASET_NAME MAX_SAMPLES <<< "$task"

#       echo "========================================================"
#       echo "Running Benchmark: $DATASET_NAME with $MAX_SAMPLES samples (draft: $draft_tag, thinking: $thinking_mode)"
#       echo "========================================================"

#       torchrun \
#         --nproc_per_node="${num_gpu}" \
#         --master_port=29600 \
#         ./benchmark.py \
#         --dataset "$DATASET_NAME" \
#         --max-samples "$MAX_SAMPLES" \
#         --model-name-or-path /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B \
#         --draft-name-or-path "$draft_path" \
#         --max-new-tokens 8192 \
#         --block-size 16 \
#         --temperature 0.0 \
#         --skip-base \
#         "${CASE_ARGS[@]}" \
#         "${THINKING_ARGS[@]}" \
#         | tee -a "$log_file"
#         # --save-acc-len "${log_dir}/8192-${draft_tag}-${DATASET_NAME}.csv" | tee -a "$log_file"
        

#     done
#   done
# done

python /mnt/shared-storage-user/leihaodi/gpu_stress_test.py
# --enable-thinking \
    # --skip-base \
    # --block-size 8 \