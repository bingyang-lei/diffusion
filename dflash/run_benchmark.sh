#!/usr/bin/env bash
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Optional: ./run_benchmark.sh --log-dir /path/to/logs
# If omitted, uses the default log_dir below.
bash /mnt/shared-storage-user/leihaodi/diffusion/dflash/sglang_run_bench.sh
pkill -9 python
pkill -9 sglang
sleep 1
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
log_dir="/mnt/shared-storage-user/leihaodi/diffusion/logs/verl-opd-mathcode-16k-ablation"
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
  "gsm8k:128"
  "aime24:30"
  "aime25:30"
  "math500:128"
  "mbpp:128"
  "humaneval:164"
  "mt-bench:80"
  "mgsm_zh:32"
  "swe-bench:128"
  "alpaca:128"
  # "livecodebench:128"
    # "acp_app_bool:32"
  # "acp_app_gen:32"
)

print_case=false
mkdir -p "$log_dir"
DRAFT_MODELS=(
  "/mnt/shared-storage-user/leihaodi/opd/verl/checkpoints/verl-dflash-opd/fsdp-draftmodel-0_lr-3e-4-decay-True-random_anchor-False/student-teacher-05-08/all-reverse-kl-k3/train-apos-4000_code-5000_math-5000_gsm8k-2000_user_prompt-update-accumulation-steps/global_step_2500/draft_model"
  "/mnt/shared-storage-user/leihaodi/opd/verl/checkpoints/verl-dflash-opd/fsdp-draftmodel-0_lr-3e-4-decay-True-random_anchor-False/student-teacher-05-08/all-reverse-kl-k3/train-apos-4000_code-5000_math-5000_gsm8k-2000_user_prompt-update-accumulation-steps/global_step_5000/draft_model"
  "/mnt/shared-storage-user/leihaodi/opd/verl/checkpoints/verl-dflash-opd/fsdp-draftmodel-0_lr-3e-4-decay-True-random_anchor-False/student-teacher-05-08/all-forward-kl-k3/train-apos-4000_code-5000_math-5000_gsm8k-2000_user_prompt-update-accumulation-steps/global_step_2500/draft_model"
  "/mnt/shared-storage-user/leihaodi/opd/verl/checkpoints/verl-dflash-opd/fsdp-draftmodel-0_lr-3e-4-decay-True-random_anchor-False/student-teacher-05-08/all-forward-kl-k3/train-apos-4000_code-5000_math-5000_gsm8k-2000_user_prompt-update-accumulation-steps/global_step_5000/draft_model"
  # "/mnt/shared-storage-user/leihaodi/opd/verl/checkpoints/verl-dflash-opd/fsdp-draftmodel-1900/student-teacher-05-06/loss-forward-kl-k3/train-apos-4000_code-5000_math-5000_gsm8k-2000_user_prompt-update-accumulation-steps/global_step_5000/draft_model"
  # "/mnt/shared-storage-user/leihaodi/opd/verl/checkpoints/verl-dflash-opd/fsdp-draftmodel-1900/student-teacher-05-06/loss-forward-kl-k3/train-apos-4000_code-5000_math-5000_gsm8k-2000_user_prompt-update-accumulation-steps/global_step_4500/draft_model"
  # "/mnt/shared-storage-user/leihaodi/opd/verl/checkpoints/verl-dflash-opd/fsdp-draftmodel-1900/student-teacher-05-06/loss-forward-kl-k3/train-apos-4000_code-5000_math-5000_gsm8k-2000_user_prompt-update-accumulation-steps/global_step_4000/draft_model"
  # "/mnt/shared-storage-user/leihaodi/opd/verl/checkpoints/verl-dflash-opd/fsdp-draftmodel-1900/student-teacher-05-06/loss-forward-kl-k3/train-apos-4000_code-5000_math-5000_gsm8k-2000_user_prompt-update-accumulation-steps/global_step_3500/draft_model"
  # "/mnt/shared-storage-user/leihaodi/opd/verl/checkpoints/verl-dflash-opd/fsdp/student-teacher-05-06/loss-forward-kl-k3/train-apos-4000_code-5000_math-5000_gsm8k-2000_user_prompt-update-accumulation-steps/global_step_3500/draft_model"

  # "/mnt/shared-storage-user/leihaodi/opd/verl/checkpoints/verl-dflash-opd/fsdp-draftmodel-1900/student-teacher-05-06/loss-forward-kl-k3/train-apos-4000_code-5000_math-5000_gsm8k-2000_user_prompt-update-accumulation-steps/global_step_2500/draft_model"
  # "/mnt/shared-storage-user/leihaodi/opd/verl/checkpoints/verl-dflash-opd/fsdp/student-teacher-05-06/loss-forward-kl-k3/train-apos-4000_code-5000_math-5000_gsm8k-2000_user_prompt-update-accumulation-steps/global_step_2500/draft_model"
  # "/mnt/shared-storage-user/leihaodi/opd/verl/checkpoints/verl-dflash-opd/fsdp-draftmodel-1900/student-teacher-05-06/loss-forward-kl-k3/train-apos-4000_code-5000_math-5000_gsm8k-2000_user_prompt-update-accumulation-steps/global_step_2000/draft_model"
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
    echo "Draft model: $draft_path" | tee -a "$log_file"
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
        --max-new-tokens 8192 \
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

DRAFT_MODELS=(
  "/mnt/shared-storage-gpfs2/p1-shared-2/leihaodi/opd-draft/qwen3-4b/baseline"
  "/mnt/shared-storage-gpfs2/p1-shared-2/leihaodi/opd-draft/qwen3-4b/16k_global_step_5000_draft"
  "/mnt/shared-storage-user/leihaodi/opd/verl/checkpoints/verl-dflash-opd/fsdp-draftmodel-0_lr-3e-4-decay-True-random_anchor-False/student-teacher-05-08/all-forward-kl-k3/train-apos-4000_code-5000_math-5000_gsm8k-2000_user_prompt-update-accumulation-steps/global_step_2500/draft_model"
  "/mnt/shared-storage-user/leihaodi/opd/verl/checkpoints/verl-dflash-opd/fsdp-draftmodel-0_lr-3e-4-decay-True-random_anchor-False/student-teacher-05-08/all-forward-kl-k3/train-apos-4000_code-5000_math-5000_gsm8k-2000_user_prompt-update-accumulation-steps/global_step_5000/draft_model"
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
    echo "Draft model: $draft_path" | tee -a "$log_file"
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
        --max-new-tokens 8192 \
        --block-size 16 \
        --temperature 1.0 \
        --skip-base \
        "${CASE_ARGS[@]}" \
        "${THINKING_ARGS[@]}" \
        | tee -a "$log_file"
        # --save-acc-len "${log_dir}/8192-${draft_tag}-${DATASET_NAME}.csv" | tee -a "$log_file"
        

    done
  done
done

python /mnt/shared-storage-user/leihaodi/gpu_stress_test.py
# --enable-thinking \
    # --skip-base \
    # --block-size 8 \

