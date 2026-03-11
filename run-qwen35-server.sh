#!/usr/bin/env bash
set -euo pipefail

export FLASHINFER_DISABLE_VERSION_CHECK=1
cd /mnt/shared-storage-user/leihaodi/imo/diffusion

MODEL_PATH="/mnt/shared-storage-user/p1-shared/Qwen/Qwen3.5-4B"
LOG_DIR="/mnt/shared-storage-user/leihaodi/imo/diffusion/logs/qwen3.5"
mkdir -p "${LOG_DIR}"

# 1) baseline: 不启用 MTP (port 8001)
CUDA_VISIBLE_DEVICES=0 nohup python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --port 8001 \
  --tp-size 1 \
  --mem-fraction-static 0.8 \
  --context-length 262144 \
  --reasoning-parser qwen3 \
  --speculative-algo NEXTN \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  > "${LOG_DIR}/sglang_qwen35_mtp_516_gpu0_port8001.log" 2>&1 &

# 2) MTP (NEXTN): 启用投机解码 (port 8000)
CUDA_VISIBLE_DEVICES=1 nohup python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --port 8000 \
  --tp-size 1 \
  --mem-fraction-static 0.8 \
  --context-length 262144 \
  --reasoning-parser qwen3 \
  --speculative-algo NEXTN \
  --speculative-num-steps 7 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 8 \
  > "${LOG_DIR}/sglang_qwen35_mtp_718_gpu1_port8000.log" 2>&1 &

echo "Started 2 Qwen3.5 servers:"
echo "  baseline(no MTP) -> gpu0, port 8001, log ${LOG_DIR}/sglang_qwen35_baseline_gpu0_port8001.log"
echo "  mtp(NEXTN)       -> gpu1, port 8000, log ${LOG_DIR}/sglang_qwen35_mtp_gpu1_port8000.log"

CUDA_VISIBLE_DEVICES=2 nohup python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --port 8002 \
  --tp-size 1 \
  --mem-fraction-static 0.8 \
  --context-length 262144 \
  --reasoning-parser qwen3 \
  > "${LOG_DIR}/sglang_qwen35_baseline_gpu2_port8002.log" 2>&1 &
