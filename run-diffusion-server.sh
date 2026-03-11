# # sglang-path: /mnt/shared-storage-gpfs2/p1-shared-2/leihaodi/sglang
# export FLASHINFER_DISABLE_VERSION_CHECK=1
# export CUDA_VISIBLE_DEVICES=0

# # import os
# # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# python -m sglang.launch_server \
#     --model-path /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B \
#     --tp-size 1 \
#     --dtype bfloat16 \
#     --attention-backend fa3 \
#     --mem-fraction-static 0.75 \
#     --trust-remote-code \
#     --speculative-algorithm DFLASH \
#     --speculative-draft-model-path /mnt/shared-storage-user/p1-shared/leihaodi/spec_decode/draft-model/dflash-16-qwen3-4b

#!/usr/bin/env bash
set -euo pipefail

export FLASHINFER_DISABLE_VERSION_CHECK=1
cd /mnt/shared-storage-user/leihaodi/diffusion

MODEL_PATH="/mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B"
DFLASH_DRAFT_PATH="/mnt/shared-storage-user/p1-shared/leihaodi/spec_decode/draft-model/dflash-qwen3-4b" 
# EAGLE3_DRAFT_PATH="/mnt/shared-storage-user/p1-shared/leihaodi/spec_decode/draft-model/Qwen3-4B_eagle3-AngelSlim"

LOG_DIR="/mnt/shared-storage-user/leihaodi/diffusion/dflash/logs"
mkdir -p "${LOG_DIR}"

# CUDA_VISIBLE_DEVICES=4 nohup python -m sglang.launch_server \
#   --model-path /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-Coder-30B-A3B-Instruct \
#   --tp-size 1 \
#   --dtype bfloat16 \
#   --attention-backend fa3 \
#   --mem-fraction-static 0.75 \
#   --trust-remote-code \
#   --port 30003 \
#   > "${LOG_DIR}/sglang_baseline_gpu3_port30000.log" 2>&1 &
# 1) baseline: 不使用投机解码 (GPU 0, port 30000)
# CUDA_VISIBLE_DEVICES=2 nohup python -m sglang.launch_server \
#   --model-path "${MODEL_PATH}" \
#   --tp-size 1 \
#   --dtype bfloat16 \
#   --attention-backend fa3 \
#   --mem-fraction-static 0.75 \
#   --trust-remote-code \
#   --port 30000 \
#   > "${LOG_DIR}/sglang_baseline_gpu0_port30000.log" 2>&1 &

# 2) DFLASH: 当前配置 (GPU 1, port 30001)
CUDA_VISIBLE_DEVICES=3 nohup python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --tp-size 1 \
  --dtype bfloat16 \
  --attention-backend fa3 \
  --mem-fraction-static 0.75 \
  --trust-remote-code \
  --port 30001 \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "${DFLASH_DRAFT_PATH}" \
  > "${LOG_DIR}/sglang_dflash_gpu1_port30001.log" 2>&1 &

# # 3) EAGLE3: 指定 draft model (GPU 2, port 30002)
# CUDA_VISIBLE_DEVICES=7 nohup python -m sglang.launch_server \
#   --model-path "${MODEL_PATH}" \
#   --tp-size 1 \
#   --dtype bfloat16 \
#   --attention-backend fa3 \
#   --mem-fraction-static 0.75 \
#   --trust-remote-code \
#   --port 30002 \
#   --speculative-algorithm EAGLE3 \
#   --speculative-draft-model-path "${EAGLE3_DRAFT_PATH}" \
#   > "${LOG_DIR}/sglang_eagle3_gpu2_port30002.log" 2>&1 &

echo "Started 3 servers:"
echo "  baseline -> gpu0, port 30000, log ${LOG_DIR}/sglang_baseline_gpu0_port30000.log"
echo "  dflash   -> gpu1, port 30001, log ${LOG_DIR}/sglang_dflash_gpu1_port30001.log"
# echo "  eagle3   -> gpu2, port 30002, log ${LOG_DIR}/sglang_eagle3_gpu2_port30002.log"