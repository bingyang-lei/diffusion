export FLASHINFER_DISABLE_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES=2,3
cd /mnt/shared-storage-user/leihaodi/imo/diffusion

python -m sglang.launch_server \
    --model-path /mnt/shared-storage-user/p1-shared/Qwen/model--openai-mirror--gpt-oss-20b \
    --tp-size 2 \
    --dtype bfloat16 \
    --attention-backend fa3 \
    --mem-fraction-static 0.6 \
    --trust-remote-code \
    --port 30001 \
    # --speculative-algorithm DFLASH \
    # --speculative-draft-model-path /mnt/shared-storage-user/p1-shared/leihaodi/spec_decode/draft-model/dflash-gpt-oss-20b \