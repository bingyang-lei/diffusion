CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
    --model-path /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path /mnt/shared-storage-user/leihaodi/opd/verl/checkpoints/verl-dflash-opd/fsdp/student-teacher/loss-k3/train-8w_plus_test_user_prompt-update-accumulation-steps/global_step_1500/draft_model \
    --port 30000 \
    --tp-size 1 \
    --dtype bfloat16 \
    --attention-backend fa3 \
    --mem-fraction-static 0.75 \
    --trust-remote-code &

# CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
#     --model-path /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B \
#     --speculative-algorithm DFLASH \
#     --speculative-draft-model-path /mnt/shared-storage-user/leihaodi/opd/verl/checkpoints/verl-dflash-opd/fsdp/student-teacher/loss-k3/train-8w_plus_test_user_prompt/global_step_1200/draft_model \
#     --port 30001 \
#     --tp-size 1 \
#     --dtype bfloat16 \
#     --attention-backend fa3 \
#     --mem-fraction-static 0.75 \
#     --trust-remote-code &

# CUDA_VISIBLE_DEVICES=2 python -m sglang.launch_server \
#     --model-path /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B \
#     --speculative-algorithm DFLASH \
#     --speculative-draft-model-path /mnt/shared-storage-user/leihaodi/opd/verl/checkpoints/verl-dflash-opd/fsdp/student-teacher/loss-k3/train-merged_math4w_code4w_user_prompt/global_step_100/draft_model \
#     --port 30002 \
#     --tp-size 1 \
#     --dtype bfloat16 \
#     --attention-backend fa3 \
#     --mem-fraction-static 0.75 \
#     --trust-remote-code &

# CUDA_VISIBLE_DEVICES=3 python -m sglang.launch_server \
#     --model-path /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B \
#     --speculative-algorithm DFLASH \
#     --speculative-draft-model-path /mnt/shared-storage-user/leihaodi/opd/verl/checkpoints/verl-dflash-opd/fsdp/student-teacher/loss-k3/train-merged_math4w_code4w_user_prompt/global_step_800/draft_model \
#     --port 30003 \
#     --tp-size 1 \
#     --dtype bfloat16 \
#     --attention-backend fa3 \
#     --mem-fraction-static 0.75 \
#     --trust-remote-code &

# CUDA_VISIBLE_DEVICES=4 python -m sglang.launch_server \
#     --model-path /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B \
#     --speculative-algorithm DFLASH \
#     --speculative-draft-model-path /mnt/shared-storage-user/leihaodi/opd/verl/checkpoints/verl-dflash-opd/fsdp/student-teacher/loss-k3/train-merged_math4w_code4w_user_prompt/global_step_1200/draft_model \
#     --port 30004 \
#     --tp-size 1 \
#     --dtype bfloat16 \
#     --attention-backend fa3 \
#     --mem-fraction-static 0.75 \
#     --trust-remote-code &

CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
    --model-path /mnt/shared-storage-user/p1-shared/Qwen/Qwen3-4B \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path /mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash_data/epoch_5_step_295000 \
    --port 30001 \
    --tp-size 1 \
    --dtype bfloat16 \
    --attention-backend fa3 \
    --mem-fraction-static 0.75 \
    --trust-remote-code &

wait