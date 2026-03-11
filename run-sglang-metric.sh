# cd /mnt/shared-storage-user/leihaodi/diffusion
# # python sglang-metrics.py --port 8002 --num 256 --batch-size 64 --max-tokens 8192 --output ./logs/qwen3.5/metrics-math500.log --bench-list math500 --context-file ./mbpp.txt &
# # python sglang-metrics.py --port 8000 --num 256 --batch-size 64 --max-tokens 8192 --output ./logs/qwen3.5/metrics-math500.log --bench-list math500 --context-file ./mbpp.txt &
# # python sglang-metrics.py --port 8001 --num 256 --batch-size 64 --max-tokens 8192 --output ./logs/qwen3.5/metrics-math500.log --bench-list math500 --context-file ./mbpp.txt &
# # # python sglang-metrics.py --port 30002 --num 15 --max-tokens 8192 --output ./logs/metrics-aime25.log --bench-list aime25 --context-file ./mbpp.txt &
# # wait

# # echo "math500 done"

# # cd /mnt/shared-storage-user/leihaodi/imo/diffusion
# python sglang-metrics.py --port 8002 --num 96 --batch-size 96 --max-tokens 8192 --output ./logs/qwen3.5/metrics-mbpp.log --bench-list mbpp --context-file ./mbpp.txt &
# python sglang-metrics.py --port 8000 --num 96 --batch-size 96 --max-tokens 8192 --output ./logs/qwen3.5/metrics-mbpp.log --bench-list mbpp --context-file ./mbpp.txt &
# python sglang-metrics.py --port 8001 --num 96 --batch-size 96 --max-tokens 8192 --output ./logs/qwen3.5/metrics-mbpp.log --bench-list mbpp --context-file ./mbpp.txt &
# # python sglang-metrics.py --port 30002 --num 15 --max-tokens 8192 --output ./logs/metrics-aime26.log --bench-list aime26 --context-file ./mbpp.txt &
# wait

# echo "mbpp done"

cd /mnt/shared-storage-user/leihaodi/diffusion
python sglang-metrics.py --port 30000 --num 10 --max-tokens 2048 --bench-list math500 &
python sglang-metrics.py --port 30002 --num 10 --max-tokens 2048 --bench-list math500 &
# python sglang-metrics.py --port 8000 --num 10 --max-tokens 8192 --output ./logs/qwen3.5/metrics-math500.log --bench-list math500 --context-file ./mbpp.txt &
# python sglang-metrics.py --port 8001 --num 10 --max-tokens 8192 --output ./logs/qwen3.5/metrics-math500.log --bench-list math500 --context-file ./mbpp.txt &
# python sglang-metrics.py --port 30002 --num 15 --max-tokens 8192 --output ./logs/metrics-aime25.log --bench-list aime25 --context-file ./mbpp.txt &
wait

echo "gsm8k done"

# python sglang-metrics.py --port 30000 --num 80 --batch-size 64 --max-tokens 2048 --output /mnt/shared-storage-user/leihaodi/pretrain/mtp-debug/mtp-mt-bench-30000.log --bench-list mt-bench &
# python sglang-metrics.py --port 30002 --num 80 --batch-size 64 --max-tokens 2048 --output /mnt/shared-storage-user/leihaodi/pretrain/mtp-debug/mtp-mt-bench-30002.log --bench-list mt-bench &
# wait

# echo "mt-bench done"

# python sglang-metrics.py --port 30003 --num 10 --max-tokens 8192 --bench-list gsm8k
# wait

# echo "dapo done"

# python sglang-metrics.py --port 30001 --num 10 --max-tokens 28000 --output ./logs/dapo2.log --bench-list dapo &
# wait

# echo "dapo done"

# cd /mnt/shared-storage-user/leihaodi/imo/diffusion
# python sglang-metrics.py --port 8002 --num 15 --max-tokens 8192 --output ./logs/qwen3.5/metrics-mbpp.log --bench-list mbpp --context-file ./mbpp.txt &
# python sglang-metrics.py --port 8000 --num 15 --max-tokens 8192 --output ./logs/qwen3.5/metrics-mbpp.log --bench-list mbpp --context-file ./mbpp.txt &
# python sglang-metrics.py --port 8001 --num 15 --max-tokens 8192 --output ./logs/qwen3.5/metrics-mbpp.log --bench-list mbpp --context-file ./mbpp.txt &
# # python sglang-metrics.py --port 30002 --num 15 --max-tokens 8192 --output ./logs/metrics-aime26.log --bench-list aime26 --context-file ./mbpp.txt &
# wait

# echo "mbpp done"