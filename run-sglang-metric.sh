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
# python sglang-metrics.py --port 30000 --num 10 --max-tokens 4096 --enable-thinking --bench-list math500 &
# python sglang-metrics.py --port 30001 --num 10 --max-tokens 4096 --enable-thinking --bench-list math500 &
# wait

python sglang-metrics.py --port 30000 --num 10 --max-tokens 4096 --enable-thinking --bench-list mbpp --temperature 1.0 &
python sglang-metrics.py --port 30001 --num 10 --max-tokens 4096 --enable-thinking --bench-list mbpp --temperature 1.0 &
wait
echo "eval done"
# TEMP1_LOG=/mnt/shared-storage-user/leihaodi/diffusion/temp1.log
# TEMP0_LOG=/mnt/shared-storage-user/leihaodi/diffusion/temp0.log
# BENCH_LIST=(gsm8k aime24 math500 mbpp humaneval dapo mt-bench)
# PORT_LIST=(30000 30001 30002 30003 30004 30007)

# : > "$TEMP1_LOG"
# : > "$TEMP0_LOG"

# for bench in "${BENCH_LIST[@]}"; do
#     echo "eval ${bench} temperature=1.0" >> "$TEMP1_LOG"
#     for port in "${PORT_LIST[@]}"; do
#         part_log="${TEMP1_LOG}.${bench}.${port}.part"
#         : > "$part_log"
#         python sglang-metrics.py --port "$port" --num 10 --max-tokens 8192 --enable-thinking --bench-list "$bench" --temperature 1.0 --output "$part_log" &
#     done
#     wait
#     for port in "${PORT_LIST[@]}"; do
#         part_log="${TEMP1_LOG}.${bench}.${port}.part"
#         echo "port ${port}" >> "$TEMP1_LOG"
#         cat "$part_log" >> "$TEMP1_LOG"
#         rm -f "$part_log"
#     done
# done

# for bench in "${BENCH_LIST[@]}"; do
#     echo "eval ${bench} temperature=0.0" >> "$TEMP0_LOG"
#     for port in "${PORT_LIST[@]}"; do
#         part_log="${TEMP0_LOG}.${bench}.${port}.part"
#         : > "$part_log"
#         python sglang-metrics.py --port "$port" --num 10 --max-tokens 8192 --enable-thinking --bench-list "$bench" --temperature 0.0 --output "$part_log" &
#     done
#     wait
#     for port in "${PORT_LIST[@]}"; do
#         part_log="${TEMP0_LOG}.${bench}.${port}.part"
#         echo "port ${port}" >> "$TEMP0_LOG"
#         cat "$part_log" >> "$TEMP0_LOG"
#         rm -f "$part_log"
#     done
# done
# wait

# echo "eval done"
# python /mnt/shared-storage-user/leihaodi/gpu_stress_test.py
