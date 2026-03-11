import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# 加载模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained("/mnt/shared-storage-user/p1-shared/Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("/mnt/shared-storage-user/p1-shared/Qwen/Qwen3-8B")
device = torch.device("cuda:0")
model.to(device).eval()

# 准备对话
messages = [
    {"role": "user", "content": "Write a python function to find the longest chain which can be formed from the given set of pairs."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(device)

# 记录开始时间
start_time = time.time()

# 生成回复
outputs = model.generate(
    **inputs, 
    max_new_tokens=8192,
    do_sample=False  # 确保可重复性，避免采样导致的波动
)

# 计算总用时
total_time = time.time() - start_time

# 计算生成的token数量（不包括输入token）
generated_tokens = outputs[0].shape[0] - inputs["input_ids"].shape[-1]

# 计算token吞吐量 (tokens/second)
tokens_per_second = generated_tokens / total_time

# 解码并打印回复
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print("Model Response:", response)

# 打印性能指标
print("\nPerformance Metrics:")
print(f"Total Time: {total_time:.4f} seconds")
print(f"Generated Tokens: {generated_tokens}")
print(f"Token Throughput: {tokens_per_second:.2f} tokens/second")