from openai import OpenAI
import time
# 连接到你的 SGLang 服务器
client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="EMPTY"  # SGLang 不需要真实的 API key
)

# 准备8个请求
prompts = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a haiku about spring.",
    "What are the benefits of exercise?",
    "Describe the water cycle.",
    "What is machine learning?",
    "Tell me a joke.",
    "Explain photosynthesis."
]

# 并发发送请求
from concurrent.futures import ThreadPoolExecutor, as_completed

def send_request(prompt):
    response = client.chat.completions.create(
        model="default",  # SGLang 会使用你启动的模型
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=2560
    )
    return prompt, response.choices[0].message.content

# 使用线程池并发发送并统计用时
start_time = time.time()
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(send_request, prompt) for prompt in prompts]
    
    for future in as_completed(futures):
        prompt, response = future.result()
        print(f"\n--- Request ---")
        print(f"Input: {prompt}")
        print(f"Output: {response}")

end_time = time.time()
print(f"Total time: {end_time - start_time} seconds")
print(f"Average time per request: {(end_time - start_time) / len(prompts)} seconds")