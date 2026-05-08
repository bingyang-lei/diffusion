

import requests
import time
import json
port = 30000
url = f"http://localhost:{port}/v1/responses"
context = """
Janet had 22 green pens and 10 yellow pens. Then she bought 6 bags of blue pens and 2 bags of red pens. There were 9 pens in each bag of blue and 6 pens in each bag of red. How many pens does Janet have now?\nPlease reason step by step, and put your final answer within \\boxed{}
"""   # a test case

# context = """
# Jen enters a lottery by picking $4$ distinct numbers from $S=\\{1,2,3,\\cdots,9,10\\}.
# $4$ numbers are randomly chosen from $S.$ She wins a prize if at least two of her numbers were $2$ of the randomly chosen numbers, 
# and wins the grand prize if all four of her numbers were the randomly chosen numbers. 
# The probability of her winning the grand prize given that she won a prize is $\\tfrac{m}{n}$ where $m$ and $n$ are relatively prime positive integers.
# Find $m+n$.  # AIME answer: 116
# """

# context = """
# Let $x,y$ and $z$ be positive real numbers that satisfy the following system of equations: 
# \\[\\log_2\\left({x \\over yz}\right) = {1 \\over 2}\\]
# \\[\\log_2\\left({y \\over xz}\right) = {1 \\over 3}\\]
# \\[\\log_2\\left({z \\over xy}\right) = {1 \\over 4}\\]
# Then the value of $\\left|\\log_2(x^4y^3z^2)\\right|$ is $\\tfrac{m}{n}$ where $m$ and $n$ are relatively prime positive integers. Find $m+n$.
# """  # AIME answer: 33
# context = "hello, introduce yourself"
# context = "Write a python function to find the longest chain which can be formed from the given set of pairs[]."



# data = {
#     "model": "None",
#     "input": context,
#     "max_output_tokens": 512,
# }
# start_time = time.time()
# response = requests.post(url, json=data)
# print(json.dumps(response.json(), indent=4, ensure_ascii=False))
# end_time = time.time()
# print(f"Response time taken: {end_time - start_time} seconds")
context = "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}.\n\nHow many 6-digit numbers $\\overline{a_1a_2a_3a_4a_5a_6}$ (where $a_1 \\neq 0$) are there, with $0 \\leq a_i \\leq 9$ for each $i$, such that $a_1 < a_2 > a_3 < a_4 > a_5 < a_6$?"
url2 = f"http://localhost:{port}/v1/chat/completions"
data2 = {
    "model": "None",
    "messages": [{"role": "user", "content": context}],
    "max_tokens": 4096,
    "temperature": 1
    # "chat_template_kwargs": {"enable_thinking": False}
}
start_time2 = time.time()
response2 = requests.post(url2, json=data2)
print(json.dumps(response2.json(), indent=4, ensure_ascii=False))
end_time2 = time.time()
print(f"Response2 time taken: {end_time2 - start_time2} seconds")

# url3 = f"http://localhost:{port}/v1/completions"
# data3 = {
#     "model": "None",
#     "prompt": context,
#     "max_tokens": 1024,
# }
# start_time3 = time.time()
# response3 = requests.post(url3, json=data3)
# print(json.dumps(response3.json(), indent=4, ensure_ascii=False))
# end_time3 = time.time()
# print(f"Response3 time taken: {end_time3 - start_time3} seconds")




# from openai import OpenAI

# # 创建客户端，指定本地服务器地址
# client = OpenAI(
#     base_url=f"http://localhost:30010/v1",  # 注意这里需要包含 /v1
#     api_key="not-needed"  # 本地服务可能不需要验证，随便填一个即可
# )

# response = client.chat.completions.create(
#     model="XiaomiMiMo/MiMo-7B-RL",
#     messages=[{"role": "user", "content": "Tell me a story about a cat."}],
# )

# print(response.choices[0].message.content)



# import time
# import openai

# # 指向 SGLang HTTP server
# client = openai.Client(
#     base_url="http://10.102.207.120:15000/v1",
#     api_key="EMPTY",  # SGLang 本地 HTTP 不校验 key
# )

# start_time = time.time()

# response = client.chat.completions.create(
#     model="dummy",  # SGLang 一般不强校验 model 名
#     messages=[
#         {"role": "system", "content": "You are a helpful AI assistant."},
#         {
#             "role": "user",
#             "content": (
#                 "Solve the following math problem step by step. "
#                 "The last line of your response should be of the form "
#                 "Answer: \\boxed{$Answer}.\n\n"
#                 "Let $ABC$ be an isosceles triangle with $\\angle A = 90^\\circ$. "
#                 "Points $D$ and $E$ are selected on sides $AB$ and $AC$, and "
#                 "points $X$ and $Y$ are the feet of the altitudes from $D$ and $E$ "
#                 "to side $BC$. Given that $AD = 48\\sqrt{2}$ and $AE = 52\\sqrt{2}$, "
#                 "compute $XY$."
#             ),
#         },
#     ],
#     temperature=0.7,
# )

# print("Response:")
# print(response.choices[0].message.content)
# print(f"Latency: {time.time() - start_time:.2f}s")


# import openai
# import time  # 导入time模块用于测量时间

# # 配置客户端指向你的本地服务
# client = openai.Client(
#     base_url="http://localhost:30000/v1",
#     api_key="EMPTY"  # 本地服务通常不需要真实的 Key
# )

# # 记录开始时间
# start_time = time.time()

# # 发送请求
# response = client.chat.completions.create(
#     model="/mnt/shared-storage-user/cuiganqu/Qwen/Qwen3-30B-A3B-Thinking-2507_slime/thinking-nofilter-128-16-gspo-64k-remote-tis-partial-pass88zero-resume-120-0924-162826/iter_0006001_hf",
#     messages=[
#         {"role": "system", "content": "You are a helpful AI assistant."},
#         {"role": "user", "content": "Solve the following math problem step by step. The last line of your response should be of the form Answer: \\boxed{$Answer} where $Answer is the answer to the problem.\n\nLet $ABC$ be an isosceles triangle with $\\angle A = 90^\\circ$. Points $D$ and $E$ are selected on sides $AB$ and $AC$, and points $X$ and $Y$ are the feet of the altitudes from $D$ and $E$ to side $BC$. Given that $AD = 48\\sqrt{2}$ and $AE = 52\\sqrt{2}$, compute $XY$.\n\nRemember to put your answer on its own line after 'Answer':"},
#     ],
#     temperature=0.7,
# )

# # 记录结束时间
# end_time = time.time()

# # 计算生成时间
# generation_time = end_time - start_time

# # 打印生成时间
# print(f"生成回复所用时间: {generation_time:.4f} 秒")

# # 打印回复内容
# print("\n回复内容:")
# print(response.choices[0].message.content)





# import openai
# import time
# import asyncio
# from openai import AsyncOpenAI

# async def send_request(base_url, port):
#     """向指定端口发送请求并测量时间"""
#     start_time = time.time()
    
#     # 创建异步客户端
#     client = AsyncOpenAI(
#         base_url=f"http://localhost:{port}/v1",
#         api_key="EMPTY"
#     )
    
#     try:
#         response = await client.chat.completions.create(
#             model="/mnt/shared-storage-user/cuiganqu/Qwen/Qwen3-30B-A3B-Thinking-2507_slime/thinking-nofilter-128-16-gspo-64k-remote-tis-partial-pass88zero-resume-120-0924-162826/iter_0006001_hf",
#             messages=[
#                 {"role": "system", "content": "You are a helpful AI assistant."},
#                 {"role": "user", "content": "Solve the following math problem step by step. The last line of your response should be of the form Answer: \\boxed{$Answer} where $Answer is the answer to the problem.\n\nLet $ABC$ be an isosceles triangle with $\\angle A = 90^\\circ$. Points $D$ and $E$ are selected on sides $AB$ and $AC$, and points $X$ and $Y$ are the feet of the altitudes from $D$ and $E$ to side $BC$. Given that $AD = 48\\sqrt{2}$ and $AE = 52\\sqrt{2}$, compute $XY$.\n\nRemember to put your answer on its own line after 'Answer':"},
#             ],
#             temperature=0.7,
#         )
        
#         end_time = time.time()
#         generation_time = end_time - start_time
        
#         return {
#             "port": port,
#             "response": response.choices[0].message.content,
#             "time": generation_time
#         }
#     except Exception as e:
#         end_time = time.time()
#         generation_time = end_time - start_time
#         return {
#             "port": port,
#             "response": f"请求失败: {str(e)}",
#             "time": generation_time,
#             "error": True
#         }

# async def main():
#     """主函数，同时向两个端口发送请求"""
#     print("正在同时向端口30001和30002发送请求...")
#     print("=" * 60)
    
#     # 同时发送两个请求
#     tasks = [
#         send_request("http://localhost", 30001),
#         send_request("http://localhost", 30002)
#     ]
    
#     results = await asyncio.gather(*tasks)
    
#     # 打印结果
#     for result in results:
#         port = result["port"]
#         response = result["response"]
#         generation_time = result["time"]
#         is_error = result.get("error", False)
        
#         print(f"\n{'=' * 30} 端口 {port} 的结果 {'=' * 30}")
#         print(f"生成回复所用时间: {generation_time:.4f} 秒")
        
#         if is_error:
#             print("错误信息:")
#             print(response)
#         else:
#             print("\n回复内容:")
#             print(response[-100:])  # 只打印回复的最后100个字符以节省空间
        
#         print(f"{'=' * 60}")

# # 运行异步主函数
# if __name__ == "__main__":
#     asyncio.run(main())