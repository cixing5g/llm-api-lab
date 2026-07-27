import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError("[02 Speaking]没有读取到 DEEPSEEK_API_KEY，请检查 .env 文件")


# 创建大模型客户端
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)


messages = [
    {
        "role": "system",
        "content": "你是一名学者，回答要准确、简洁、容易理解。",
    },
    {
        "role": "user",
        "content": "请给我讲讲荆轲刺秦王的故事。",
    },
]


# stream=True：开启流式输出
response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=messages,
    temperature=0.2,
    max_tokens=300,
    stream=True,
)


print("[02 Speaking]模型回答：")

# response 不再是完整回答，而是可以不断取出内容的数据流
for chunk in response:
    content = chunk.choices[0].delta.content

    if content:
        print(content, end="", flush=True)