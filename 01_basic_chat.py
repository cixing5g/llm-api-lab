import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError("[01 Speaking]没有读取到 DEEPSEEK_API_KEY，请检查 .env 文件")


# 创建大模型客户端
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)


# 调用一次大模型
response = client.chat.completions.create(
    model="deepseek-v4-flash",
    max_tokens=300,
    messages=[
        {
            "role": "system",
            "content": "你是一名Java面试官，回答要准确、简洁、容易理解。",
        },
        {
            "role": "user",
            "content": "请解释一下Java中的ThreadLocal。",
        },
    ],
    stream=False,

    # 第一次实验先关闭思考模式，响应更快
    extra_body={
        "thinking": {
            "type": "disabled"
        }
    },
)


print("[01 Speaking]模型回答：")
print(response.choices[0].message.content)
print("\n")

print("[01 Speaking]Token 使用情况：")
print(response.usage)
print("\n")