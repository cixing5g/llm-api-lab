import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError("[03 Speaking]没有读取到 DEEPSEEK_API_KEY，请检查 .env 文件")


client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)


# 整个对话的历史记录
messages = [
    {
        "role": "system",
        "content": "回答要准确、简洁、容易理解。",
    }
]


print("[03 Speaking]开始对话，输入 exit 退出程序。")

while True:
    user_input = input("\n[03 Speaking]你：").strip()

    # 用户输入 exit 时，结束循环
    if user_input.lower() == "exit":
        print("[03 Speaking]对话结束。")
        break

    # 用户直接按回车，不发送空消息
    if not user_input:
        print("[03 Speaking]请输入内容。")
        continue

    # 把用户本轮问题加入历史记录
    messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=messages,
        temperature=0.8,
        max_tokens=300,
    )

    assistant_answer = response.choices[0].message.content

    print(f"\n[03 Speaking]模型：{assistant_answer}")

    # 把模型本轮回答也加入历史记录
    messages.append(
        {
            "role": "assistant",
            "content": assistant_answer,
        }
    )