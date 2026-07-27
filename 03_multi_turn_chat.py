from llm_common import create_chat_completion, create_client


client = create_client()


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

    response = create_chat_completion(
        client,
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
