from llm_common import create_chat_completion, create_client


client = create_client()


# 调用一次大模型
response = create_chat_completion(
    client,
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
    max_tokens=300,
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
