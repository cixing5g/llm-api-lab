from llm_common import create_chat_completion, create_client


client = create_client()


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
response = create_chat_completion(
    client,
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
