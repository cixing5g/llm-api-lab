import json

from agent_tools import TOOLS_CAT, add_cat_sound
from llm_common import create_chat_completion, create_client


client = create_client()


def run_agent(messages: list[dict]) -> str:
    """请求模型；如果模型调用工具，就执行工具并继续请求。"""
    while True:
        response = create_chat_completion(
            client,
            messages=messages,
            tools=TOOLS_CAT,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=300,
            extra_body={"thinking": {"type": "disabled"}},
        )

        assistant_message = response.choices[0].message
        messages.append(assistant_message.model_dump(exclude_none=True))

        # 没有工具调用，说明模型已经生成最终回答。
        if not assistant_message.tool_calls:
            return assistant_message.content

        # 本示例只有一个工具，因此直接处理第一个工具调用。
        tool_call = assistant_message.tool_calls[0] # 第一个工具调用
        arguments = json.loads(tool_call.function.arguments)
        result = add_cat_sound(**arguments)

        print(f"执行工具：{tool_call.function.name}")
        print(f"工具结果：{result}")

        # 工具结果也要加入历史消息，模型才能继续回答。
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
        )


messages = [
    {
        "role": "system",
        "content": "你是一个猫咪助手。用户说话后，必须使用工具在句子末尾加上“喵”。",
    }
]


print("请输入内容，输入 exit 退出。")

while True:
    user_input = input("\n你：").strip()

    if user_input.lower() == "exit":
        print("对话结束。")
        break

    if not user_input:
        continue

    # 关键：messages 不会在循环中重新创建，所以它会保存多轮对话历史。
    messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    answer = run_agent(messages)

    print("\nAgent：")
    print(answer)
