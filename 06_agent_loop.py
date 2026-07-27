import json

from llm_common import create_chat_completion, create_client


client = create_client()


# -------------------- 工具函数 --------------------


def add_cat_sound(text: str) -> str:
    """在用户提供的句子末尾加上“喵”。"""
    return f"{text}喵"


# -------------------- 工具描述 --------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_cat_sound",
            "description": "在用户提供的句子末尾加上一个“喵”。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "用户提供的原始句子。",
                    }
                },
                "required": ["text"],
            },
        },
    },
]


AVAILABLE_TOOLS = {
    "add_cat_sound": add_cat_sound,
}


def run_agent(messages: list[dict], max_iterations: int = 5) -> str:
    """循环请求模型并执行工具，直到模型生成最终文本。"""
    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Agent 第 {iteration} 轮 ---")

        response = create_chat_completion(
            client,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=500,
            # 关闭思考模式，先专注理解 Tool Calling 的基本循环。
            extra_body={"thinking": {"type": "disabled"}},
        )

        assistant_message = response.choices[0].message
        messages.append(assistant_message.model_dump(exclude_none=True))

        # 没有工具调用，说明模型已经可以直接给出最终回答。
        if not assistant_message.tool_calls:
            return assistant_message.content or "模型没有返回文本内容。"

        # 有工具调用：逐个执行，并将结果追加到对话历史。
        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            tool_arguments = json.loads(tool_call.function.arguments)
            tool_function = AVAILABLE_TOOLS.get(tool_name)

            if tool_function is None:
                raise RuntimeError(f"模型请求了未注册的工具：{tool_name}")

            tool_result = tool_function(**tool_arguments)

            print(f"执行工具：{tool_name}({tool_arguments})")
            print(f"工具结果：{tool_result}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

    raise RuntimeError(f"Agent 达到最大循环次数 {max_iterations}，仍未生成最终回答")


print("请输入一句话，输入 exit 退出。")

while True:
    user_input = input("\n你：").strip()

    if user_input.lower() == "exit":
        print("对话结束。")
        break

    if not user_input:
        print("请输入内容。")
        continue

    messages = [
        {
            "role": "system",
            "content": "你是一个猫咪助手。用户说话后，必须使用 add_cat_sound 工具在句子末尾加上“喵”。",
        },
        {
            "role": "user",
            "content": user_input,
        },
    ]

    final_answer = run_agent(messages)

    print("\nAgent 最终回答：")
    print(final_answer)
