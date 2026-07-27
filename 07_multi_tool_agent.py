import json

from llm_common import create_chat_completion, create_client


client = create_client()


# -------------------- 工具函数 --------------------


def add_cat_sound(text: str) -> str:
    """在句子末尾加上“喵”。"""
    return f"{text}喵"


def get_text_length(text: str) -> str:
    """返回文字长度。"""
    return str(len(text))


def divide_numbers(dividend: float, divisor: float) -> str:
    """计算两个数字相除的结果。"""
    if divisor == 0:
        raise ValueError("除数不能为 0")
    return str(dividend / divisor)


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
    {
        "type": "function",
        "function": {
            "name": "get_text_length",
            "description": "统计用户提供的文字包含多少个字符。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "需要统计长度的文字。",
                    }
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "divide_numbers",
            "description": "计算两个数字相除的结果。",
            "parameters": {
                "type": "object",
                "properties": {
                    "dividend": {
                        "type": "number",
                        "description": "被除数。",
                    },
                    "divisor": {
                        "type": "number",
                        "description": "除数。",
                    },
                },
                "required": ["dividend", "divisor"],
            },
        },
    },
]


AVAILABLE_TOOLS = {
    "add_cat_sound": add_cat_sound,
    "get_text_length": get_text_length,
    "divide_numbers": divide_numbers,
}


def execute_tool(tool_call) -> str:
    """根据模型的工具调用执行 Python 函数，并把异常转换成工具结果。"""
    tool_name = tool_call.function.name

    try:
        tool_arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as error:
        return f"工具执行失败：参数不是合法 JSON，{error}"

    tool_function = AVAILABLE_TOOLS.get(tool_name)
    if tool_function is None:
        return f"工具执行失败：没有注册工具 {tool_name}"

    try:
        result = tool_function(**tool_arguments)
        print(f"执行工具：{tool_name}({tool_arguments})")
        print(f"工具结果：{result}")
        return result
    except Exception as error:
        error_message = f"工具执行失败：{type(error).__name__}: {error}"
        print(f"执行工具：{tool_name}({tool_arguments})")
        print(error_message)
        return error_message


def run_agent(messages: list[dict], max_iterations: int = 5) -> str:
    """循环选择和执行工具，直到模型生成最终回答。"""
    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Agent 第 {iteration} 轮 ---")

        response = create_chat_completion(
            client,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=500,
            extra_body={"thinking": {"type": "disabled"}},
        )

        assistant_message = response.choices[0].message
        messages.append(assistant_message.model_dump(exclude_none=True))

        if not assistant_message.tool_calls:
            return assistant_message.content or "模型没有返回文本内容。"

        # 同一轮中可能有多个工具调用，逐个执行并追加结果。
        for tool_call in assistant_message.tool_calls:
            tool_result = execute_tool(tool_call)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

    raise RuntimeError(f"Agent 达到最大循环次数 {max_iterations}，仍未生成最终回答")


print("你可以要求 Agent 加喵、统计文字长度或做除法。输入 exit 退出。")
print("例如：请给‘你好’加喵，并统计‘你好’有几个字。")

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
            "content": (
                "你是一个工具助手。根据用户需求选择合适的工具。"
                "如果用户提出多个任务，可以调用多个工具。"
                "如果工具返回错误，要如实告诉用户，不要假装工具成功。"
            ),
        },
        {
            "role": "user",
            "content": user_input,
        },
    ]

    final_answer = run_agent(messages)

    print("\nAgent 最终回答：")
    print(final_answer)
