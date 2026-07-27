import json

from llm_common import create_chat_completion, create_client


client = create_client()


# -------------------- Python 工具函数 --------------------


def add_cat_sound(text: str) -> str:
    """在句子末尾加上“喵”。"""
    return f"{text}喵"


def get_text_length(text: str) -> str:
    """返回文字长度。"""
    return str(len(text))


# -------------------- 提供给模型的工具描述 --------------------

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
]


AVAILABLE_TOOLS = {
    "add_cat_sound": add_cat_sound,
    "get_text_length": get_text_length,
}


messages = [
    {
        "role": "system",
        "content": "你是一个文字助手。需要加喵时使用 add_cat_sound，需要统计字数时使用 get_text_length。",
    },
    {
        "role": "user",
        "content": "请给“今天天气不错”加喵，并统计这句话有多少个字。",
    },
]


# 第一次请求：模型判断是否需要调用工具
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
    print("模型没有调用工具，直接回答：")
    print(assistant_message.content)
else:
    print("模型请求调用工具：")

    for tool_call in assistant_message.tool_calls:
        tool_name = tool_call.function.name
        tool_arguments = json.loads(tool_call.function.arguments)
        tool_function = AVAILABLE_TOOLS.get(tool_name)

        if tool_function is None:
            tool_result = f"未知工具：{tool_name}"
        else:
            tool_result = tool_function(**tool_arguments)

        print(f"- {tool_name}({tool_arguments}) -> {tool_result}")

        # 把 Python 工具的执行结果返回给模型
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            }
        )

    # 第二次请求：模型根据工具结果组织最终回答
    final_response = create_chat_completion(
        client,
        messages=messages,
        tools=TOOLS,
        temperature=0.2,
        max_tokens=500,
        extra_body={"thinking": {"type": "disabled"}},
    )

    print("\n模型最终回答：")
    print(final_response.choices[0].message.content)
