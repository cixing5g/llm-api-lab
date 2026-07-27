import ast
import json
import operator
import random

from llm_common import create_chat_completion, create_client


client = create_client()


# -------------------- Python 工具函数 --------------------

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _evaluate_number(node: ast.AST) -> int | float:
    """只计算数字和四则运算，避免直接执行任意 Python 代码。"""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_number(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value

    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left = _evaluate_number(node.left)
        right = _evaluate_number(node.right)
        return _OPERATORS[type(node.op)](left, right)

    raise ValueError("只支持数字和 +、-、*、/ 四则运算")


def calculate(expression: str) -> str:
    """计算一个只包含数字和四则运算符的表达式。"""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate_number(tree.body)
        return str(result)
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError) as error:
        return f"计算失败：{error}"


INTERVIEW_QUESTIONS = {
    "Java 基础": [
        "Java 中 == 和 equals() 有什么区别？",
        "请解释 Java 的垃圾回收机制。",
    ],
    "Java 并发": [
        "ThreadLocal 为什么可能导致内存泄漏？",
        "synchronized 和 ReentrantLock 有什么区别？",
    ],
}


def get_random_interview_question(category: str) -> str:
    """根据面试题类别随机返回一道 Java 面试题。"""
    questions = INTERVIEW_QUESTIONS.get(category)
    if not questions:
        available_categories = "、".join(INTERVIEW_QUESTIONS)
        return f"没有找到类别“{category}”，可选类别：{available_categories}"
    return random.choice(questions)


# -------------------- 提供给模型的工具描述 --------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算只包含数字和四则运算符的数学表达式。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的表达式，例如 12 * 8。",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_random_interview_question",
            "description": "根据 Java 面试题类别随机返回一道面试题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "面试题类别，例如 Java 基础 或 Java 并发。",
                    }
                },
                "required": ["category"],
            },
        },
    },
]


AVAILABLE_TOOLS = {
    "calculate": calculate,
    "get_random_interview_question": get_random_interview_question,
}


messages = [
    {
        "role": "system",
        "content": "你是一个 Java 面试助手。需要计算时使用计算工具，需要出题时使用面试题工具。",
    },
    {
        "role": "user",
        "content": "请计算 12 * 8，并随机出一道 Java 并发面试题。",
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
