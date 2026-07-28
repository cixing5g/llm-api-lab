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


ADD_CAT_SOUND_TOOL = {
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
}


GET_TEXT_LENGTH_TOOL = {
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
}


DIVIDE_NUMBERS_TOOL = {
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
}


TOOLS_CAT = [ADD_CAT_SOUND_TOOL]
TOOLS_TEXT = [ADD_CAT_SOUND_TOOL, GET_TEXT_LENGTH_TOOL]
TOOLS_ALL = [ADD_CAT_SOUND_TOOL, GET_TEXT_LENGTH_TOOL, DIVIDE_NUMBERS_TOOL]


AVAILABLE_TOOLS = {
    "add_cat_sound": add_cat_sound,
    "get_text_length": get_text_length,
    "divide_numbers": divide_numbers,
}
