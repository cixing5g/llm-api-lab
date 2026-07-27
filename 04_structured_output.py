import json
import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError("[04 Speaking]没有读取到 DEEPSEEK_API_KEY，请检查 .env 文件")


client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)


messages = [
    {
        "role": "system",
        "content": """
你是一名Java面试题整理助手。

请将回答严格输出为JSON格式，不要输出Markdown代码块，也不要输出任何额外文字。

JSON格式如下：
{
  "topic": "知识点名称，字符串",
  "summary": "知识点概述，字符串",
  "key_points": ["关键点1", "关键点2", "关键点3"],
  "difficulty": 1
}

字段要求：
1. topic：知识点名称
2. summary：简洁概括知识点
3. key_points：字符串数组，包含3至5个关键点
4. difficulty：整数，范围为1至5
""",
    },
    {
        "role": "user",
        "content": "请整理Java中的ThreadLocal。",
    },
]


response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=messages,
    temperature=0.2,
    max_tokens=400,

    # 要求模型返回合法的JSON字符串
    response_format={
        "type": "json_object",
    },
)


# 此时得到的仍然是字符串
content = response.choices[0].message.content

if not content:
    raise RuntimeError("[04 Speaking]模型返回内容为空")


print("[04 Speaking]模型原始输出：")
print(content)

# 把JSON字符串转换为Python字典
result = json.loads(content)

print("\n[04 Speaking]转换后的Python对象：")
print(result)

print("\n[04 Speaking]单独读取字段：")
print("[04 Speaking]知识点：", result["topic"])
print("[04 Speaking]概述：", result["summary"])
print("[04 Speaking]关键点：", result["key_points"])
print("[04 Speaking]难度：", result["difficulty"])