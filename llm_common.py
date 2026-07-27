import os

from dotenv import load_dotenv
from openai import OpenAI


MODEL_NAME = "deepseek-v4-flash"

# 读取环境变量，创建client客户端
def create_client() -> OpenAI:
    load_dotenv()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("没有读取到 DEEPSEEK_API_KEY，请检查 .env 文件")

    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )


# 发起对话请求
def create_chat_completion(client: OpenAI, messages: list[dict], **kwargs):
    return client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        **kwargs,
    )
