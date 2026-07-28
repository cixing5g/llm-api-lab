# json 用来解析模型返回的工具参数。
import json

# FastAPI：创建 HTTP Web 应用。
from fastapi import FastAPI

# BaseModel：定义和校验 HTTP 请求体、响应体的数据结构。
from pydantic import BaseModel

# 导入公共工具：
# - TOOLS_CAT：告诉模型有哪些工具可以调用
# - add_cat_sound：模型请求工具后，真正由 Python 执行的函数
from agent_tools import TOOLS_CAT, add_cat_sound

# 导入公共大模型客户端和请求函数。
from llm_common import create_chat_completion, create_client


# 创建大模型客户端。
# API Key 会由 llm_common.py 从 .env 中读取。
client = create_client()

# 创建 FastAPI 应用对象。
# 后面的 @app.get() 和 @app.post() 都是在这个对象上注册接口。
app = FastAPI(title="很简陋的Agent Demo")

# ----------------------------以上为准备工作------------------------------
# ----------------------------以上为准备工作------------------------------
# ----------------------------以上为准备工作------------------------------


# 客户端发送 POST /chat 时，请求 JSON 必须符合这个结构：
# {
#     "session_id": "demo",
#     "message": "今天天气不错"
# }
class ChatRequest(BaseModel):
    # session_id 用来区分不同用户或不同会话。
    session_id: str

    # message 是用户本次发送给 Agent 的内容。
    message: str


# 这是 POST /chat 返回结果的结构：
# {
#     "session_id": "demo",
#     "answer": "今天天气不错喵"
# }
class ChatResponse(BaseModel):
    session_id: str
    answer: str


# 保存所有会话的消息历史。
#
# 字典的结构大致如下：
# {
#     "demo": [
#         {"role": "system", "content": "..."},
#         {"role": "user", "content": "..."},
#         {"role": "assistant", "content": "..."},
#         {"role": "tool", "content": "..."}
#     ]
# }
#
# 这里使用内存保存，所以程序重启后历史会丢失。
sessions: dict[str, list[dict]] = {}


def get_or_create_messages(session_id: str) -> list[dict]:
    """获取指定会话的消息历史；第一次使用时创建初始 system 消息。"""

    # 如果这是一个新会话，就先创建它的 system 消息。
    # system 消息用于规定 Agent 的身份和行为规则。
    if session_id not in sessions:
        sessions[session_id] = [
            {
                "role": "system",
                "content": "你是一个猫咪助手。用户说话后，必须使用工具在句子末尾加上“喵”。",
            }
        ]

    # 返回这个会话对应的消息列表。
    # 后续用户消息、模型消息和工具消息都会追加到同一个列表中。
    return sessions[session_id]


def run_agent(messages: list[dict]) -> str:
    """执行 Agent 主循环，直到模型生成最终文本回答。"""

    while True:
        # 第一步：把当前完整对话历史和工具描述发送给模型。
        response = create_chat_completion(
            client,
            messages=messages,
            tools=TOOLS_CAT,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=300,
            extra_body={"thinking": {"type": "disabled"}},
        )

        # 取出模型本次返回的 assistant 消息。
        assistant_message = response.choices[0].message

        # 必须把 assistant 消息保存到历史中。
        # 如果模型请求了工具，这条消息中会包含 tool_calls。
        messages.append(assistant_message.model_dump(exclude_none=True))

        # 如果没有 tool_calls，说明模型已经生成了最终回答，循环结束。
        if not assistant_message.tool_calls:
            return assistant_message.content or "模型没有返回内容。"

        # 如果存在工具调用，取出模型请求的第一个工具。
        # 当前示例只有 add_cat_sound 一个工具。
        tool_call = assistant_message.tool_calls[0]

        # 模型返回的 arguments 是 JSON 字符串，先转换成 Python 字典。
        arguments = json.loads(tool_call.function.arguments)

        # 模型只负责决定“调用哪个工具、传什么参数”，
        # 工具本身必须由 Python 程序实际执行。
        result = add_cat_sound(**arguments)

        # 把 Python 工具的执行结果追加回消息历史。
        # tool_call_id 用来告诉模型：这个结果对应哪一次工具调用。
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
        )


# --------------------------FASTAPI---------------------------------
# --------------------------FASTAPI---------------------------------
# --------------------------FASTAPI---------------------------------


# 健康检查接口。
# 浏览器访问 http://127.0.0.1:8000/ 时会执行这个函数。
# 它不调用大模型，只用于确认 FastAPI 服务是否启动成功。
@app.get("/")
def health_check() -> dict[str, str]:
    return {"message": "Agent API is running"}


# 聊天接口。
# 客户端向 POST /chat 发送 JSON 后，FastAPI 会自动：
# 1. 将 JSON 转换成 ChatRequest 对象
# 2. 校验 session_id 和 message 是否存在
# 3. 调用下面的 chat() 函数
# 4. 将 ChatResponse 对象转换成 JSON 返回
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    # 根据 session_id 获取对应的历史消息。
    # 相同 session_id 会使用同一段对话历史。
    messages = get_or_create_messages(request.session_id)

    # 将本次用户输入追加到历史消息中。
    messages.append(
        {
            "role": "user",
            "content": request.message,
        }
    )

    # 执行 Agent 主循环。
    # run_agent 可能会调用模型一次或多次，直到得到最终回答。
    answer = run_agent(messages)

    # 返回结构化响应。
    return ChatResponse(
        session_id=request.session_id,
        answer=answer,
    )
