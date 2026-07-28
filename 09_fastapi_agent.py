import json

from fastapi import FastAPI
from pydantic import BaseModel

from llm_common import create_chat_completion, create_client


client = create_client()
app = FastAPI(title="Simple Agent API")


def add_cat_sound(text: str) -> str:
    """在句子末尾加上“喵”。"""
    return f"{text}喵"


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


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str


# 每个 session_id 对应一份独立的对话历史。
sessions: dict[str, list[dict]] = {}


def get_or_create_messages(session_id: str) -> list[dict]:
    if session_id not in sessions:
        sessions[session_id] = [
            {
                "role": "system",
                "content": "你是一个猫咪助手。用户说话后，必须使用工具在句子末尾加上“喵”。",
            }
        ]
    return sessions[session_id]


def run_agent(messages: list[dict]) -> str:
    """请求模型，执行工具，再请求模型生成最终回答。"""
    while True:
        response = create_chat_completion(
            client,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=300,
            extra_body={"thinking": {"type": "disabled"}},
        )

        assistant_message = response.choices[0].message
        messages.append(assistant_message.model_dump(exclude_none=True))

        if not assistant_message.tool_calls:
            return assistant_message.content or "模型没有返回内容。"

        tool_call = assistant_message.tool_calls[0]
        arguments = json.loads(tool_call.function.arguments)
        result = add_cat_sound(**arguments)

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
        )


@app.get("/")
def health_check() -> dict[str, str]:
    return {"message": "Agent API is running"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    messages = get_or_create_messages(request.session_id)
    messages.append(
        {
            "role": "user",
            "content": request.message,
        }
    )

    answer = run_agent(messages)

    return ChatResponse(
        session_id=request.session_id,
        answer=answer,
    )
