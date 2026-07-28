import json

from agent_tools import AVAILABLE_TOOLS, TOOLS_ALL
from llm_common import create_chat_completion, create_client


client = create_client()


# -------------------- Agent 状态 --------------------


def create_agent_state() -> dict:
    """创建一个新的 Agent 状态。"""
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一个有记忆的工具助手。"
                    "根据用户需求选择合适的工具。"
                    "你可以参考前面对话，但不要捏造工具结果。"
                ),
            }
        ],
        "task": {
            "turn_count": 0,
            "tool_call_count": 0,
            "last_tool_results": [],
            "last_answer": "",
        },
    }


def execute_tool(agent_state: dict, tool_call) -> str:
    """执行工具，并更新任务状态。"""
    tool_name = tool_call.function.name
    agent_state["task"]["tool_call_count"] += 1

    try:
        tool_arguments = json.loads(tool_call.function.arguments)
        tool_function = AVAILABLE_TOOLS.get(tool_name)

        if tool_function is None:
            raise ValueError(f"没有注册工具：{tool_name}")

        result = tool_function(**tool_arguments)
    except Exception as error:
        result = f"工具执行失败：{type(error).__name__}: {error}"

    print(f"执行工具：{tool_name}")
    print(f"工具结果：{result}")

    agent_state["task"]["last_tool_results"].append(
        {
            "tool_name": tool_name,
            "result": result,
        }
    )

    return result


def run_agent(agent_state: dict, user_input: str, max_iterations: int = 5) -> str:
    """使用同一个状态执行一轮用户任务。"""
    messages = agent_state["messages"]
    task = agent_state["task"]

    task["turn_count"] += 1
    task["last_tool_results"] = []
    messages.append({"role": "user", "content": user_input})

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Agent 第 {iteration} 轮 ---")

        response = create_chat_completion(
            client,
            messages=messages,
            tools=TOOLS_ALL,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=500,
            extra_body={"thinking": {"type": "disabled"}},
        )

        assistant_message = response.choices[0].message
        messages.append(assistant_message.model_dump(exclude_none=True))

        if not assistant_message.tool_calls:
            answer = assistant_message.content or "模型没有返回文本内容。"
            task["last_answer"] = answer
            return answer

        for tool_call in assistant_message.tool_calls:
            tool_result = execute_tool(agent_state, tool_call)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

    raise RuntimeError(f"Agent 达到最大循环次数 {max_iterations}，仍未生成最终回答")


def print_memory(agent_state: dict) -> None:
    """显示当前保存的消息数量和任务状态。"""
    messages = agent_state["messages"]
    task = agent_state["task"]

    print("\n--- 当前 Agent 状态 ---")
    print(f"对话消息数量：{len(messages)}")
    print(f"已完成对话轮数：{task['turn_count']}")
    print(f"累计工具调用次数：{task['tool_call_count']}")
    print(f"最近工具结果：{task['last_tool_results']}")
    print(f"最近一次回答：{task['last_answer']}")


agent_state = create_agent_state()

print("请输入内容，输入 /state 查看状态，输入 exit 退出。")

while True:
    user_input = input("\n你：").strip()

    if user_input.lower() == "exit":
        print("对话结束。")
        break

    if user_input == "/state":
        print_memory(agent_state)
        continue

    if not user_input:
        print("请输入内容。")
        continue

    final_answer = run_agent(agent_state, user_input)

    print("\nAgent 最终回答：")
    print(final_answer)
