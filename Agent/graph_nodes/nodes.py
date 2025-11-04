import os
from pathlib import Path
from typing import TypedDict

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

from agent.prompts import chat_prompt, router_prompt, style_prompt
from tools import drink_ordering, get_current_time, get_weather, web_search

BASE_DIR = Path(__file__).resolve().parents[2]

openai_api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
openai_api_base = os.getenv("OPENAI_API_BASE", "http://localhost:8888/v1")
model_name = os.getenv(
    "WECHATROBOT_MODEL_PATH",
    str(BASE_DIR / "Saved_models" / "rlhf" / "4B_lora_PPO_V3" / "merged"),
)

router_openai_api_key = os.getenv(
    "ROUTER_OPENAI_API_KEY",
    "ROUTER_KEY_PLACEHOLDER",
)
router_openai_api_base = os.getenv(
    "ROUTER_OPENAI_API_BASE",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
router_model_name = os.getenv("ROUTER_MODEL_NAME", "qwen-max")


tools = [get_weather, web_search, get_current_time, drink_ordering]

llm = ChatOpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
    model_name=model_name,
    max_tokens=512,
    temperature=0.8,
    extra_body={
        "top_k": 20,
        "top_p": 0.95,
        "chat_template_kwargs": {"enable_thinking": False},
    }
)

router_llm = ChatOpenAI(
    api_key=router_openai_api_key,
    base_url=router_openai_api_base,
    model_name=router_model_name,
    max_tokens=256,
    temperature=0.8,
    extra_body={
        "top_k": 20,
        "top_p": 0.95,
        "chat_template_kwargs": {"enable_thinking": False},
    }
)

class PromptLoggerCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        """LLM 调用开始时触发，打印完整 prompt"""
        print("\n===== 发送给大模型的完整 Prompt =====")
        for prompt in prompts:
            print(prompt)  # 打印每个 prompt（通常只有一个）
        print("=====================================\n")

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    callbacks=[PromptLoggerCallback()],
    handle_parsing_errors=True
)


# 定义 State
class State(TypedDict):
    user_input: str
    tools_output: str
    route: str


# 输入节点
def get_user_input_node(state: State) -> State:
    user_input = input()
    return {**state, "user_input": user_input}

# 路由节点
def router_node(state: State) -> State:
    prompt_input = router_prompt.format(query=state["user_input"])
    route = router_llm.invoke(prompt_input).content
    route = route.strip().lower()
    return {**state, "route": route, "tools_output": ""}

# 工具节点
def tools_node(state: State) -> State:
    if state["route"] == "agent":
        tools_output = agent_executor.invoke({"input": state["user_input"]})
        print(tools_output["output"])
    return {**state, "tools_output": tools_output, "route": "tool_chat"}

# 对话节点
def chat_node(state: State) -> State:
    print("当前模式：", state["route"])
    if state["route"] == "chat":
        prompt_input = chat_prompt.format(tools_output=state["tools_output"])
        answer = llm.invoke(prompt_input)
        print(answer.content)
    elif state["route"] == "tool_chat":
        prompt_input = style_prompt.format(tools_output=state["tools_output"])
        answer = llm.invoke(prompt_input)
        print(answer.content)






def build_workflow():
    graph = StateGraph(State)
    graph.add_node("get_user_input", get_user_input_node)
    graph.add_node("router", router_node)
    graph.add_node("tools", tools_node)
    graph.add_node("chat", chat_node)

    graph.set_entry_point("get_user_input")
    graph.add_edge("get_user_input", "router")
    graph.add_edge("tools", "chat")
    graph.add_edge("chat", "get_user_input")



    graph.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "agent": "tools",
            "chat": "chat"
        }
    )
    # 构建工作流
    workflow = graph.compile()

    return workflow


if __name__ == "__main__":
    res = agent_executor.invoke({"input": "北京天气怎么样"})
    print(res["output"])
