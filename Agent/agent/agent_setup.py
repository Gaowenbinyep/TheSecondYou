from pathlib import Path

from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI

from agent.prompts import style_prompt, router_prompt
from tools import get_weather

BASE_DIR = Path(__file__).resolve().parents[2]

# 本地部署
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8888/v1"
model_name = str(BASE_DIR / "Saved_models" / "rlhf" / "4B_lora_PPO_V3" / "merged")

llm = ChatOpenAI(
    api_key = openai_api_key,
    base_url = openai_api_base,
    model_name = model_name,
    max_tokens = 256, 
    temperature = 0.8,
    extra_body = {
        "top_k": 20,
        "top_p": 0.95, 
        "chat_template_kwargs": {"enable_thinking": False},
    }
)




tools = [get_weather]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

style_chain = style_prompt | llm
router_chain = router_prompt | llm




if __name__ == "__main__":
    # print(single_chat("你是不是不爱我了"))
    print(agent.run("北京天气怎么样"))
