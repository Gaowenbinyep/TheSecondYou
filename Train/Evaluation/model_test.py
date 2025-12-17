import pandas as pd
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
import asyncio
from pathlib import Path

# 本地部署
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8888/v1"
model_name = "4B_lora_PPO"
result_data_path = "./test_result_8B_300step.json"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


async def model_test(test_data_path, result_data_path):
    # 改用异步客户端
    client = AsyncOpenAI(
        api_key = openai_api_key,
        base_url = openai_api_base
    )
    
    # 异步版本的single_chat
    async def single_chat(query):
        messages = []
        messages.append({"role": "system", "content": """
            你是志明，你要回复你女朋友的消息。
        """})
        messages.append({"role": "user", "content": query})

        chat_response = await client.chat.completions.create(
            model = model_name,
            messages = messages,
            max_tokens = 256, 
            temperature = 0.8, 
            top_p = 0.95, 
            extra_body = {
                "chat_template_kwargs": {"enable_thinking": False},
            }, 
        )
        return chat_response.choices[0].message.content

    datas = pd.read_json(test_data_path, lines=True)
    new_datas = []
    for _, data in tqdm(datas.iterrows(), total=len(datas)):
        query = data["conversations"][1]["content"]
        true_response = data["conversations"][2]["content"]

        response = await single_chat(query)
        new_datas.append({
            "query": query,
            "response": response,
            "true_response": true_response
        })
    pd.DataFrame(new_datas).to_json(result_data_path, orient="records", lines=True, force_ascii=False)



if __name__ == "__main__":
    test_data_path = PROJECT_ROOT / "Train" / "Data" / "eval" / "Single_text.json"
    asyncio.run(model_test(str(test_data_path), result_data_path))
