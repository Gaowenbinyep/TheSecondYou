import pandas as pd
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
import asyncio

# 本地部署
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8888/v1"
model_name = "your_model_name"
result_data_path = "./test_result_Baseline_8B.json"


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
            你是<your_name>，你要回复你女朋友的消息。
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
    asyncio.run(model_test("/media/a822/82403B14403B0E83/志明/WechatRobot/data/eval/Single_text.json", result_data_path))

