import asyncio
from pathlib import Path

import pandas as pd
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent

# 本地部署
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8888/v1"
model_name = str(BASE_DIR / "Saved_models" / "rlhf" / "4B_lora_PPO_V3" / "merged")
# model_name = str(BASE_DIR / "Base_models" / "Qwen3-1.7B")


async def model_test(test_data_path, result_data_path):
    # 改用异步客户端
    client = AsyncOpenAI(
        api_key = openai_api_key,
        base_url = openai_api_base
    )
    
    # 异步版本的single_chat
    async def single_chat(query):
        messages = []
        # messages.append({"role": "system", "content": "你是<your_name>，你要回复你女朋友的消息。"})
        messages.append({"role": "system", "content": """
        你是<your_name>，你要回复你女朋友的消息。
        你要参照示例学习<your_name>作为男朋友的语言风格，保持风格个性化且有回应性。
        ---
        【语言风格示例】（请模仿此风格）：
        1. 诶嘿 好想你哦
        2. 记得吃点甜甜的缓一缓叭
        3. 小打工仔今天很乖呐
        4. 不准再不开心啦🙅🏻‍♂️
        5. 要是我在就给你揉揉头啦
        6. 忙完奖励自己点好吃的哦！
        7. 我的宝藏女孩今天也好棒呀！
        8. 哈哈哈哈你也太可爱了吧
        9. 我滴宝藏宝宝😊
        10. 好叭好叭你说了算！
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
    test_path = BASE_DIR / "data" / "eval" / "Single_text.json"
    output_path = EVAL_DIR / "test_result_V5.json"
    asyncio.run(model_test(str(test_path), str(output_path)))
