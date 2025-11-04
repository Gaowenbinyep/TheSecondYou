import pandas as pd
import os
import json
from tqdm import tqdm
from openai import AsyncOpenAI
import re
from evaluation.llm_eva import evaluation_prompt

os.environ.setdefault("DASHSCOPE_API_KEY", "REPLACE_WITH_DASHSCOPE_KEY")

# ======================================
# 配置区
MODEL = "qwen-plus"  # 替换成可用的模型名
DEV_FILE = "./data/test_1.json"
EVALUATION_FILE = "./data/test_2.json"
# ======================================


async def call_qwen(prompt, semaphore):
    """ 异步调用 Qwen3-325B """
    async with semaphore:
        client = AsyncOpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        completion = await client.chat.completions.create(
            model=MODEL,
            messages=prompt,
            extra_body={"enable_thinking": False},
        )
        return completion.choices[0].message.content

async def process_data(data, semaphore, pbar):
    user_query = data["query"]
    assistant_response = data["response"]
    prompt = evaluation_prompt(user_query, assistant_response)
    result = await call_qwen(prompt, semaphore)
    
    # 原有解析逻辑保持不变...
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", result)
    if match:
        result = match.group(1)
    
    pbar.update(1)
    return {
        "conversations": [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": assistant_response}
        ],
        "assistant_review": json.loads(result)
    }

if __name__ == "__main__":
    import asyncio
    datas = pd.read_json(DEV_FILE, lines=True, orient="records")
    
    # 控制并发量（根据API限制调整）
    semaphore = asyncio.Semaphore(10)
    
    with tqdm(total=len(datas), desc="处理进度") as pbar:
        loop = asyncio.get_event_loop()
        tasks = [process_data(data, semaphore, pbar) for _, data in datas.iterrows()]
        results = loop.run_until_complete(asyncio.gather(*tasks))
        
        with open(EVALUATION_FILE, "w") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
