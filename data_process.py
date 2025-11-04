import pandas as pd
import os
import json
import re
import asyncio
# from tqdm import tqdm
from tqdm.asyncio import tqdm
from openai import OpenAI, AsyncOpenAI

from PROMPT import score_prompt, detection_prompt, parse_score_result, parse_detection_result, optimize_prompt, parse_optimize_result

os.environ.setdefault("DASHSCOPE_API_KEY", "REPLACE_WITH_DASHSCOPE_KEY")

# ======================================
# 配置区
MODEL = "qwen-plus"  # 替换成可用的模型名
INPUT_FILE = "./data/JUN_train.json"
OUTPUT_FILE = "./data/score_train.json"
CLEAN_OUTPUT_FILE = "./data/final_train.json"
PPO_RM_TRAIN = "./data/ppo_rm_train.json"
PPO_TRAIN = "./data/ppo_train.json"
TEST_0 = "./data/JUN_dev.json"
TEST_1 = "./data/test_1.json"
TEST_2 = "./data/test_2.json"


GEN_MULTI_PATH = "./data/LCCC/sharegpt_format.json"
GEN_SINGLE_PATH = "./data/Gen_single_train.json"
GEN_TARGET_PATH = "./data/v1.0/Gen_single_train.json"

MULTI_PATH = "./data/Multi_train.json"
SINGLE_PATH = "./data/Single_train.json"



GENERATION_QUERY_FILE = "./data/generation_querys.json"
# ======================================


def call_qwen(prompt):
    """ 调用 Qwen3-325B 返回结果 """
    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"enable_thinking": False},
    )
    response = completion.model_dump_json()
    score = json.loads(response)["choices"][0]["message"]["content"]
    return score

async def async_call_qwen(prompt):
    """ 异步调用 Qwen3-325B """
    client = AsyncOpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = await client.chat.completions.create(
        model=MODEL,
        messages=prompt,
        temperature=0.0,
        top_p=1.0,
        extra_body={"enable_thinking": False},
    )
    return completion.choices[0].message.content



def Mutli2Single(mutli_path, single_path):
    datas = pd.read_json(mutli_path)
    single_datas = []
    for _, data in datas.iterrows():
        conversations = data["conversations"]
        for i in range(len(conversations)):
            if conversations[i]["role"] == "user":
                query = conversations[i]["content"]
            if conversations[i]["role"] == "assistant":
                response = conversations[i]["content"]
                if len(query) >= 4:
                    single_datas.append({
                        "conversations": [
                            {
                                "role":"system",
                                "content": "你是高文彬，一个聪明、热情、幽默的人，你的回复简洁但温柔，习惯用“叭”、“呐”、“诶嘿”等口头语"
                            },
                            {
                                "role": "user",
                                "content": query.replace("\n", "，")
                            },
                            {
                                "role": "assistant",
                                "content": response.replace("\n", "，")
                            }
                        ]
                    })
    pd.DataFrame(single_datas).to_json(single_path, orient="records", lines=True, force_ascii=False)



async def conversation_score(single_path):
    datas = pd.read_json(single_path, lines=True)
    scored_datas = []
    semaphore = asyncio.Semaphore(10)  # 控制最大并发量
    
    # 异步处理单个对话
    async def process_conversation(idx, data):
        async with semaphore:
            conversations = data["conversations"]
            query = conversations[1]["content"]
            response = conversations[2]["content"]
            prompt = score_prompt(query, response)
            retry = 0
            score = None
            
            while retry < 3 and score is None:
                try:
                    # 调用异步API（注意传递消息对象列表）
                    score_content = await async_call_qwen([{"role": "user", "content": prompt}])
                    score = parse_score_result(score_content)
                except Exception as e:
                    print(f"对话 {idx} 处理失败: {str(e)}")
                
                retry += 1
                if score is None and retry < 3:
                    await asyncio.sleep(1)
            
            return {
                "conversations": conversations,
                "score": score
            }
    
    # 创建任务列表
    tasks = [process_conversation(idx, data) for idx, data in datas.iterrows()]
    
    # 异步进度条：使用tqdm.asyncio.gather包装任务
    scored_datas = await tqdm.gather(
        *tasks,
        total=len(tasks),
        desc="异步处理对话评分"
    )
    
    # 过滤无效结果并保存
    valid_datas = [item for item in scored_datas if item is not None]
    pd.DataFrame(valid_datas).to_json(
        single_path,
        orient="records",
        lines=True,
        force_ascii=False
    )

async def conversation_detection(single_path):
    datas = pd.read_json(single_path, lines=True)
    detection_datas = []
    semaphore = asyncio.Semaphore(10)  # 控制最大并发量
    
    # 异步处理单个对话
    async def process_conversation(idx, data):
        async with semaphore:
            conversations = data["conversations"]
            query = conversations[1]["content"]
            response = conversations[2]["content"]
            prompt = detection_prompt(query, response)
            retry = 0
            # score = data["score"]
            detection = None
            
            while retry < 3 and detection is None:
                try:
                    # 调用异步API（注意传递消息对象列表）
                    detection_content = await async_call_qwen([{"role": "user", "content": prompt}])
                    detection = parse_detection_result(detection_content)
                except Exception as e:
                    print(f"对话 {idx} 处理失败: {str(e)}")
                
                retry += 1
                if detection is None and retry < 3:
                    await asyncio.sleep(1)
            
            # return {
            #     "conversations": conversations,
            #     "score": score,
            #     "detection": detection
            # }
            return {
                "conversations": conversations,
                "detection": detection
            }
    
    # 创建任务列表
    tasks = [process_conversation(idx, data) for idx, data in datas.iterrows()]
    
    # 异步进度条：使用tqdm.asyncio.gather包装任务
    detection_datas = await tqdm.gather(
        *tasks,
        total=len(tasks),
        desc="异步处理对话评分"
    )
    
    # 过滤无效结果并保存
    valid_datas = [item for item in detection_datas if item is not None]
    pd.DataFrame(valid_datas).to_json(
        single_path,
        orient="records",
        lines=True,
        force_ascii=False
    )

def data_select(single_path, selected_path):
    datas = pd.read_json(single_path, lines=True)
    selected_datas = []
    for _, data in tqdm(datas.iterrows(), total=len(datas), desc="处理进度"):
        score = data["score"]
        detection = data["detection"]
        system = data["conversations"][0]["content"]
        query = data["conversations"][1]["content"]
        if score:
            response = score["concise_response"] if score["has_redundancy"] else data["conversations"][2]["content"]
            average_score = score["relevance_score"]*0.4 + score["effectiveness_score"]*0.4 + score["coherence_score"]*0.2
        else:
            response = data["conversations"][2]["content"]
            average_score = 0
        if detection == None and average_score >= 8:
            detection = {"suggestion": "保留"}
        else:
            detection = {"suggestion": "删除"}
        if average_score >= 8 or detection["suggestion"]=="保留":

            selected_datas.append({
                "conversations": [
                    {
                        "role": "system",
                        "content": system
                    },
                    {
                        "role": "user",
                        "content": query
                    },
                    {
                        "role": "assistant",
                        "content": response
                    }
                ]
            })
    pd.DataFrame(selected_datas).to_json(selected_path, orient="records", lines=True, force_ascii=False)

async def data_generation(source_path, target_path):
    datas = pd.read_json(source_path, lines=True)
    scored_datas = []
    semaphore = asyncio.Semaphore(10)  # 控制最大并发量
    
    # 异步处理单个对话
    async def process_conversation(idx, data):
        async with semaphore:
            conversations = data["conversations"]
            query = conversations[1]["content"]
            response = conversations[2]["content"]
            prompt = optimize_prompt(query, response)
            retry = 0
            optimized_response = None
            while retry < 3 and optimized_response is None:
                try:
                    # 调用异步API（注意传递消息对象列表）
                    optimize_content = await async_call_qwen([{"role": "user", "content": prompt}])
                    optimized_response = parse_optimize_result(optimize_content)
                except Exception as e:
                    print(f"对话 {idx} 处理失败: {str(e)}")
                
                retry += 1
                if optimized_response is None and retry < 3:
                    await asyncio.sleep(1)
            
            return {
                "conversations": conversations,
                "optimized_response": optimized_response
            }
    
    # 创建任务列表
    tasks = [process_conversation(idx, data) for idx, data in datas.iterrows()]
    
    # 异步进度条：使用tqdm.asyncio.gather包装任务
    scored_datas = await tqdm.gather(
        *tasks,
        total=len(tasks),
        desc="异步处理对话评分"
    )
    
    # 过滤无效结果并保存
    valid_datas = [item for item in scored_datas if item is not None]
    pd.DataFrame(valid_datas).to_json(
        target_path,
        orient="records",
        lines=True,
        force_ascii=False
    )
if __name__ == "__main__":
    # datas = pd.read_json("./data/Gen_single_train.json", lines=True)
    # new_data = []
    # for _, data in datas.iterrows():
    #     conversations = data["conversations"]
    #     system = conversations[0]["content"]
    #     query = conversations[1]["content"]
    #     optimized_response = data["optimized_response"]
    #     if optimized_response:
    #         if optimized_response["is_applicable"]:
    #             response = optimized_response["rewritten_text"]
    #         else:
    #             continue
    #     if len(query) >= 4:
    #         new_data.append({
    #             "conversations": [
    #                 {
    #                     "role":"system",
    #                     "content": system
    #                 },
    #                 {
    #                     "role": "user",
    #                     "content": query
    #                 },
    #                 {
    #                     "role": "assistant",
    #                     "content": response
    #                 }
    #             ]
    #         })
    # pd.DataFrame(new_data).to_json("./data/v1.0/LCCC_single_train.json", orient="records", lines=True, force_ascii=False)
    data_select("./data/Single_train.json", "./data/v2.0/Single_train.json")
