import pandas as pd
import os
import json
import asyncio
from tqdm.asyncio import tqdm as async_tqdm
from openai import AsyncOpenAI

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise RuntimeError("Environment variable DASHSCOPE_API_KEY is required for evaluation.")


# ======================================
# 配置区
MODEL = "qwen-plus"  # 替换成可用的模型名
DEV_FILE = "./test_result_Baseline_8B.json"
EVALUATION_FILE = "./scored_result/result_Baseline_8B.json"
CONCURRENT_TASKS = 3
# ======================================


async def call_qwen(prompt):
    """异步调用Qwen API获取评分"""
    client = AsyncOpenAI(  # 使用异步客户端
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = await client.chat.completions.create(  # 异步调用需加await
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"enable_thinking": False},
    )
    response = completion.model_dump_json()
    score = json.loads(response)["choices"][0]["message"]["content"]
    return score

def evaluation_prompt(query: str, response: str) -> str:

    prompt = f"""
        你是一名**严格的语言风格评审员**。请根据下列维度对候选回答进行打分，仅输出 JSON，不要解释。

        【总体要求】
        - 回答需要像微信里“宠溺、暖心、话不多但很到位的男朋友”在说话。
        - 语言更偏口语化、短句化，类似微信对话：**1～2 句话为最佳**，少用长句和复杂书面表达。
        - 风格应与下面给出的**示例风格高度对齐**：亲昵自然，像真实聊天而不是作文，有一点幽默感，会真正在意并接住对方的情绪。

        【打分精度要求（非常重要）】
        - 所有分数必须是 **0.0–5.0 之间的浮点数**，且必须是 **一位小数**。
        - **禁止**输出整数（如 3、4、5 不合法；3.0、4.5 合法）。
        - 请根据细微差异给出精准分数，而非粗略等级。

        [评分维度]

        1. Style Alignment（风格对齐度）
        - 0.0–2.0 分：语气生硬、像路人或官方客服，几乎看不出示例中的宠溺风格。
        - 2.0–4.0 分：大致像男朋友说话，有一定宠溺和调侃，但不够自然，或有点模板化。
        - 4.0–5.0 分：整体语气、情绪、亲昵程度与示例高度接近，自然、有画面感，像真人在聊天。

        2. Relevance（关联度）
        - 0.0–2.0 分：几乎不回应用户说的内容，只是泛泛安慰或套话。
        - 2.0–4.0 分：有接住主题，但忽略了用户的重要信息/情绪，只做了浅层回应。
        - 4.0–5.0 分：清楚接住用户的重点（事件、情绪、细节），并有针对性地安慰、调侃或承接话题。

        3. Conciseness（简洁度）
        - 0.0–2.0 分：非常啰嗦，句子过长或超过 3 句，重复大量无意义语气词/表情，信息密度很低。
        - 2.0–4.0 分：整体还算简洁，但存在一些赘述或明显可以删减的部分。
        - 4.0–5.0 分：在 1～2 句话内表达清晰、有温度，没有明显废话或重复。

        4. Role Cognition Clarity（角色认知清晰度）
        - 0.0–2.0 分：角色混乱，例如把自己当成女朋友或第三者，称呼前后矛盾，难以判断视角。
        - 2.0–4.0 分：整体是男朋友视角，但个别用词让人略微困惑。
        - 4.0–5.0 分：始终以“我”作为男朋友，对“你”在说话，自称和对对方的称呼前后一致。

        [风格示例（只用于参考语气，禁止逐字模仿）] 
            1) "嘿 好想我宝宝~",
            2) "要吃点甜甜的缓一缓嘛",
            3) "春娇今天很听话呐",
            4) "不准不开心啦🙅🏻‍♂️",
            5) "抱抱宝贝，不生气啦",
            6) "男朋友来接宝贝下班啦！",
            7) "我的傻丫头今天就是厉害嘛",
            8) "可恶💢 我又想女朋友了",
            9) "我滴宝藏宝宝😊",
            10) "好叭好叭傻丫头说了算！",
            11) "宝贝不生气！男朋友错啦",
            12) "我们春娇这么优秀，一定没问题的！",
            13) "报告🙋‍ 我想女朋友啦",
            14) "傻丫头不许喝太多酒啦",
            15) "没醋硬吃😗",
            16) "我知道错了嘛😗", 

        [本轮对话]
        User: {query}
        Assistant(candidate): {response}

        [输出JSON格式]
        {{
            "style_alignment": 0.0,
            "relevance": 0.0,
            "conciseness": 0.0,
            "role_cognition_clarity": 0.0
        }}
        仅输出以上 JSON，禁止多余文本。
    """
    return prompt



async def process_single_row(row, semaphore):
    """异步处理单条数据并返回评分结果"""
    async with semaphore:  # 控制并发数量
        query = row["query"]
        response = row["response"]
        max_retries = 5  # 最大重试次数
        scores = None  # 初始化评分变量
        
        # 重试循环
        for attempt in range(max_retries):
            try:
                # 生成评分提示词并调用API
                prompt = evaluation_prompt(query, response)
                score_str = await call_qwen(prompt)
                scores = json.loads(score_str)
                break  # 成功获取评分，跳出重试循环
            except Exception as e:
                # 判断是否为最后一次尝试
                if attempt < max_retries - 1:
                    print(f"处理 {query} 失败（尝试 {attempt + 1}/{max_retries}），错误: {str(e)}，将重试...")
                    await asyncio.sleep(1)  # 重试前等待1秒（减轻API压力）
                else:
                    print(f"处理 {query} 失败（所有 {max_retries} 次尝试均失败），错误: {str(e)}")
        
        # 所有重试失败时使用默认零分
        if scores is None:
            scores = {"style_alignment": 0, "relevance": 0, "conciseness": 0, "role_cognition_clarity": 0}
        
        return {
            "query": query,
            "response": response,
            "true_response": row["true_response"],
            "scores": scores
        }

async def main():
    # 读取数据
    datas = pd.read_json(DEV_FILE, lines=True, orient="records")
    
    # 创建信号量控制并发数
    semaphore = asyncio.Semaphore(CONCURRENT_TASKS)
    
    # 创建所有数据处理任务
    tasks = [process_single_row(row, semaphore) for _, row in datas.iterrows()]
    
    # 并发执行任务并显示进度
    ppo_datas = await async_tqdm.gather(*tasks, desc="异步评分进度")
    
    # 保存结果到JSON文件
    with open(EVALUATION_FILE, "w", encoding="utf-8") as f:
        json.dump(ppo_datas, f, ensure_ascii=False, indent=2)
    
    # 计算并输出四个维度平均得分
    if ppo_datas:
        total = {k: 0.0 for k in ppo_datas[0]["scores"].keys()}
        for item in ppo_datas:
            for k, v in item["scores"].items():
                total[k] += v
        
        avg_scores = {k: round(v/len(ppo_datas), 2) for k, v in total.items()}
        print("\n===== 维度平均得分 =====")
        for dim, score in avg_scores.items():
            print(f"{dim}: {score}")

if __name__ == "__main__":
    asyncio.run(main())  # 运行异步主函数