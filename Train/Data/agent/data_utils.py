import random
import os
import sys
import json
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from typing import Dict, List
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "Agent" / "tools"))
from tools import (
    drink_ordering,
    get_weather,
    get_hot_search,
    get_horoscope
)

api_key = os.environ.get("DASHSCOPE_API_KEY")
if not api_key:
    raise RuntimeError("环境变量 DASHSCOPE_API_KEY 未设置")
client = OpenAI(
    api_key=api_key,
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
model_name="qwen-plus"

def generate_reverse_queries_for_drink(
    drink: Dict,
    n_queries: int = 10,
) -> List[Dict]:
    """
    给定一杯奶茶，生成 n 条反向查询，并输出你要的数据格式：
    {
        "query": "...",
        "attributes": {
            "drink_name": "...",
            "options": {...}
        }
    }
    """
    name = drink.get("name", "")
    options = drink.get("options", {})

    options_text = json.dumps(options, ensure_ascii=False, indent=2)

    prompt = f"""
        你是一个奶茶推荐系统的数据生成助手。请根据一杯奶茶的信息，为它生成 {n_queries} 条自然语言的“用户需求”（query）。

        这些 query 的用途，是为了训练一个可以根据用户描述推荐奶茶的模型。

        【奶茶信息】
        名称: {name}
        可选属性(JSON):
        {options_text}

        生成要求：
        1. 输出 JSON 数组，每一项都是一个字符串 query。
        2. 每条 query 要表现出适合推荐这杯奶茶的需求（如口味、冰度、糖度、风格、描述偏好等）。
        3. query 语言自然、像真实用户说话。
        4. 不要输出解释说明，只输出 JSON 数组。

        示例格式：
        [
        "我今天想喝点特别清爽的绿茶奶盖",
        "来一杯超级冰的奶茶，甜一点"
        ]
    """

    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=128,
        temperature=0.9,
        top_p=0.95,
        extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
    )

    content = resp.choices[0].message.content

    # 尝试解析 JSON 数组
    try:
        queries = json.loads(content)
        queries = [q for q in queries if isinstance(q, str)]
    except Exception:
        # 兜底策略：按行拆
        queries = [line.strip() for line in content.splitlines() if line.strip()]

    # ▼ 这里产出你要的数据结构
    results = []
    for q in queries:
        results.append({
            "query": q,
            "attributes": {
                "drink_name": name,
                "options": options
            }
        })

    return results


def build_drink_reverse_query_dataset(
    json_path: str,
    n_queries_per_drink: int = 1,
    output_path: str | None = None,
):
    """
    读奶茶 JSON 文件，为每杯奶茶生成反向查询，并以 JSON 数据对输出：
    { "query": xxx, "attributes": { "drink_name": xxx, "options": {...} } }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        drinks = json.load(f)

    dataset = []

    for drink in tqdm(drinks, desc="Generating reverse queries"):
        pairs = generate_reverse_queries_for_drink(drink, n_queries_per_drink)
        dataset.extend(pairs)

    # 如果提供 output_path → 存成 JSONL
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return dataset



def drink_order_data(query, attributes):
    prompt = f"""
        你是一个奶茶推荐机器人的数据生成助手。你的任务是：

        1. 根据用户需求，从可选属性中选择最合适的属性（只从给定列表中选，不得编造）。
        2. 输出饮品名称 + 已选属性，用自然语言组成一句点单话。

        【用户需求】
        {query}

        【可选属性（包含饮品名称与各属性可选项）】
        {attributes}

        请严格遵守规则：
        1. 所有选择必须来自可选属性，不能增加类别或选项。
        2. 除奶茶名称外，用户未提到的属性可以不选择。
        3. 最终必须输出一句自然语言点单结果，格式类似：
        为你点好了<属性>，...，<属性>的<饮品名称>

        只输出最终一句话，不要解释过程。
    """

    chat_response = client.chat.completions.create(
        model = model_name,
        messages = [{"role": "user", "content": prompt}],
        max_tokens = 64, 
        temperature = 0.8, 
        top_p = 0.95, 
        extra_body = {
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        }, 
    )
    return chat_response.choices[0].message.content

def weather_data():
    citys = [
        "北京", "上海", "广州", "深圳", "天津", "重庆", "杭州", "南京", "成都", "武汉",
        "西安", "青岛", "沈阳", "大连", "长春", "哈尔滨", "济南", "郑州", "石家庄", "合肥",
        "福州", "厦门", "南昌", "长沙", "昆明", "贵阳", "南宁", "海口", "兰州", "西宁",
        "拉萨", "乌鲁木齐", "呼和浩特", "太原", "包头", "鄂尔多斯", "唐山", "秦皇岛", "保定", "张家口",
        "邯郸", "临沂", "烟台", "威海", "淄博", "东营", "泰安", "聊城", "洛阳", "开封",
        "宜昌", "襄阳", "十堰", "芜湖", "蚌埠", "马鞍山", "淮南", "银川", "桂林", "北海",
        "柳州", "珠海", "中山", "佛山", "惠州", "汕头", "汕尾", "江门", "温州", "宁波",
        "嘉兴", "绍兴", "金华", "台州", "镇江", "常州", "无锡", "苏州", "扬州", "盐城",
        "南通", "连云港", "徐州", "咸阳", "宝鸡", "榆林", "安阳", "新乡", "许昌", "平顶山",
        "绵阳", "德阳", "宜宾", "泸州", "南充", "遵义", "安顺", "三亚", "三明", "漳州"
    ]

    city_name = random.choice(citys)
    user_query = f"帮我看一下{city_name}现在什么天气，要不要多穿点？"

    # 工具调用
    tool_result = get_weather.invoke({"city": city_name})

    # 让大模型把“用户问题 + 工具结果”组织成最终回复
    prompt = f"""
        你是一个聊天机器人的数据生成助手。现在已经通过天气工具拿到了查询结果，请你基于“用户问题”和“工具返回结果”，生成一条发给用户的最终回复。

        【用户问题】
        {user_query}

        【工具返回结果】
        {tool_result}

        请遵守以下规则：
        1. 所有关于天气的描述必须来自工具返回结果，不得编造温度、天气情况等数据。
        2. 回复里要点名城市“{city_name}”。
        3. 只输出最终给用户看的回复，不要解释工具调用过程，不要输出任何系统说明。
    """

    chat_response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.8,
        top_p=0.95,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return chat_response.choices[0].message.content

def hot_search_data():
    # 模拟用户问题
    user_query = "现在微博上都在聊什么？帮我看看热搜里有什么有意思的。"

    # 工具调用
    tool_result = get_hot_search.invoke({})

    prompt = f"""
        你是一个聊天机器人的数据生成助手。现在已经通过“微博热搜工具”拿到了热搜列表，请你基于“用户问题”和“工具返回结果”，生成一条发给用户的最终回复。

        【用户问题】
        {user_query}

        【工具返回结果（热搜词列表）】
        {tool_result}

        请遵守以下规则：
        1. 只能基于给出的热搜词进行回答，不得凭空创造不存在的热搜话题。
        2. 可以帮用户简单概括/串联一下这些热搜，比如用“今天热搜上主要是这些：...”的方式。
        3. 可以适当加一点轻松的语气或简单评价，但不要胡编细节（比如具体事件内容、当事人言论等）,回复简短些。
        4. 挑 2 个热搜词说一说，不需要把全部都列出来。
        5. 只输出最终给用户看的回复，不要解释工具调用过程，不要输出任何系统说明。
    """

    chat_response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.8,
        top_p=0.95,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return chat_response.choices[0].message.content

def horoscope_data():
    signs = [
        "aries", "taurus", "gemini", "cancer", "leo", "virgo",
        "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces"
    ]
    sign_name = random.choice(signs)

    user_query = f"帮我看看今天{sign_name}的运势怎么样？"

    # 工具调用
    tool_result = get_horoscope.invoke({"sign": sign_name})

    prompt = f"""
        你是一个聊天机器人的数据生成助手。现在已经通过“星座运势工具”拿到了 {sign_name} 的当日运势文案，请你基于“用户问题”和“工具返回结果”，生成一条发给用户的最终回复。

        【用户问题】
        {user_query}

        【工具返回结果（英文运势原文）】
        {tool_result}

        请遵守以下规则：
        1. 回复内容需要基于工具返回的运势信息，用自己的话做适当概括，但不要改变整体倾向（比如好/一般/需要注意等）。
        2. 必须用中文回复，不得捏造运势走向，总体回复简短些。
        5. 只输出最终给用户看的回复，不要解释工具调用过程，不要输出任何系统说明。
    """

    chat_response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.8,
        top_p=0.95,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return chat_response.choices[0].message.content



def main():
    pass

if __name__ == "__main__":
    main()
    # print(drink_order_data())
    # print(hot_search_data())
    # print(horoscope_data())
    # print(weather_data())
