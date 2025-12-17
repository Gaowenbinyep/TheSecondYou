from langchain.tools import tool
from openai import OpenAI
import os
import requests
import datetime
from pydantic import BaseModel, Field, field_validator
from drink_ordering.utils import query_drink_db, format_drink_list
from difflib import get_close_matches

class WebSearch(BaseModel):
    query: str = Field(default="", description="用户查询内容")
@tool(args_schema=WebSearch)
def web_search(query: str) -> str:
    """
    智能搜索工具，返回搜索结果摘要。
    """
    api_key = os.environ.get("QIANFAN_API_KEY")
    if not api_key:
        raise RuntimeError("环境变量 QIANFAN_API_KEY 未设置")
    client = OpenAI(
        api_key=api_key,  # 千帆AppBuilder平台的ApiKey
        base_url="https://qianfan.baidubce.com/v2/ai_search") # 智能搜索生成V2版本接口
    response = client.chat.completions.create(
        model="ernie-4.0-turbo-8k",
        messages=[
            {"role": "user", "content": query}
        ],
        stream=False
    )
    return response.choices[0].message.content

class GetCurrentTime(BaseModel):
    input: str = Field(default=None, description="用户输入")
@tool(args_schema=GetCurrentTime)
def get_current_time(input: str = None) -> str:
    """获取当前时间"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class GetWeather(BaseModel):
    city: str = Field(default="北京", description="城市名称")
    # @field_validator("city")
    # def validate_city(cls, v):
    #     VALID_CITIES = ["北京", "青岛", "上海", "深圳"]
    #     if v not in VALID_CITIES:
    #         match = get_close_matches(v, VALID_CITIES, n=1, cutoff=0.4)
    #         if match:
    #             return match[0]   # 自动纠正
    #         raise ValueError(f"不支持的城市: {v}，只能选择 {VALID_CITIES}")
    #     return v
@tool(args_schema=GetWeather)
def get_weather(city: str = "北京") -> str:
    """查询城市实时天气"""
    url = f"https://wttr.in/{city}?format=2"
    response = requests.get(url)
    return response.text if response.status_code == 200 else "查询失败"

class DrinkOrdering(BaseModel):
    query: str = Field(default="", description="用户想点的饮品名称或对想查询饮品的期待")
@tool(args_schema=DrinkOrdering)
def drink_ordering(query: str) -> str:
    """
    奶茶点单工具，返回饮品推荐（自动跳过空字段）。
    """
    drinks = query_drink_db(query, top_k=1)
    drink = format_drink_list(drinks["documents"][0])[0]

    name = drink.get("饮品名称", "")
    size = drink.get("size", "")
    sugar = drink.get("sugar", "")
    ice = drink.get("ice", "")
    topping = drink.get("topping", "")

    lines = []
    if name:
        lines.append(f"名称: {name}")
    if size:
        lines.append(f"容量可选: {size}")
    if sugar:
        lines.append(f"糖度可选: {sugar}")
    if ice:
        lines.append(f"温度/冰块可选: {ice}")
    if topping:
        lines.append(f"配料可选: {topping}")

    return "\n".join(lines)

class GetHotSearch(BaseModel):
    pass
@tool(args_schema=GetHotSearch)
def get_hot_search() -> str:
    """
        获取当前微博热门搜索关键词
    """
    url = "https://weibo.com/ajax/side/hotSearch"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://weibo.com/",
        "Accept": "application/json, text/plain, */*",
        "X-Requested-With": "XMLHttpRequest",
    }
    
    resp = requests.get(url, headers=headers, timeout=5)
    data = resp.json()

    hot_list = data["data"]["realtime"]
    return [item["word"] for item in hot_list[:3]]


class GetHoroscope(BaseModel):
    sign: str = Field(default="sign", description="要查询的星座名称，必须从以下星座名称中选择：aries, taurus, gemini, cancer, leo, virgo, libra, scorpio, sagittarius, capricorn, aquarius, pisces")

@tool(args_schema=GetHoroscope)
def get_horoscope(sign: str = "scorpio"):
    """获取指定星座的每日运势"""
    url = f"https://ohmanda.com/api/horoscope/{sign}/"
    r = requests.get(url).json()
    return r.get("horoscope", "获取失败")



if __name__ == "__main__":
    # print(get_weather("北京"))
    # print(web_search("第二次世界大战的时间"))
    # print(get_current_time("当前时间"))
    # print(drink_ordering.invoke("我想点一杯有红枣的奶茶"))
    # print(get_weather.args_schema.model_json_schema())
    # print(get_horoscope())
    print(get_hot_search())
