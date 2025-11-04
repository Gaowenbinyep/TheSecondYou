from langchain.tools import tool
from openai import OpenAI
import requests
import datetime
import os
from pydantic import BaseModel, Field, field_validator
from .drink_ordering.utils import query_drink_db, format_drink_list
from difflib import get_close_matches



class GetWeather(BaseModel):
    city: str = Field(default="北京", description="城市名称")

    @field_validator("city")
    def validate_city(cls, v):
        VALID_CITIES = ["北京", "青岛", "上海", "深圳"]
        if v not in VALID_CITIES:
            match = get_close_matches(v, VALID_CITIES, n=1, cutoff=0.4)
            if match:
                return match[0]   # 自动纠正
            raise ValueError(f"不支持的城市: {v}，只能选择 {VALID_CITIES}")
        return v
@tool(args_schema=GetWeather)
def get_weather(city: str = "北京") -> str:
    """查询城市实时天气"""
    url = f"https://wttr.in/{city}?format=2"
    response = requests.get(url)
    return response.text if response.status_code == 200 else "查询失败"


class WebSearch(BaseModel):
    query: str = Field(default="", description="用户查询内容")
@tool(args_schema=WebSearch)
def web_search(query: str) -> str:
    """
    智能搜索工具，返回搜索结果摘要。
    """
    client = OpenAI(
        api_key=os.getenv("BAIDU_QIANFAN_API_KEY", "BAIDU_QIANFAN_API_KEY_PLACEHOLDER"),  # 千帆AppBuilder平台的ApiKey
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


class DrinkOrdering(BaseModel):
    query: str = Field(default="", description="用户想点的饮品名称或对想查询饮品的期待")
@tool(args_schema=DrinkOrdering)
def drink_ordering(query: str) -> str:
    """
    奶茶点单工具，返回饮品推荐。
    """
    drinks = query_drink_db(query, top_k=3)
    drinks = drinks["documents"][0]
    formatted_drinks = format_drink_list(drinks)
    choice = input("要喝哪个呢？输入序号选择：")
    if choice.isdigit() and 1 <= int(choice) <= len(formatted_drinks):
        selected_drink = formatted_drinks[int(choice) - 1]
        return f"已经为你下单了 {selected_drink.get('饮品名称', '未知')}"
    else:
        return "无效选择，请输入正确的序号。"
    

if __name__ == "__main__":
    # print(get_weather("北京"))
    # print(web_search("第二次世界大战的时间"))
    # print(get_current_time("当前时间"))
    # print(drink_ordering("我想点一杯奶茶"))
    # print(get_weather.args_schema.model_json_schema())
