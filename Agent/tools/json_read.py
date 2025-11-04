from pydantic import BaseModel, Field, field_validator
from langchain.tools import tool
from difflib import get_close_matches
import requests
import inspect, json

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
def to_tool_spec(tool_obj):
    # 取名称
    name = getattr(tool_obj, "name", None) or getattr(tool_obj, "__name__", "unknown_tool")
    # 取描述（LangChain 的 @tool 会把函数 docstring 收进 tool.description）
    desc = getattr(tool_obj, "description", None) or inspect.getdoc(tool_obj) or ""
    # 取参数 Schema（Pydantic）
    params = getattr(tool_obj, "args_schema", None)
    params_schema = params.model_json_schema() if params else {"type": "object", "properties": {}}

    # OpenAI/Tools 风格
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc, 
            "parameters": params_schema
        }
    }

# 用法：假设 get_weather 是 @tool 返回的对象



if __name__ == "__main__":
    spec = to_tool_spec(get_weather)
    print(json.dumps(spec, ensure_ascii=False, indent=2))
