from openai import OpenAI, AsyncOpenAI
import json
SYSTEM = """
    你是高文彬，你要回复你女朋友的消息。
    你要参照示例学习高文彬作为男朋友的语言风格，保持风格个性化且有回应性。
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
    10. 好叭好叭你说了算！"""


# 本地部署
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8888/v1"
model_name = "./Saved_models/sft/1.7B_full_V3"

# 云端调用
# openai_api_key="sk-************************"
# openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
# model_name = "qwen-plus"

client = OpenAI(
    api_key = openai_api_key,
    base_url = openai_api_base
)

def single_chat(query):
    messages = []
    messages.append({"role": "system", "content": SYSTEM})
    messages.append({"role": "user", "content": query})
    chat_response = client.chat.completions.create(
        model = model_name,
        messages = messages,
        max_tokens = 256, 
        temperature = 0.8, 
        top_p = 0.95, 
        extra_body = {
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        }, 
    )
    return chat_response.choices[0].message.content

async def async_single_chat(query):  # Added async keyword
    client = AsyncOpenAI(      # Changed to AsyncClient
        api_key=openai_api_key,
        base_url=openai_api_base
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": query}
    ]
    chat_response = await client.chat.completions.create(  # Added await
        model=model_name,
        messages=messages,
        max_tokens=256,
        temperature=0.8,
        top_p=0.95,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return chat_response.choices[0].message.content

def multi_chat():
    messages = []
    messages.append({"role": "system", "content": SYSTEM})
    history_len = 20
    print("输入回车可退出对话。")
    while True:
        # if len(messages) >= history_len:
        #     messages.pop(1)
        #     messages.pop(1)
        query = input("请输入对话: ").strip()
        if query == "\n":
            break
        messages = [{"role": "system", "content": SYSTEM}]
        messages.append({"role": "user", "content": query})
        try:
            chat_response = client.chat.completions.create(
                model = model_name,
                messages = messages,
                max_tokens = 256, 
                temperature = 0.6, 
                top_p = 0.95, 
                extra_body = {
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                }, 
            )
            response = chat_response.choices[0].message.content
            print(response)
            messages.append({"role": "assistant", "content": response})
        except Exception as e:
            print(f"发生错误: {e}")
            continue

if __name__ == "__main__":
    # print(single_chat("你是不是不爱我了"))
    multi_chat()
