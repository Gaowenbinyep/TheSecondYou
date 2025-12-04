from openai import OpenAI, AsyncOpenAI
import json
SYSTEM = """
    你是高文彬，你要回复你女朋友的消息。
"""


# 本地部署
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8888/v1"
model_name = "your_model_name"

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