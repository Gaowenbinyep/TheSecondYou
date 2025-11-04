import json
import os
from pathlib import Path

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions
from langchain.tools import tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parents[3]
TOOLS_DIR = Path(__file__).resolve().parent
EMBEDDING_MODEL_DIR = BASE_DIR.parent / "RAG" / "embedding_model" / "bge-m3"


class BGEEmbedding(EmbeddingFunction):
        def __call__(self, input: Documents) -> Embeddings:
            return hf_emb.embed_documents(input)
        
        def query(self, query: str) -> Embeddings:
            return hf_emb.embed_query(query)

hf_emb = HuggingFaceEmbeddings(
        model_name=str(EMBEDDING_MODEL_DIR),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

client = chromadb.PersistentClient(path=str(TOOLS_DIR / "drink_db"))
db = client.get_collection("drink_menu")
embed_func = BGEEmbedding()

def build_drink_db():
    # 1. 加载饮品 JSON 数据
    with open("drinks.json", "r", encoding="utf-8") as f:
        drinks = json.load(f)

    # 2. 初始化 Chroma 客户端（存储到本地目录 ./drink_db）
    client = chromadb.PersistentClient(path="./drink_db")

    # 3. 设置 Embedding 模型
    embed_func = BGEEmbedding()

    # 4. 创建 / 获取 Collection
    collection = client.get_or_create_collection(
        name="drink_menu",
        embedding_function=embed_func,
        metadata={"description": "奶茶/果茶菜单知识库"}
    )

    # 5. 将饮品数据插入向量库
    docs = []
    metadatas = []
    ids = []

    for i, drink in tqdm(enumerate(drinks), desc="导入饮品数据"):
        # 拼接检索文本
        doc_text = f"饮品名称: {drink['name']}\n介绍: {drink['intro']}"
        if "options" in drink:
            for k, v in drink["options"].items():
                doc_text += f"\n{k}: {', '.join(v)}"
        
        docs.append(doc_text)
        metadatas.append({k: json.dumps(v) if isinstance(v, dict) else v for k, v in drink.items()})
        ids.append(f"drink_{i}")

    # 批量插入
    collection.add(
        documents=docs,
        metadatas=metadatas,
        ids=ids
    )

    print(f"✅ 成功导入 {len(drinks)} 条饮品数据到 Chroma 向量库！")

def query_drink_db(query: str, top_k: int = 3):
    results = db.query(
        query_embeddings=embed_func.query(query),
        n_results=top_k,
    )
    return results

def format_drink_list(drink_list):
    formatted_list = []

    for item in drink_list:
        lines = item.split("\n")
        drink_dict = {}
        for line in lines:
            if ": " in line:
                key, value = line.split(": ", 1)
                drink_dict[key] = value
        formatted_list.append(drink_dict)

    for i, drink in enumerate(formatted_list, 1):
        print(f"=== 奶茶推荐 {i} ===")
        print(f"名称: {drink.get('饮品名称', '')}")
        print(f"介绍: {drink.get('介绍', '')}")
        if 'size' in drink: print(f"容量: {drink['size']}")
        if 'sugar' in drink: print(f"糖度: {drink['sugar']}")
        if 'ice' in drink: print(f"温度/冰块: {drink['ice']}")
        if 'topping' in drink: print(f"配料: {drink.get('topping', '无')}")
        print("\n")
    return formatted_list
if __name__ == "__main__":
    drinks = query_drink_db("我想喝点苦苦的东西", top_k=5)
    drinks = drinks["documents"][0]
    formatted_drinks = format_drink_list(drinks)
