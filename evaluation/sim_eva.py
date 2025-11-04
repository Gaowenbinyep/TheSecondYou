import pandas as pd
import random
import numpy as np
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader



def contrastive_pair(positive_path, negative_path, save_path, pairs_num):
    positive_df = pd.read_json(positive_path, lines=True)
    negative_df = pd.read_json(negative_path, lines=True)
    pos_data = []
    neg_data = []
    for _, data in positive_df.iterrows():
        pos_data.append(data["conversations"][2]["content"])
    for _, data in negative_df.iterrows():
        neg_data.append(data["conversations"][2]["content"])
    
    contrastive_data = []
    for i in range(pairs_num//2):
        s1, s2 = random.sample(pos_data, 2)
        contrastive_data.append({"text1": s1, "text2": s2, "label": 1})
        s1 = random.choice(pos_data)
        s2 = random.choice(neg_data)
        contrastive_data.append({"text1": s1, "text2": s2, "label": 0})
    random.shuffle(contrastive_data)
    contrastive_df = pd.DataFrame(contrastive_data)
    contrastive_df.to_json(save_path, orient="records", lines=True, force_ascii=False)

def train_scorer():
    model = SentenceTransformer("./text2vec-base-chinese")
    df = pd.read_json("contrastive.json", lines=True)
    train_examples = [
        InputExample(texts=[row["text1"], row["text2"]], label=row["label"])
        for _, row in df.iterrows()  # 遍历DataFrame行
    ]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=100,
        output_path="./scorer_model",
    )

class Scorer:
    def __init__(self):
        self.model = SentenceTransformer("./scorer_model")
        # self.model = SentenceTransformer("./text2vec-base-chinese")
    def __call__(self, text1, text2):
        return self.similarity(text1, text2)
    def similarity(self, text1, text2):
        if len(text1) != len(text2):
            raise ValueError("text1 and text2 must have the same length")
        all_texts = text1 + text2
        embeddings = self.model.encode(all_texts, normalize_embeddings=True)
        n = len(text1)
        embeddings1, embeddings2 = embeddings[:n], embeddings[n:]
        dot_product = (embeddings1 * embeddings2).sum(axis=1)
        norm1 = np.linalg.norm(embeddings1, axis=1)
        norm2 = np.linalg.norm(embeddings2, axis=1)
        epsilon = 1e-8  # 避免除零错误
        cosine_similarity = dot_product / (norm1 * norm2 + epsilon)
        average_similarity = np.average(cosine_similarity)
        return average_similarity


def eval_scorer(result_path):
    scorer = Scorer()
    datas = pd.read_json(result_path, lines=True)
    text1 = datas["response"].tolist()
    text2 = datas["true_response"].tolist()
    scores = scorer(text1, text2)
    print(scores)
    
if __name__ == "__main__":
    # 生成对比样本
    # contrastive_pair(
    #     positive_path="../data/v1.0/Single_train.json",
    #     negative_path="../data/Gen_single_train.json",
    #     save_path="../data/eval/contrastive.json",
    #     pairs_num=4000)

    # # 训练打分模型
    # train_scorer()
    # eval_scorer("./test_result_V0_pe.json")
    # eval_scorer("./test_result_V0.json")
    # eval_scorer("./test_result_V1.json")
    # eval_scorer("./test_result_V2.json")
    # eval_scorer("./test_result_V3.json")
    # eval_scorer("./test_result_V4.json")
    # eval_scorer("./test_result_V5.json")
