import pickle
from __future__ import annotations

import time
import random
import math
from collections import deque
from typing import ClassVar
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import rankdata

# Load LLaMA model and tokenizer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FLAN_T5_MODEL_NAME = "google/flan-t5-large"  # Replace with your model path

flan_t5_tokenizer = AutoTokenizer.from_pretrained(FLAN_T5_MODEL_NAME)
flan_t5_model = AutoModelForSeq2SeqLM.from_pretrained(FLAN_T5_MODEL_NAME).to(DEVICE)


# Helper function to load `.pkl` datasets
def load_pkl(filepath: str) -> list[dict]:
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


# Helper function to interact with the FLAN-T5 model
def generate_flan_response(prompt: str, max_length=100, temperature=0.7) -> (str, int, int):
    input_ids = flan_t5_tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    start_time = time.time()
    output_ids = flan_t5_model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_p=0.9,
        do_sample=True,
    )
    latency = time.time() - start_time
    pro_tokens = input_ids.numel()
    gen_tokens = output_ids.numel()
    response = flan_t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response, pro_tokens, gen_tokens, latency


# MCTS Node definition
class Node(BaseModel):
    text: str
    parent: Node | None = None
    children: list[Node] = []
    visit: int = 0
    Q: float = 0.0
    reward_samples: list[int] = []

    def __hash__(self):
        return hash(self.text)

    def add_child(self, child_node: Node):
        self.children.append(child_node)

    def add_reward(self, reward: int):
        self.reward_samples.append(reward)
        avg_reward = np.mean(self.reward_samples)
        min_reward = np.min(self.reward_samples)
        self.Q = (min_reward + avg_reward) / 2


# MCTS Implementation with FLAN-T5
class MC_NEST_Ranking(BaseModel):
    question: str
    texts: list[str]
    max_rollouts: int
    exploration_constant: float = 1.0
    reward_limit: int = 95
    excess_reward_penalty: int = 5

    root: Node = Node(text="Root Node")

    def __init__(self, **data):
        super().__init__(**data)
        self.root = Node(text="Root Node")

    def self_evaluate(self, node: Node):
        reward, pro_tokens, gen_tokens, latency = self._evaluate_text(node)
        if reward > self.reward_limit:
            reward -= self.excess_reward_penalty
        node.add_reward(reward=reward)
        node.visit += 1
        return pro_tokens, gen_tokens, latency

    def _evaluate_text(self, node: Node) -> (int, int, int, float):
        prompt = f"Evaluate the relevance of the following text to the question:\n\n" \
                 f"Question: {self.question}\nText: {node.text}\n\n" \
                 "Provide a score between -100 and 100."
        response, pro_tokens, gen_tokens, latency = generate_flan_response(prompt, max_length=50)
        try:
            return int(response.strip()), pro_tokens, gen_tokens, latency
        except ValueError:
            return -100, pro_tokens, gen_tokens, latency

    def run(self):
        self.root.children = [Node(text=text, parent=self.root) for text in self.texts]

        metrics = {"#Inferences": 0, "Pro tokens": 0, "Gen tokens": 0, "Latency(s)": 0}
        for _ in tqdm(range(self.max_rollouts)):
            node = self.select_node()
            pro_tokens, gen_tokens, latency = self.self_evaluate(node)
            metrics["#Inferences"] += 1
            metrics["Pro tokens"] += pro_tokens
            metrics["Gen tokens"] += gen_tokens
            metrics["Latency(s)"] += latency

        return self.get_ranking(), metrics

    def select_node(self) -> Node:
        candidates = []
        to_consider = deque([self.root])
        while to_consider:
            current_node = to_consider.popleft()
            if current_node.children:
                to_consider.extend(current_node.children)
            else:
                candidates.append(current_node)

        if not candidates:
            return self.root
        return max(candidates, key=self.uct)

    def uct(self, node: Node):
        if not node.parent:
            return 10_000
        return node.Q + self.exploration_constant * math.sqrt(math.log(node.parent.visit + 1) / (node.visit + 1e-10))

    def get_ranking(self) -> list[tuple[str, float]]:
        nodes = [child for child in self.root.children]
        ranked_nodes = sorted(nodes, key=lambda x: x.Q, reverse=True)
        return [(node.text, node.Q) for node in ranked_nodes]


# NDCG Metric Calculation
def ndcg_at_k(ranked_list, true_relevance, k=10):
    ranked_relevance = [true_relevance.get(doc, 0) for doc in ranked_list[:k]]
    dcg = sum((2 ** rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(ranked_relevance))
    ideal_relevance = sorted(true_relevance.values(), reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(ideal_relevance))
    return dcg / idcg if idcg > 0 else 0


if __name__ == "__main__":
    # Load `.pkl` datasets
    dataset_2019 = load_pkl("TRECDL2019_ANCE/test_ground_truth.pkl")
    dataset_2020 = load_pkl("TRECDL2020_ANCE/test_ground_truth.pkl")

    test_size = 100  # Limit to first 100 entries
    all_metrics = []
    for dataset in [dataset_2019, dataset_2020]:
        subset = dataset[:test_size]  # Select first `test_size` entries
        for entry in tqdm(subset):
            question = entry["query"]
            texts = entry["candidates"]
            true_relevance = {text: score for text, score in zip(entry["candidates"], entry["relevance"])}

            mcts = MC_NEST_Ranking(
                question=question,
                texts=texts,
                max_rollouts=10,
                exploration_constant=1.0,
            )
            rankings, metrics = mcts.run()

            ranked_texts = [text for text, _ in rankings]
            ndcg = ndcg_at_k(ranked_texts, true_relevance, k=10)

            metrics["NDCG@10"] = ndcg
            all_metrics.append(metrics)

    # Calculate and print overall metrics
    print("Overall Metrics:")
    for key in all_metrics[0]:
        avg_metric = np.mean([m[key] for m in all_metrics])
        print(f"{key}: {avg_metric:.4f}")
