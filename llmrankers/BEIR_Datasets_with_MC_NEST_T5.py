import os
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import tqdm
import numpy as np
from collections import deque
import math
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time

# Load LLaMA model and tokenizer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FLAN_T5_MODEL_NAME = "google/flan-t5-large"  # Replace with your model path

flan_t5_tokenizer = AutoTokenizer.from_pretrained(FLAN_T5_MODEL_NAME)
flan_t5_model = AutoModelForSeq2SeqLM.from_pretrained(FLAN_T5_MODEL_NAME).to(DEVICE)

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

# MCTS Node class
class Node(BaseModel):
    text: str
    parent: Node | None = None
    children: list[Node] = []
    visit: int = 0
    Q: float = 0.0
    reward_samples: list[int] = []

    def add_child(self, child_node: Node):
        self.children.append(child_node)

    def add_reward(self, reward: int):
        self.reward_samples.append(reward)
        avg_reward = np.mean(self.reward_samples)
        min_reward = np.min(self.reward_samples)
        self.Q = (min_reward + avg_reward) / 2

# MCTS with ranking (MC-NEST)
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
        prompt = f"Evaluate the relevance of the following text to the question:\n\n" \
                 f"Question: {self.question}\nText: {node.text}\n\n" \
                 "Provide a score between -100 and 100."
        response, _, _, _ = generate_flan_response(prompt, max_length=50)
        try:
            reward = int(response.strip())
        except ValueError:
            reward = -100
        if reward > self.reward_limit:
            reward -= self.excess_reward_penalty
        node.add_reward(reward=reward)
        node.visit += 1

    def self_refine(self, node: Node) -> Node:
    prompt = f"Rethink the relevance score for the following text in relation to the question:\n\n" \
             f"Question: {self.question}\nText: {node.text}\n\n" \
             "Provide a relevance score between -100 and 100, considering how well the text aligns with the question. Do not modify the text."
    new_score, _, _, _ = generate_flan_response(prompt, max_length=50)
    try:
        refined_score = int(new_score.strip())
    except ValueError:
        refined_score = -100  # Default to the minimum score if parsing fails

    node.add_reward(reward=refined_score)
    node.visit += 1
    return node


    def uct(self, node: Node):
        if not node.parent:
            return 10_000
        return node.Q + self.exploration_constant * math.sqrt(math.log(node.parent.visit + 1) / (node.visit + 1e-10))

    def run(self):
        self.root.children = [Node(text=text, parent=self.root) for text in self.texts]

        for _ in tqdm.tqdm(range(self.max_rollouts)):
            node = self.select_node()
            self.self_evaluate(node)
            if not node.children:  # Create a refined child
                refined_node = self.self_refine(node)
                node.add_child(refined_node)
                self.self_evaluate(refined_node)
            self.backpropagate(node)

        return self.get_ranking()

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

    def backpropagate(self, node: Node):
        parent = node.parent
        while parent:
            best_child_Q = max(child.Q for child in parent.children)
            parent.Q = (parent.Q + best_child_Q) / 2
            parent.visit += 1
            parent = parent.parent

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
    # BEIR datasets to evaluate
    beir_datasets = ["trec-covid", "nfcorpus", "arguana", "dbpedia-entity", "scifact", "signal1m", "trec-news", "robust04"]
    k = 10
    max_rollouts = 10
    all_metrics = []

    for dataset_name in beir_datasets:
        print(f"Processing dataset: {dataset_name}")

        # Load dataset
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        dataset_path = util.download_and_unzip(url, "datasets")
        data = GenericDataLoader(dataset_path).load(split="test")
        queries, corpus, qrels = data

        # Evaluate each query
        for query_id, query_text in tqdm.tqdm(queries.items(), desc=f"Dataset: {dataset_name}"):
            texts = [corpus[doc_id]["text"] for doc_id in corpus.keys()]
            true_relevance = qrels.get(query_id, {})

            mcts = MC_NEST_Ranking(
                question=query_text,
                texts=texts,
                max_rollouts=max_rollouts,
                exploration_constant=1.0,
            )
            rankings = mcts.run()
            ranked_texts = [text for text, _ in rankings]
            ndcg = ndcg_at_k(ranked_texts, true_relevance, k=k)

            all_metrics.append({"Dataset": dataset_name, "NDCG@10": ndcg})

    # Calculate and print overall metrics
    print("Overall Metrics:")
    ndcg_scores = [m["NDCG@10"] for m in all_metrics]
    print(f"Average NDCG@10: {np.mean(ndcg_scores):.4f}")
