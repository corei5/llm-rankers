import os
from beir import util
import tqdm
import numpy as np
import openai
from collections import deque
import math
from pydantic import BaseModel

# Assuming OpenAI API key is set in the environment
openai.api_key = 'your-openai-api-key'

# Function to get dataset path
def get_data_path(dataset_name: str) -> str:
    return f"/path/to/datasets/{dataset_name}"  # Adjust this path to your local dataset directory

# MCTS Node class
class Node(BaseModel):
    text: str
    parent: Node | None = None
    children: list[Node] = []
    visit: int = 0
    Q: float = 0.0
    reward_samples: list[int] = []

    def __hash__(self):
        return hash(self.text)

    def __eq__(self, other):
        return isinstance(other, Node) and self.text == other.text

    def add_child(self, child_node: Node):
        self.children.append(child_node)

    def add_reward(self, reward: int):
        self.reward_samples.append(reward)
        avg_reward = np.mean(self.reward_samples)
        min_reward = np.min(self.reward_samples)
        self.Q = (min_reward + avg_reward) / 2

    def __repr__(self):
        return f"Node(text='{self.text[:30]}...', Q={self.Q}, visit={self.visit})"


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
        reward = self._evaluate_text(node)
        if reward > self.reward_limit:
            reward = reward - self.excess_reward_penalty
        node.add_reward(reward=reward)
        node.visit += 1

    def self_refine(self, node: Node) -> Node:
        refined_text = self._refine_text(node)
        return Node(text=refined_text, parent=node)

    def _evaluate_text(self, node: Node) -> int:
        messages = [
            {
                "role": "system",
                "content": "Provide a relevance score between -100 and 100 for the text in response to the question. Return only the score.",
            },
            {
                "role": "user",
                "content": f"Question: {self.question}\n\nText: {node.text}",
            },
        ]
        response = openai.ChatCompletion.create(model="gpt-4", messages=messages, temperature=0.7, max_tokens=100)
        return int(response['choices'][0]['message']['content'])

    def _refine_text(self, node: Node) -> str:
        messages = [
            {
                "role": "system",
                "content": "Provide a refined version of the text to better match the question. Return only the refined text.",
            },
            {
                "role": "user",
                "content": f"Question: {self.question}\n\nCurrent Text: {node.text}",
            },
        ]
        response = openai.ChatCompletion.create(model="gpt-4", messages=messages, temperature=0.7, max_tokens=400)
        return response['choices'][0]['message']['content']

    def uct(self, node: Node):
        if not node.parent:
            return 10_000
        return node.Q + self.exploration_constant * math.sqrt(math.log(node.parent.visit + 1) / (node.visit + 1e-10))

    def run(self):
        # Initialize root with all texts as children
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


# Load BEIR Datasets
datasets = ["Covid", "NFCorpus", "Touche2020", "DBPedia", "SciFact", "Signal", "News", "Robust04"]
queries_and_texts = {}

# Download and Extract query-document pairs from each dataset
for dataset in datasets:
    dataset_path = util.download_dataset(dataset)  # This downloads the dataset from BEIR
    dataset_info = util.read_dataset(dataset, dataset_path)
    queries_and_texts[dataset] = [(query, doc) for query, doc in dataset_info["dev"]["query"].items()]
    print(f"Extracted {len(queries_and_texts[dataset])} query-document pairs from {dataset}.")

# Test MC-NEST on each dataset
for dataset_name, query_doc_pairs in queries_and_texts.items():
    print(f"\nEvaluating {dataset_name}:")
    for query, doc in query_doc_pairs:
        # You can adjust max_rollouts and exploration_constant as needed
        mcts = MC_NEST_Ranking(
            question=query,
            texts=[doc],  # Evaluate based on the query-document pair
            max_rollouts=10,
            exploration_constant=1.0,
        )
        rankings = mcts.run()  # Get rankings for the document based on the query
        for rank, (text, score) in enumerate(rankings, start=1):
            print(f"Rank {rank}: {text} (Score: {score})")
