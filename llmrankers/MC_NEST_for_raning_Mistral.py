from __future__ import annotations

import random
import math
from collections import deque
import tqdm
import numpy as np
from typing import ClassVar
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load LLaMA model and tokenizer
DEVICE = "cuda" #if torch.cuda.is_available() else "cpu"
LLAMA_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"  # Replace with your model path

llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME)
llama_model = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL_NAME).to(DEVICE)

# Helper function to interact with the LLaMA model
def generate_llama_response(prompt: str, max_length=100, temperature=0.7) -> str:
    input_ids = llama_tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    output_ids = llama_model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_p=0.9,
        do_sample=True,
    )
    return llama_tokenizer.decode(output_ids[0], skip_special_tokens=True)

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

# MCTS Implementation with LLaMA
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
        prompt = f"Evaluate the relevance of the following text to the question:\n\n" \
                 f"Question: {self.question}\nText: {node.text}\n\n" \
                 "Provide a score between -100 and 100."
        response = generate_llama_response(prompt, max_length=50)
        try:
            return int(response.strip())
        except ValueError:
            return -100  # Default to lowest score if parsing fails

    def _refine_text(self, node: Node) -> str:
        prompt = f"Refine the following text to better answer the question:\n\n" \
                 f"Question: {self.question}\nText: {node.text}\n\n" \
                 "Provide the refined text."
        return generate_llama_response(prompt, max_length=200)

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

# Example usage
if __name__ == "__main__":
    question = "What is the capital of Germany?"
    texts = [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
        "The capital of France is Lyon.",
        "France's capital is Paris.",
    ]
    mcts = MC_NEST_Ranking(
        question=question,
        texts=texts,
        max_rollouts=5,
        exploration_constant=1.0,
    )
    rankings = mcts.run()
    for rank, (text, score) in enumerate(rankings, start=1):
        print(f"Rank {rank}: {text} (Score: {score})")
