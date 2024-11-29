from typing import List, Tuple
from .rankers import LlmRanker, SearchResult
import openai
import time
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import copy
from collections import Counter
import tiktoken
import random
random.seed(929)


class SetwiseLlmRanker(LlmRanker):
    CHARACTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W"]

    def __init__(self, model_name_or_path, tokenizer_name_or_path, device, num_child=3, k=10, scoring='generation', method="heapsort", num_permutation=1, cache_dir=None):
        self.device = device
        self.num_child = num_child
        self.num_permutation = num_permutation
        self.k = k
        self.config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.initialize_model_and_tokenizer(model_name_or_path, tokenizer_name_or_path, cache_dir)

        self.scoring = scoring
        self.method = method
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0

    def initialize_model_and_tokenizer(self, model_name_or_path, tokenizer_name_or_path, cache_dir):
        if self.config.model_type == 't5':
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name_or_path or model_name_or_path, cache_dir=cache_dir)
            self.llm = T5ForConditionalGeneration.from_pretrained(model_name_or_path,
                                                                  device_map='auto',
                                                                  torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                                                                  cache_dir=cache_dir)
            self.decoder_input_ids = self.tokenizer.encode("<pad> Passage", return_tensors="pt", add_special_tokens=False).to(self.device)
            self.target_token_ids = self.tokenizer.batch_encode_plus([f'<pad> Passage {self.CHARACTERS[i]}' for i in range(len(self.CHARACTERS))],
                                                                     return_tensors="pt", add_special_tokens=False, padding=True).input_ids[:, -1]
        elif self.config.model_type == 'llama':
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
            self.tokenizer.use_default_system_prompt = False
            self.llm = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                            device_map='auto',
                                                            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                                                            cache_dir=cache_dir).eval()
        else:
            raise NotImplementedError(f"Model type {self.config.model_type} is not supported yet for setwise.")

    def compare(self, query: str, docs: List):
        self.total_compare += 1 if self.num_permutation == 1 else self.num_permutation
        passages = "\n\n".join([f'Passage {self.CHARACTERS[i]}: "{doc.text}"' for i, doc in enumerate(docs)])
        input_text = f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n' + passages + '\n\nOutput only the passage label of the most relevant passage:'

        if self.scoring == 'generation':
            return self.generate_scoring(query, docs, input_text)
        elif self.scoring == 'likelihood':
            return self.likelihood_scoring(query, docs, input_text)
        else:
            raise NotImplementedError(f"Scoring method {self.scoring} not implemented.")

    def generate_scoring(self, query, docs, input_text):
        if self.config.model_type == 't5':
            return self.generate_t5_scoring(query, docs, input_text)
        elif self.config.model_type == 'llama':
            return self.generate_llama_scoring(query, input_text)
        else:
            raise NotImplementedError

    def generate_t5_scoring(self, query, docs, input_text):
        if self.num_permutation == 1:
            return self.t5_single_permutation(input_text)
        else:
            return self.t5_multiple_permutations(query, docs, input_text)

    def t5_single_permutation(self, input_text):
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        self.total_prompt_tokens += input_ids.shape[1]
        output_ids = self.llm.generate(input_ids, decoder_input_ids=self.decoder_input_ids, max_new_tokens=2)[0]
        self.total_completion_tokens += output_ids.shape[0]
        output = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return output[-1]

    def t5_multiple_permutations(self, query, docs, input_text):
        id_passage = [(i, p) for i, p in enumerate(docs)]
        labels = [self.CHARACTERS[i] for i in range(len(docs))]
        batch_data = [(random.sample(id_passage, len(id_passage)), random.sample(labels, len(labels))) for _ in range(self.num_permutation)]
        input_ids = self.tokenizer([self.prepare_batch_input_text(batch, query, docs) for batch in batch_data], return_tensors="pt").input_ids.to(self.device)
        self.total_prompt_tokens += input_ids.shape[1] * input_ids.shape[0]
        output_ids = self.llm.generate(input_ids, decoder_input_ids=self.decoder_input_ids.repeat(input_ids.shape[0], 1), max_new_tokens=2)
        return self.t5_vote_for_winner(batch_data, output_ids)

    def prepare_batch_input_text(self, batch, query, docs):
        ref, characters = batch
        passages = "\n\n".join([f'Passage {characters[i]}: "{docs[p[0]].text}"' for i in range(len(ref))])
        return f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n' + passages + '\n\nOutput only the passage label of the most relevant passage:'

    def t5_vote_for_winner(self, batch_data, output_ids):
        output = self.tokenizer.batch_decode(output_ids[:, self.decoder_input_ids.shape[1]:], skip_special_tokens=True)
        candidates = [self.vote_for_passage(ref, result) for ref, result in zip(batch_data, output)]
        return self.resolve_tie(candidates)

    def vote_for_passage(self, ref, result):
        result = result.strip().upper()
        docids, characters = ref
        if len(result) != 1 or result not in characters:
            print(f"Unexpected output: {result}")
            return None
        return docids[characters.index(result)]

    def resolve_tie(self, candidates):
        candidate_counts = Counter(candidates)
        max_count = max(candidate_counts.values())
        most_common_candidates = [candidate for candidate, count in candidate_counts.items() if count == max_count]
        return self.CHARACTERS[random.choice(most_common_candidates)] if len(most_common_candidates) > 1 else self.CHARACTERS[most_common_candidates[0]]

    def generate_llama_scoring(self, query, input_text):
        conversation = [{"role": "user", "content": input_text}]
        prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True) + " Passage:"
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        self.total_prompt_tokens += input_ids.shape[1]
        output_ids = self.llm.generate(input_ids, do_sample=False, temperature=0.0, top_p=None, max_new_tokens=1)[0]
        self.total_completion_tokens += output_ids.shape[0]
        return self.tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip().upper()

    def likelihood_scoring(self, query, docs, input_text):
        if self.config.model_type == 't5':
            return self.t5_likelihood_scoring(input_text, docs)
        else:
            raise NotImplementedError

    def t5_likelihood_scoring(self, input_text, docs):
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        self.total_prompt_tokens += input_ids.shape[1]
        with torch.no_grad():
            logits = self.llm(input_ids=input_ids, decoder_input_ids=self.decoder_input_ids).logits[0][-1]
            distributions = torch.softmax(logits, dim=0)
            scores = distributions[self.target_token_ids[:len(docs)]]
            ranked = sorted(zip(self.CHARACTERS[:len(docs)], scores), key=lambda x: x[1], reverse=True)
            return ranked[0][0]

    def rerank(self, query: str, ranking: List[SearchResult]) -> Tuple[str, List[SearchResult]]:
        # Step 1: Compute scores for each document in the ranking
        scores = []
        for result in ranking:
            score = self.compare(query, [result])  # Compare query with each document
            scores.append((result, score))  # Store the result along with its score
        
        # Step 2: Sort the results based on their scores (you may need to define what `score` is)
        ranked_results = sorted(scores, key=lambda x: x[1], reverse=True)  # Sort in descending order of relevance
        
        # Step 3: Return the most relevant result and the list of re-ranked results
        top_result = ranked_results[0][0]  # The most relevant document
        ranked_docs = [result[0] for result in ranked_results]  # List of documents sorted by score
        
        return top_result, ranked_docs
