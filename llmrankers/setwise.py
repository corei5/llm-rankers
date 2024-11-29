from typing import List
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
    CHARACTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
                  "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W"]

    def __init__(self,
                 model_name_or_path,
                 tokenizer_name_or_path,
                 device,
                 num_child=3,
                 k=10,
                 scoring='generation',
                 method="heapsort",
                 num_permutation=1,
                 cache_dir=None):
        self.device = device
        self.num_child = num_child
        self.num_permutation = num_permutation
        self.k = k
        self.config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        if self.config.model_type == 't5':
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name_or_path
                                                         if tokenizer_name_or_path is not None else
                                                         model_name_or_path,
                                                         cache_dir=cache_dir)
            self.llm = T5ForConditionalGeneration.from_pretrained(model_name_or_path,
                                                                  device_map='auto',
                                                                  torch_dtype=torch.float16 if device == 'cuda'
                                                                  else torch.float32,
                                                                  cache_dir=cache_dir)
            self.decoder_input_ids = self.tokenizer.encode("<pad> Passage",
                                                           return_tensors="pt",
                                                           add_special_tokens=False).to(self.device) if self.tokenizer else None

            self.target_token_ids = self.tokenizer.batch_encode_plus([f'<pad> Passage {self.CHARACTERS[i]}'
                                                                      for i in range(len(self.CHARACTERS))],
                                                                     return_tensors="pt",
                                                                     add_special_tokens=False,
                                                                     padding=True).input_ids[:, -1]
        elif self.config.model_type == 'llama':
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
            self.tokenizer.use_default_system_prompt = False
            self.llm = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                            device_map='auto',
                                                            torch_dtype=torch.float16 if device == 'cuda'
                                                            else torch.float32,
                                                            cache_dir=cache_dir).eval()
        else:
            raise NotImplementedError(f"Model type {self.config.model_type} is not supported yet for setwise:(")

        self.scoring = scoring
        self.method = method
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        self.feedback_history = []  # Store past feedbacks

    def compare(self, query: str, docs: List):
        self.total_compare += 1 if self.num_permutation == 1 else self.num_permutation

        passages = "\n\n".join([f'Passage {self.CHARACTERS[i]}: "{doc.text}"' for i, doc in enumerate(docs)])
        input_text = f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n' \
                     + passages + '\n\nOutput only the passage label of the most relevant passage:'

        if self.scoring == 'generation':
            if self.config.model_type == 't5':
                if self.num_permutation == 1:
                    input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
                    self.total_prompt_tokens += input_ids.shape[1]

                    output_ids = self.llm.generate(input_ids,
                                                   decoder_input_ids=self.decoder_input_ids,
                                                   max_new_tokens=2)[0]

                    self.total_completion_tokens += output_ids.shape[0]

                    output = self.tokenizer.decode(output_ids,
                                                   skip_special_tokens=True).strip()
                    output = output[-1]
                else:
                    id_passage = [(i, p) for i, p in enumerate(docs)]
                    labels = [self.CHARACTERS[i] for i in range(len(docs))]
                    batch_data = []
                    for _ in range(self.num_permutation):
                        batch_data.append([random.sample(id_passage, len(id_passage)),
                                           random.sample(labels, len(labels))])

                    batch_ref = []
                    input_text = []
                    for batch in batch_data:
                        ref = []
                        passages = []
                        characters = []
                        for p, c in zip(batch[0], batch[1]):
                            ref.append(p[0])
                            passages.append(p[1].text)
                            characters.append(c)
                        batch_ref.append((ref, characters))
                        passages = "\n\n".join([f'Passage {characters[i]}: "{passages[i]}"' for i in range(len(passages))])
                        input_text.append(f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n' \
                                          + passages + '\n\nOutput only the passage label of the most relevant passage:')

                    input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
                    self.total_prompt_tokens += input_ids.shape[1] * input_ids.shape[0]

                    output_ids = self.llm.generate(input_ids,
                                                   decoder_input_ids=self.decoder_input_ids.repeat(input_ids.shape[0], 1),
                                                   max_new_tokens=2)
                    output = self.tokenizer.batch_decode(output_ids[:, self.decoder_input_ids.shape[1]:],
                                                         skip_special_tokens=True)

                    candidates = []
                    for ref, result in zip(batch_ref, output):
                        result = result.strip().upper()
                        docids, characters = ref
                        if len(result) != 1 or result not in characters:
                            print(f"Unexpected output: {result}")
                            continue
                        win_doc = docids[characters.index(result)]
                        candidates.append(win_doc)

                    if len(candidates) == 0:
                        print(f"Unexpected voting: {output}")
                        output = "Unexpected voting."
                    else:
                        candidate_counts = Counter(candidates)
                        max_count = max(candidate_counts.values())
                        most_common_candidates = [candidate for candidate, count in candidate_counts.items() if
                                                  count == max_count]
                        if len(most_common_candidates) == 1:
                            output = self.CHARACTERS[most_common_candidates[0]]
                        else:
                            output = self.CHARACTERS[random.choice(most_common_candidates)]

            elif self.config.model_type == 'llama':
                conversation = [{"role": "user", "content": input_text}]
                prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                prompt += " Passage:"

                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
                self.total_prompt_tokens += input_ids.shape[1]

                output_ids = self.llm.generate(input_ids,
                                               do_sample=False,
                                               temperature=0.0,
                                               top_p=None,
                                               max_new_tokens=1)[0]

                self.total_completion_tokens += output_ids.shape[0]

                output = self.tokenizer.decode(output_ids[input_ids.shape[1]:],
                                               skip_special_tokens=True).strip().upper()

        elif self.scoring == 'likelihood':
            if self.config.model_type == 't5':
                input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
                self.total_prompt_tokens += input_ids.shape[1]
                with torch.no_grad():
                    logits = self.llm(input_ids=input_ids, decoder_input_ids=self.decoder_input_ids).logits[0][-1]
                    distributions = torch.softmax(logits, dim=0)
                    scores = distributions[self.target_token_ids[:len(docs)]]
                    ranked = sorted(zip(self.CHARACTERS[:len(docs)], scores), key=lambda x: x[1], reverse=True)
                    output = ranked[0][0]
            elif self.config.model_type == 'llama':
                input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
                self.total_prompt_tokens += input_ids.shape[1]
                with torch.no_grad():
                    logits = self.llm(input_ids=input_ids).logits[0]
                    logits = logits[-1]
                    distributions = torch.softmax(logits, dim=0)
                    scores = distributions
                    ranked = sorted(zip(self.CHARACTERS[:len(docs)], scores), key=lambda x: x[1], reverse=True)
                    output = ranked[0][0]
        else:
            raise NotImplementedError(f"Scoring method {self.scoring} is not implemented yet.")

        self._evaluate_and_reflect(query, output, docs)

        return output

    def _evaluate_and_reflect(self, query: str, output: str, docs: List):
        """
        Evaluate the quality of the current decision and reflect on it.
        """
        expected_answer = self._get_expected_answer(query)
        if output != expected_answer:
            self.feedback_history.append({
                'query': query,
                'selected_passage': output,
                'expected_answer': expected_answer,
                'evaluation': 'incorrect'
            })
            self._adjust_ranking_behavior()

    def _get_expected_answer(self, query: str):
        # Implement this to return expected answers (from a ground truth or heuristic)
        return "ExpectedAnswer"  # Placeholder

    def _adjust_ranking_behavior(self):
        """
        Adjust ranking behavior based on accumulated feedback.
        """
        # Track mistakes
        mistakes = [feedback for feedback in self.feedback_history if feedback['evaluation'] == 'incorrect']
        mistake_ratio = len(mistakes) / len(self.feedback_history) if self.feedback_history else 0

        # Modify scoring strategy based on performance
        if mistake_ratio > 0.5:
            self.scoring = 'likelihood'  # If the model has a high mistake ratio, switch to likelihood-based scoring.
        else:
            self.scoring = 'generation'  # Otherwise, stick with generation-based scoring.

        # Adjust model parameters if needed (e.g., temperature, top-p)
        if mistake_ratio > 0.7:
            print("Adjusting model behavior: Lowering randomness.")
            self.llm.config.temperature = 0.2  # Reduce randomness
        elif mistake_ratio < 0.3:
            print("Adjusting model behavior: Increasing randomness.")
            self.llm.config.temperature = 0.8  # Increase randomness for exploration

        """
        pass
