import tiktoken
from .rankers import LlmRanker, SearchResult
from typing import List
import copy
import openai
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM, AutoConfig


def max_tokens(model):
    if 'gpt-4' in model:
        return 8192
    else:
        return 4096


def get_post_prompt(query, num):
    return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."


def get_prefix_prompt(query, num):
    return [{'role': 'system',
             'content': "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."},
            {'role': 'user',
             'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    if model == "gpt-3.5-turbo":
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        tokens_per_message, tokens_per_name = 0, 0

    try:
        encoding = tiktoken.get_encoding(model)
    except:
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    if isinstance(messages, list):
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
    else:
        num_tokens += len(encoding.encode(messages))
    num_tokens += 3  # every reply is primed with <|start|>
    return num_tokens


def create_permutation_instruction_chat(query: str, docs: List[SearchResult], model_name='gpt-3.5-turbo'):
    num = len(docs)

    max_length = 300
    while True:
        messages = get_prefix_prompt(query, num)
        rank = 0
        for doc in docs:
            rank += 1
            content = doc.text
            content = content.replace('Title: Content: ', '')
            content = content.strip()
            # For Japanese should cut by character: content = content[:int(max_length)]
            content = ' '.join(content.split()[:int(max_length)])
            messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
            messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
        messages.append({'role': 'user', 'content': get_post_prompt(query, num)})

        if model_name is not None:
            if num_tokens_from_messages(messages, model_name) <= max_tokens(model_name) - 200:
                break
            else:
                max_length -= 1
        else:
            break
    return messages


def create_permutation_instruction_complete(query: str, docs: List[SearchResult]):
    num = len(docs)
    message = f"This is RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query.\n\n" \
              f"The following are {num} passages, each indicated by number identifier []. " \
              f"I can rank them based on their relevance to query: {query}\n\n"

    rank = 0
    for doc in docs:
        rank += 1
        content = doc.text
        content = content.replace('Title: Content: ', '')
        content = content.strip()
        content = ' '.join(content.split()[:300])
        message += f"[{rank}] {content}\n\n"
    message += f"The search query is: {query}"
    message += f"I will rank the {num} passages above based on their relevance to the search query. The passages " \
               "will be listed in descending order using identifiers, and the most relevant passages should be listed "\
               "first, and the output format should be [] > [] > etc, e.g., [1] > [2] > etc.\n\n" \
               f"The ranking results of the {num} passages (only identifiers) is:"
    return message


def clean_response(response: str):
    new_response = ''
    for c in response:
        if not c.isdigit():
            new_response += ' '
        else:
            new_response += c
    new_response = new_response.strip()
    return new_response


def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response


def receive_permutation(ranking, permutation, rank_start=0, rank_end=100):
    response = clean_response(permutation)
    response = [int(x) - 1 for x in response.split()]
    response = remove_duplicate(response)
    cut_range = copy.deepcopy(ranking[rank_start: rank_end])
    original_rank = [tt for tt in range(len(cut_range))]
    response = [ss for ss in response if ss in original_rank]
    response = response + [tt for tt in original_rank if tt not in response]

    for j, x in enumerate(response):
        ranking[j + rank_start] = cut_range[x]

    return ranking


def zero_shot_cot_reflection(query: str, docs: List[SearchResult]) -> str:
    """
    This function generates a reasoning and reflection step using Zero-Shot Chain-of-Thought (CoT) approach.
    It analyzes the provided query and documents to produce a rational evaluation and ranking of the documents.
    """
    # Construct a prompt for Zero-Shot CoT reasoning
    prompt = f"Given the search query: '{query}', evaluate the following passages based on their relevance to the query. " \
             "Provide a reasoning process before giving your final ranking of the passages. " \
             "Rank the passages in the order of relevance to the query, starting with the most relevant. " \
             "Use the format: [1] > [2] > [3] > ..."

    for i, doc in enumerate(docs):
        prompt += f"\n[{i+1}] {doc.text}"

    messages = [{'role': 'user', 'content': prompt}]
    
    # Call OpenAI API for the reasoning and reflection
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4",  # Or another relevant model
            messages=messages,
            temperature=0.7,
            request_timeout=15
        )
        return completion['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error during CoT reflection: {str(e)}")
        return ""


class OpenAiListwiseLlmRanker(LlmRanker):
    def __init__(self, model_name_or_path, api_key, window_size, step_size, num_repeat):
        self.llm = model_name_or_path
        self.tokenizer = tiktoken.encoding_for_model(model_name_or_path)
        self.window_size = window_size
        self.step_size = step_size
        self.num_repeat = num_repeat
        openai.api_key = api_key
        self.total_compare = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def compare(self, query: str, docs: List):
        self.total_compare += 1
        messages = create_permutation_instruction_chat(query, docs, self.llm)
        while True:
            try:
                completion = openai.ChatCompletion.create(
                    model=self.llm,
                    messages=messages,
                    temperature=0.0,
                    request_timeout=15)
                self.total_completion_tokens += int(completion['usage']['completion_tokens'])
                self.total_prompt_tokens += int(completion['usage']['prompt_tokens'])
                return completion['choices'][0]['message']['content']
            except Exception as e:
                print(str(e))
                if "This model's maximum context length is" in str(e):
                    print("Message size too large. Reducing message length...")
                    self.step_size = max(self.step_size - 1, 1)
                    messages = create_permutation_instruction_chat(query, docs, self.llm)
                else:
                    raise

