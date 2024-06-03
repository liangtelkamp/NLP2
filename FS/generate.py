# import pandas as pd
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random

# Check if mps is available on torch
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using cuda as device")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
#     print(f"Using mps as device")
else:
    device = torch.device("cpu")
    print("Using CPU")

# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
# model = AutoModelForCausalLM.from_pretrained(
#     "google/gemma-2b-it",
# ).to(device)


def sample_fewshot(train, k):
    # Pick random items from dictionary
    random_items = random.sample(list(train.items()), k)
    return random_items


def get_examples(train, k):
    random_items = sample_fewshot(train, k)
    prompt = ''
    c = 1
    indices = []
    for item in random_items:
        prompt += f'\nContext:\n{" ".join(item[1]["context"])}\n'
        prompt += f'Instruction: \nGenerate 40 possible continuations \nExample:'
        con = []
        for continuation in item[1]['continuations']:
            for i in range(continuation[1]):
                con.append(continuation[0])
        prompt += f'{con}\n'
        indices.append(item[0])
        c+=1
    return prompt, indices

def get_examples2(train, k):
    random_items = sample_fewshot(train, k)
    prompt = ''

    indices = []
    for item in random_items:
        c = 1
        prompt += f'\nContext: {" ".join(item[1]["context"])}\n'
        prompt += f'Possible continuations: \n'
        con = []
        for continuation in item[1]['continuations']:
            for i in range(continuation[1]):
                con.append(continuation[0])
                prompt += f'{c}. {continuation[0]}\n'
                c+=1
        indices.append(item[0])
    return prompt, indices


def test_example_to_prompt(test, idx):
    prompt = f'{" ".join(test[idx]["context"])}'
    return prompt, idx


def generate_prompt(train_file, test, k, idx):
    train_example, train_index = get_examples2(train_file, k)

    test_context, test_idx = test_example_to_prompt(test, idx)
    prompt = f"""
In this task, we aim to enhance the performance of LLMs by leveraging few-shot prompting to capture human variability across different contexts. I want you to generate a singly-word possible continuation for a context. Here are {k} examples of texts and their possible continuations made by humans, displaying linguistic variability.
{train_example}
Show the same linguistic variability as humans. Generate a possible continuation for: '{test_context}
    """
    return prompt, {'train': train_index, 'test': test_idx}


# def get_few_shot_samples(context, n_samples = 5, add_tokens = 5, model=model, tokenizer=tokenizer):
#     """Given a context we return the words that were generated during ancestral sampling for n_samples"""
#     inputs_ids = tokenizer(context, return_tensors="pt")['input_ids'].to(device)
#     outputs = model.generate(inputs_ids, do_sample=True, num_beams = 1, num_return_sequences= n_samples,
#                              max_new_tokens = add_tokens) #This samples unbiasedly from the next-token distribution of the model for add_tokens tokens, n_samples times
#     outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#     # print(outputs)
#     #Remove context to keep only generated text
#     outputs = [x.replace(context + ' ', '').replace(context, '').replace('\n', '') for x in outputs]
#     # Strip undrscores from outputs
#     outputs = [re.sub(r'_', '', x) for x in outputs]
#     outputs = [re.sub(r'<.*?>', '', x) for x in outputs]
#     print(outputs)
#     #Remove punctuation
#     outputs = [re.sub(r'[^\w\s]', '', x) for x in outputs] #removing punctuation
#     list_of_words = [x.split(' ') for x in outputs]

#     sampled_words = []
#     for generation in list_of_words:
#         if set(generation) == {''}:
#             sampled_words.append('Failed to generate word')
#         else:
#             sampled_words.append(next(x.lower() for x in generation if x))
#     print(sampled_words)
#     return sampled_words


# def generate_few_shot_results(train_file, test, k, idx, model, tokenizer, n_samples = 5, add_tokens = 5):
#     """Given a few-shot prompt we generate the few-shot samples and return the most common word"""
#     prompt, indices = generate_prompt(train_file, test, k, idx)
#     print(prompt)
#     few_shot_samples = get_few_shot_samples(prompt, n_samples = n_samples, add_tokens = add_tokens, model=model, tokenizer=tokenizer)
#     return few_shot_samples, indices

