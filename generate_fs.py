"""Improved Notebook for Few-shot Learning with Gemma"""

import random
import numpy as np
import torch
import json
import gc
import re
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
# from numba import cuda
import argparse

# --access_token TOKEN --total_samples_per_item 100 --max_batch_size 10 --max_length_increment 5 --base --model google/gemma-2b-it --k 5 --seed 42 --split 45 --DT '' --top_k 50 --top_p 0.95

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--total_samples_per_item', type=int, required=True, help='The total number of samples per item')
    parser.add_argument('--max_batch_size', type=int, required=True, help='The maximum batch size')
    parser.add_argument('--max_length_increment', type=int, default=5, required=True, help='The maximum length increment')
    parser.add_argument('--base', action='store_true', help='A boolean flag for base')
    parser.add_argument('--model', type=str, default='google/gemma-2b-it', help='The Gemma model variant')
    parser.add_argument('--k', type=int, default=5, required=True, help='Number of few-shot examples given to prompt')
    parser.add_argument('--seed', type=int, default=42, required=True, help='Random seed for reproducibility')
    parser.add_argument('--split', type=int, default=45, required=True, help='Split value for training/testing data')
    parser.add_argument('--DT', type=str, default='', required=False, help='DT flag')
    parser.add_argument('--top_k', type=int, default=50, required=False, help='Top K for sampling')
    parser.add_argument('--top_p', type=float, default=0.95, required=False, help='Top P for sampling')
    parser.add_argument('--access_token', type=str, required=True, help='Access token for the model')
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_to_json(data, file_path):
    # Check if the file path exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def manage_memory():
    torch.cuda.empty_cache()
    gc.collect()

def initialize_model_and_tokenizer(model_name, access_token):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, token=access_token).to('cuda' if torch.cuda.is_available() else 'cpu')
    return tokenizer, model

def generate_continuations(data_dict, tokenizer, model, device, args, prompt_key, suffix_key):
    for key, val in tqdm(data_dict.items()):
        input_text = val[prompt_key]
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        continuations = []

        while len(continuations) < args.total_samples_per_item:
            manage_memory()
            batch_size = min(args.max_batch_size, args.total_samples_per_item - len(continuations))
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + args.max_length_increment,
                    num_return_sequences=batch_size,
                    do_sample=True,
                    top_k=args.top_k,
                    top_p=args.top_p
                )
            continuations.extend(
                tokenizer.decode(output[input_ids.shape[1]:], skip_special_tokens=True)
                for output in outputs
            )
        data_dict[key][suffix_key] = continuations

def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = load_json('prompt_dicts/provo_dict.json')
    train_dict = {key: value for key, value in data.items() if value['text_id'] <= args.split}
    test_dict = {key: value for key, value in data.items() if value['text_id'] > args.split}

    prompt_file = f'prompt_dicts/prompt_dict_k_{args.k}'
    if args.DT:
        prompt_file += f'_{args.DT}'
    prompt_file += '.json'
    test_few_shot = load_json(prompt_file)

    tokenizer, model = initialize_model_and_tokenizer(args.model, args.access_token)

    generate_continuations(test_few_shot, tokenizer, model, device, args, 'prompt', f'fs_k_{args.k}' + (f'_{args.DT}' if args.DT else ''))
    result_file = f'prompt_results/test_fs_k_{args.k}' + (f'_{args.DT}' if args.DT else '') + '.json'
    save_to_json(test_few_shot, result_file)

    if args.base:
        try:
            generate_continuations(test_dict, tokenizer, model, device, args, 'context', 'baseline')
            save_to_json(test_dict, 'prompt_results/test_base.json')
        except Exception as e:
            print(f"An error occurred: {e}")

    print('Process completed')

if __name__ == "__main__":
    args = parse_arguments()
    main(args)