import json
import random
import argparse
import torch
from typing import Tuple, Dict, Any, List
from utils import set_seed, get_pos_tags

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate prompts for few-shot learning.")
    parser.add_argument('--k', type=int, default=5, help='Number of examples to use for generating prompts, by default 5.')
    parser.add_argument('--DT', type=str, default='', help='Flag to filter contexts based on DT POS tag.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--split', type=int, default=45, help='Split value for training/testing data.')
    return parser.parse_args()

def select_device() -> torch.device:
    """Select the best available device: CUDA, MPS, or CPU."""
    if torch.cuda.is_available():
        print("Using CUDA as device")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using MPS as device")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")

def filter_train_data(data: Dict[str, Any], split: int, dt_flag: bool) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split and filter the data into training and testing sets."""
    train_dict = {}
    test_dict = {}
    
    for key, value in data.items():
        last_context_word = value['context'][-1]
        last_pos_tag = get_pos_tags(last_context_word)[0][1]
        
        if value['text_id'] < split:
            if dt_flag == 'with' and last_pos_tag != 'DT':
                continue
            elif dt_flag == 'without' and last_pos_tag == 'DT':
                continue
            train_dict[key] = value
        else:
            test_dict[key] = value
    
    return train_dict, test_dict

def sample_fewshot(train: Dict[str, Any], k: int) -> List:
    """Select k random items from the training data."""
    return random.sample(list(train.items()), k)

def format_examples(examples: List) -> Tuple[str, List]:
    """Format the few-shot examples into a prompt string."""
    prompt = ''
    indices = []

    for idx, item in examples:
        context = " ".join(item["context"])
        prompt += f'\nContext: {context}\nPossible continuations:\n'
        continuations = [cont for cont, count in item['continuations'] for _ in range(count)]
        for i, continuation in enumerate(continuations, start=1):
            prompt += f'{i}. {continuation}\n'
        indices.append(idx)
    
    return prompt, indices

def generate_prompt(train: Dict[str, Any], test: Dict[str, Any], k: int, idx: str) -> Tuple[str, Dict[str, List]]:
    """Generate a complete prompt for the model based on training and test data."""
    random_items = sample_fewshot(train, k)
    train_example, train_index = format_examples(random_items)
    test_context = " ".join(test[idx]["context"])

    prompt = (f"\nIn this task, we aim to enhance the performance of LLMs by leveraging few-shot prompting to capture "
              f"human variability across different contexts. I want you to generate a single-word possible continuation "
              f"for a context. Here are {k} examples of texts and their possible continuations made by humans, displaying "
              f"linguistic variability.\n{train_example}\nShow the same linguistic variability as humans. Generate a possible "
              f"continuation for: '{test_context}'")

    return prompt, {'train': train_index, 'test': idx}

def main():
    args = parse_arguments()
    device = select_device()
    
    set_seed(args.seed)
    print(f"Number of examples (k): {args.k}")
    print(f"Data split value: {args.split}")

    with open('provo_dict.json', 'r') as f:
        df_dict = json.load(f)

    train_dict, test_dict = filter_train_data(df_dict, args.split, args.DT)

    prompt_dict = {}
    for key in test_dict:
        prompt, idx = generate_prompt(train_dict, test_dict, args.k, key)
        prompt_dict[key] = {'prompt': prompt, 'Example_IDs': idx['train']}

    if args.DT == 'with':
        output_file = f'prompt_dicts/prompt_dict_k_{args.k}_DT.json'
    elif args.DT == 'without':
        output_file = f'prompt_dicts/prompt_dict_k_{args.k}_without_DT.json'
    else:
        output_file = f'prompt_dicts/prompt_dict_k_{args.k}.json'
    with open(output_file, 'w') as f:
        json.dump(prompt_dict, f, indent=4)

if __name__ == "__main__":
    main()
