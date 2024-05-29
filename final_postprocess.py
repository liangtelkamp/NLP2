import json
import re
from utils import *

def load_json(file_path):
    """Utility function to load a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, file_path):
    """Utility function to save a dictionary as a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def post_process(path):
    test_data = load_json(path)
    for key, value in test_data.items():
        string = value['few_shot_gemma'][0]
        if 'Example:[' in string:
            # Strip the Example:[] part and everything after ]
            stripped_string = re.sub(r'^Example:\[|\].*', '', string).replace("'", '').replace(' ', '')
            list_string = stripped_string.split(',')
            counter = {x: list_string.count(x) for x in list_string}
            value['counter_fs'] = counter
    save_json(test_data, path)

def strip_item(item):
    item = item.lower()
    replacements = [
        'possible continuations:', 'possible continuation:', 'possible continuations', 
        'possible continuation', '\n\npossible', 'Possible continu', '*', '\n', 
        '_', '.', ',', '-', 'Continuation:'
    ]
    for rep in replacements:
        item = item.replace(rep, '')
    return item

def process_list(items):
    """Process a list of items to extract the relevant first word."""
    new_list = []
    for item in items:
        stripped_item = strip_item(item)
        split = stripped_item.split()
        if split:
            new_list.append(split[1] if split[0] == "'s" else split[0])
    return new_list

def ensemble_dict(base_results, file_path, variant):
    new_results = load_json(file_path)
    for key in base_results.keys():
        base_results[key][variant] = new_results[key][variant]

# Load base and fs JSON files
BASE_PATH = 'test_base.json'
FS_PATH = 'test_fs.json'
test_baseline = load_json(BASE_PATH)
test_fs = load_json(FS_PATH)

# Post process BASE_PATH data
for key, value in test_baseline.items():
    counter = {x: value['base'].count(x) for x in value['base']}
    value['counter'] = counter
save_json(test_baseline, BASE_PATH)

# Ensemble different results
BASE_PATH = 'prompt_results/test_base.json'
FS_PATHS = {
    'fs_k_3': 'prompt_results/test_fs_k_3.json',
    'fs_k_5': 'prompt_results/test_fs_k_5.json',
    'fs_k_7': 'prompt_results/test_fs_k_7.json',
    'fs_k_5_DT': 'prompt_results/test_fs_k_5_DT.json',
    'fs_k_5_without_DT': 'prompt_results/test_fs_k_5_without_DT.json'
}
results = load_json(BASE_PATH)

for variant, path in FS_PATHS.items():
    ensemble_dict(results, path, variant)

# Process and clean the results
for key, value in results.items():
    for variant in FS_PATHS.keys():
        value[variant] = process_list(value[variant])
processed_results = results

# Save the complete results
save_json(processed_results, 'prompt_results/results_complete.json')
