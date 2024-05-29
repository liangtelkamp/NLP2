import sys
import json
import numpy as np
from utils import get_provo_data, get_tvd_value_human_human, get_tvd_value_human_model, get_ith_element

# Ensure the parent directory is in the Python path
sys.path.append('..')

def flatten_list(nested_list):
    """Flatten a nested list into a single list."""
    return [item for sublist in nested_list for item in sublist]


def prepare_human_data(provo_data):
    """Prepare human samples and original corpus words for evaluation.

    Args:
        provo_data (dict): The dictionary containing Provo Corpus data.

    Returns:
        tuple: A tuple containing two lists - human samples and original corpus words.
    """
    human_samples = []
    original_corpus_words = []

    for key, value in provo_data.items():
        human_samp = flatten_list([[x['pred']] * int(x['count']) for x in value['human']])
        original_word = value['original']['pred']

        human_samples.append(human_samp)
        original_corpus_words.append(original_word)

    return human_samples, original_corpus_words

def prepare_generated_data(results_path):
    """Prepare generated samples for evaluation from the given JSON results file.

    Args:
        results_path (str): Path to the JSON file containing generated results.

    Returns:
        tuple: A dictionary with lists of generated samples categorized by key, and the original results.
    """
    with open(results_path, 'r') as file:
        results = json.load(file)

    sample_categories = {
        'base_samples': [],
        'fs_k3_samples': [],
        'fs_k5_samples': [],
        'fs_k7_samples': [],
        'fs_k5_DT_samples': [],
        'fs_k5_without_DT_samples': []
    }

    for key, value in results.items():
        sample_categories['base_samples'].append(value['base'])
        sample_categories['fs_k3_samples'].append(value['fs_k_3'])
        sample_categories['fs_k5_samples'].append(value['fs_k_5'])
        sample_categories['fs_k7_samples'].append(value['fs_k_7'])
        sample_categories['fs_k5_DT_samples'].append(value['fs_k_5_DT'])
        sample_categories['fs_k5_without_DT_samples'].append(value['fs_k_5_without_DT'])

    return sample_categories, results

def calculate_tvds(human_samples, generated_samples):
    """Calculate TVD values for human and generated samples.

    Args:
        human_samples (list): List of human samples.
        generated_samples (dict): Dictionary of generated samples.

    Returns:
        dict: Dictionary containing TVD values.
    """
    results = {
        'tvd_human_human': [],
        'tvd_human_base': [],
        'tvd_human_fs_k3': [],
        'tvd_human_fs_k5': [],
        'tvd_human_fs_k7': [],
        'tvd_human_fs_k5_DT': [],
        'tvd_human_fs_k5_without_DT': []
    }

    for i, sample in enumerate(human_samples):
        results['tvd_human_human'].append(get_tvd_value_human_human(sample))
        results['tvd_human_base'].append(get_tvd_value_human_model(sample, generated_samples['base_samples'][i]))
        results['tvd_human_fs_k3'].append(get_tvd_value_human_model(sample, generated_samples['fs_k3_samples'][i]))
        results['tvd_human_fs_k5'].append(get_tvd_value_human_model(sample, generated_samples['fs_k5_samples'][i]))
        results['tvd_human_fs_k7'].append(get_tvd_value_human_model(sample, generated_samples['fs_k7_samples'][i]))
        results['tvd_human_fs_k5_DT'].append(get_tvd_value_human_model(sample, generated_samples['fs_k5_DT_samples'][i]))
        results['tvd_human_fs_k5_without_DT'].append(get_tvd_value_human_model(sample, generated_samples['fs_k5_without_DT_samples'][i]))

    return results

def main():
    # List of JSON files to load Provo Corpus data from
    input_data_files = [
        'paragraphs/Paragraphs-45-47.json',
        'paragraphs/Paragraphs-48-50.json',
        'paragraphs/Paragraphs-51-53.json',
        'paragraphs/Paragraphs-54-55.json'
    ]

    # Load and process Provo Corpus data
    provo_data = get_provo_data(input_data_files)

    # Prepare human samples and original corpus words
    human_samples, original_corpus_words = prepare_human_data(provo_data)

    # Prepare generated samples for evaluation
    RESULTS_PATH = 'prompt_results/results_complete.json'
    generated_samples, results = prepare_generated_data(RESULTS_PATH)

    # Calculate TVD values
    tvd_results = calculate_tvds(human_samples, generated_samples)

    # Update results with TVD values
    for i, key in enumerate(results.keys()):
        results[key].update({
            'tvd': tvd_results['tvd_human_base'][i],
            'tvd_fs_k3': tvd_results['tvd_human_fs_k3'][i],
            'tvd_fs_k5': tvd_results['tvd_human_fs_k5'][i],
            'tvd_fs_k5_DT': tvd_results['tvd_human_fs_k5_DT'][i],
            'tvd_fs_k5_without_DT': tvd_results['tvd_human_fs_k5_without_DT'][i],
            'tvd_fs_k7': tvd_results['tvd_human_fs_k7'][i]
        })

    # Save updated results
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()
