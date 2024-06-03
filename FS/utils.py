import random
import numpy as np
import torch
import json
import nltk
from generate import generate_prompt
import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

def set_seed(seed):# Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_json(file_path):
    # Open json file
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def get_provo_data(file_paths):
    """Load and merge Provo Corpus data from multiple JSON files into a single dictionary.

    Args:
        file_paths (list): List of paths to the JSON files containing Provo Corpus data.

    Returns:
        dict: A dictionary containing merged data from all provided files.
    """
    joint_dict = {}
    count = 0

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = json.load(file)
            for text_id, words in data.items():
                for word_num, word_data in words.items():
                    joint_dict[count] = word_data
                    joint_dict[count]['original_positioning'] = {'text_id': text_id, 'word_num': word_num}
                    count += 1

    print(f'Total number of data points: {count}')
    return joint_dict


# Get the ith element of dictionary
def get_ith_element(d, i):
    """
    Get the i-th element from a dictionary.

    :param d: The dictionary from which to get the element.
    :param i: The index of the element to retrieve.
    :return: A tuple (key, value) of the i-th element.
    :raises IndexError: If the index is out of range.
    """
    if not isinstance(d, dict):
        raise TypeError("Input must be a dictionary.")
    
    if i < 0 or i >= len(d):
        raise IndexError("Index out of range.")
    
    items = list(d.items())
    return items[i]

def get_pos_tags(sentence):
    # Tokenize the sentence
    tokens = nltk.word_tokenize(sentence)
    # Get POS tags
    pos_tags = nltk.pos_tag(tokens)
    return pos_tags

# Generate prompt dictionary for few-shot learning
def prompt_dict_for_fs():
    set_seed(42)

    with open('provo_dict.json', 'r') as f:
        df_dict = json.load(f)

    # Divide in train_dict and test_dict from text_id 45 or higher it is test
    train_dict = {}
    test_dict = {}
    for key, value in df_dict.items():
        if value['text_id'] < 45:
            train_dict[key] = value
        else:
            test_dict[key] = value
    k = 7
    results_df = []
    prompt_dict = {}
    for i in range(2116, 2747):
        # If keyerror
        try:
            prompt, idx = generate_prompt(train_dict, test_dict, k, i)
            # print(prompt)
            # Make row
            # results_df.append({'QID': i, 'Example_IDs': idx['train'], 'prompt' : prompt,'Response': ''})
            dicti = {'prompt': prompt, 'Example_IDs': idx['train']}
            # Add new key and value to prompt_dict
            prompt_dict[i] = dicti
        except:
            print('KeyError')

        # Results_df to df
        results_df = pd.DataFrame(results_df)
        results_df
        # Save as json
        with open(f'prompt_dict_k_{k}.json', 'w') as f:
            json.dump(prompt_dict, f, indent=4)


def classify_tvd_indices(tvd_values, tolerance=0.1):
    """Classify indices based on TVD values. 
    
    Parameters:
    - tvd_values: List of TVD values to classify.
    - good_range: Tuple specifying the range of good TVD values.
    - bad_threshold: Threshold for bad TVD values.

    Returns: 
    - good_indices: List of indices with good TVD values.
    - bad_indices: List of indices with bad TVD values.
    """
    mean = 0.4
    good_indices = [i for i, x in enumerate(tvd_values) if mean - tolerance < x < mean + tolerance]
    bad_indices = [i for i, x in enumerate(tvd_values) if x >= mean + tolerance or x <= mean - tolerance]
    return good_indices, bad_indices

"""Score functions for evaluation of generated samples"""


def get_estimator(elements):
    """Get the MLE estimate given all words (probability of a word equals its relative frequency)"""
    c = Counter(elements)
    support = list(c.keys())
    counts = list(c.values())
    probs = [count / sum(counts) for count in counts]

    return (support, probs)

def get_common_support(support1, support2):
    """Receives supports from two distributions and return all elements appearing in at least one of them"""
    return set(support1).union(set(support2))

def change_support(old_support, old_probs, new_support):
    """Create new support by adding elements to a support that did not exist before
     (hence, their probability value is 0)"""
    new_probs = []
    for item in new_support:
        if item in old_support:
            ind = old_support.index(item)
            new_probs.append(old_probs[ind])
        else:
            new_probs.append(0)
    return list(new_support), new_probs

def get_tvd(probs1, probs2):
    """Receives the probabilities of 2 distributions to compare and returns their TVD (Total Variation Distance)"""
    tvd = np.sum(np.abs(np.array(probs1) - np.array(probs2)))/2
    return tvd

def get_oracle_elements(words, seed = 42):
    """We receive a list of words and we create two disjoint subsets
    from it by sampling without replacement from them.
    We return the two disjoint sets of words"""
    random.seed(seed)

    #if the length of the list is odd, we remove one element at random to make the list even,
    #to ensure the two disjoint subsets are of equal length
    if (len(words) % 2 == 1):
        remove_word = random.sample(words, 1)
        words.remove(remove_word[0])

    #We sample the words that will belong in the first subset and create the second subset by removing
    #from the full word list the ones sampled in the first subset
    subset1 = random.sample(words, len(words)//2)
    subset2 = words.copy()
    for item in subset1:
        subset2.remove(item)

    return subset1, subset2

def get_tvd_value_human_human(human_samples, seed=42):
    "human_samples: list of continuations for one context"
    oracle1, oracle2 = get_oracle_elements(human_samples, seed=seed)

    #Get the MLE estimates given the samples
    support_or1, probs_or1 = get_estimator(oracle1)
    support_or2, probs_or2 = get_estimator(oracle2)

    common_support = get_common_support(support_or1, support_or2)

    support_or1, probs_or1 = change_support(support_or1, probs_or1, common_support)
    support_or2, probs_or2 = change_support(support_or2, probs_or2, common_support)

    tvd = get_tvd(probs_or1, probs_or2)

    return tvd


def get_tvd_value_human_model(human_samples, model_samples):

    support_human, probs_human = get_estimator(human_samples)
    support_model, probs_model = get_estimator(model_samples)

    common_support = get_common_support(support_human, support_model)

    support_human, probs_human = change_support(support_human, probs_human, common_support)
    support_model, probs_model = change_support(support_model, probs_model, common_support)

    tvd = get_tvd(probs_human, probs_model)

    return tvd


def plot_tvd_distributions(all_tvd_data, title='TVD Distributions', filename='tvd_distributions'):
    # Data Preparation for Plotting
    names = []
    tvd = []
    for name, values in all_tvd_data.items():
        names.extend([name] * len(values))
        tvd.extend(values)

    # Create a DataFrame
    df = pd.DataFrame({'Name': names, 'TVD': tvd})

    # Set Seaborn style and font size
    sns.set(rc={'figure.figsize': (10, 6)})
    sns.set(font_scale=2)
    plt.rcParams['font.family'] = 'Serif'
    plt.rcParams['font.size'] = 12
    # Set title size of plot

    # Define line styles
    line_styles = ['-', '--', ':', '-.', (0, (5, 10))]

    avg_tvds = []
    for name, values in all_tvd_data.items():
        # print(f'Average TVD {name}: {np.mean(np.array(values))}')
        avg = np.mean(np.array(values))
        # Round to 3 decimal places
        avg = round(avg, 3)
        avg_tvds.append(avg)
        # Round to 3 decimal places

    
    # Plotting
    for i, hue_level in enumerate(df['Name'].unique()):
        subset = df[df['Name'] == hue_level]
        sns.kdeplot(subset['TVD'], label=f'{hue_level}, $\mu$={avg_tvds[i]}', linestyle=line_styles[i % len(line_styles)], clip=(0, 1), fill=True, alpha=0.2, linewidth=3)

    # Add legend
    plt.title(title, fontsize=24)
    plt.legend(loc='upper left', fontsize=12)

    plt.tight_layout()
    # Save plot as PDF
    plt.savefig(f'{filename}.pdf', dpi=300, bbox_inches='tight')
    # Display the plot
    plt.show()

    # Print average TVDs
    print('---------------------------')
    for name, values in all_tvd_data.items():
        print(f'Average TVD {name}: {np.mean(np.array(values))}')


def count_last_pos_tags(good_idx, results):
    """
    Counts the POS tags for the last word in the context of the given indices.

    Args:
        good_idx (list): List of indices to process.
        final_scores (object): Object containing results from which context is extracted.

    Returns:
        dict: A dictionary with POS tags as keys and their counts as values.
    """

    pos_tags = []

    for idx in good_idx:
        try:
            _, val = get_ith_element(results, idx)
            last_word = val['context'][-1]
            # Ensure last_word is not empty and is a valid string
            pos_tag = get_pos_tags(last_word)[0][1]
            pos_tags.append(pos_tag)

        except (IndexError, KeyError, TypeError) as e:
            print(f"Error processing index {idx}: {e}")
            continue
    pos_tags = Counter(pos_tags)
    # Use Counter to count the number of each POS tag
    return pos_tags