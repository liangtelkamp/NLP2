import pandas as pd
import numpy as np
import os
import sys
import re
from collections import Counter
import random
import torch
import os
import re
import json
import math
def preprocess_provo_corpus(provo):
    """Reading the raw Provo Corpus dataset and create a dictionary with all useful
    information we might need from it"""
    predict_norms = pd.read_csv(provo, sep='\t')
    paragraphs = predict_norms.groupby('Text_ID')['Text'].max()

    provo_processed = {}
    count = 0
    for text_id in range(1,56): #iterate over all provo paragraphs
        for word_num in predict_norms[predict_norms['Text_ID'] == text_id]['Word_Number'].unique(): #iterating over all words in each text
            word_dist = predict_norms[(predict_norms['Text_ID'] == text_id) & (predict_norms['Word_Number'] == word_num)]
            unique_human_words = word_dist['Response'].unique() #all human answered words for each context
            unique_word_dist = []
            for word in unique_human_words:
                unique_word_count = sum(word_dist[word_dist['Response'] == word]['Response_Count']) #getting all counts of the unique word and summing them
                unique_word_dist.append((word, unique_word_count))

            provo_processed[count] = {}
            provo_processed[count]['context_with_original_word'] = paragraphs[text_id].split(' ')[:int(word_num)]
            provo_processed[count]['context'] = paragraphs[text_id].split(' ')[:(int(word_num)-1)]
            provo_processed[count]['original_positioning'] = {'text_id':text_id, 'word_num':word_num}
            provo_processed[count]['human_next_word_pred'] = unique_word_dist

            count = count + 1

    return provo_processed

input_data = os.path.join(os.getcwd(), 'Provo_Corpus.tsv')
data = preprocess_provo_corpus(input_data)

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

def get_oracle_elements(words, seed = 0):
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

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
#Switching to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)



def get_model_samples(model, tokenizer, context, n_samples = 5, add_tokens = 5, seed = 0, pad_token = '<|endoftext|>', top_k = 0):
    """Given a context we return the words that were generated during ancestral sampling for n_samples"""
    # inputs_ids = tokenizer(context, return_tensors="pt")['input_ids']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs_ids = tokenizer(context, return_tensors="pt")['input_ids'].to(device)
    # inputs_ids.to(device)
    torch.manual_seed(seed)
    # outputs = model.generate(inputs_ids, do_sample=True, num_beams = 1, num_return_sequences= n_samples,
    #                          max_new_tokens = add_tokens, pad_token_id = pad_token, top_k= top_k)
    outputs = model.generate(inputs_ids, do_sample=True, num_beams = 1, num_return_sequences= n_samples,
                             max_new_tokens = add_tokens) #This samples unbiasedly from the next-token distribution of the model for add_tokens tokens, n_samples times

    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    #Remove context to keep only generated text
    outputs = [x.replace(context + ' ', '').replace(context, '').replace('\n', '') for x in outputs]
    #Remove punctuation
    outputs = [re.sub(r'[^\w\s]', '', x) for x in outputs] #removing punctuation
    list_of_words = [x.split(' ') for x in outputs]

    # print(f'len outputs = {len(list_of_words)}')

    sampled_words = []
    for generation in list_of_words:
        if set(generation) == {''}:
            sampled_words.append('Failed to generate word')
        else:
            sampled_words.append(next(x.lower() for x in generation if x))

    return sampled_words

# generating k_fold data split 
def entropy(probabilities):
    """
    Calculate the entropy of a probability distribution.

    Args:
    - probabilities (list): List of probabilities representing a probability distribution.

    Returns:
    - float: Entropy value in bits.
    """
    entropy_value = 0.0

    # Calculate entropy using the formula
    for prob in probabilities:
        if prob > 0:  # Only consider non-zero probabilities to avoid log(0) which is undefined
            entropy_value -= prob * math.log2(prob)


    return entropy_value
data_dict = torch.load('data_dictionary.pt')
i_values = [5, 15, 25, 35]
k_fold_data = {0:{}, 1: {}, 2:{}, 3:{}}
k = 0
for i in i_values:
    train_data_small = {k: v for k, v in data_dict.items() if v['paragraph_num'] < i} # 0:i
    val_data_small = {k: v for k, v in data_dict.items() if v['paragraph_num'] >= i and v['paragraph_num'] < i+10} # i : i+10
    train_data_small.update({k: v for k, v in data_dict.items() if v['paragraph_num'] >= i+10 and v['paragraph_num'] < 45}) #i+10 : 45
    test_data = {k: v for k, v in data_dict.items() if v['paragraph_num'] >= 45}

    k_fold_data[k]['train_data'] = train_data_small
    k_fold_data[k]['val_data'] = val_data_small
    k_fold_data[k]['test_data'] = test_data
    print(f'k_fold statistics \nk: {k}\ntrain: {len(train_data_small)}\nval: {len(val_data_small)}\ntest:{len(test_data)}\n')
    k += 1

for j in range(4):
    for i, sample in enumerate(k_fold_data[j]['train_data'].keys()):
        human_probs = np.array([prob for _, prob in k_fold_data[j]['train_data'][sample]['human_next_word_pred']])
        he = entropy(human_probs)
        k_fold_data[j]['train_data'][sample]['human_entropy'] = he
    
    for i, sample in enumerate(k_fold_data[j]['test_data'].keys()):
        human_probs = np.array([prob for _, prob in k_fold_data[j]['test_data'][sample]['human_next_word_pred']])
        he = entropy(human_probs)
        k_fold_data[j]['test_data'][sample]['human_entropy'] = he

data_entropy_split = {0:{}, 1: {}, 2:{}, 3:{}}
for j in range(4):
    h_entropies = []
    h_entropies_large = []
    train_data_small = {}
    test_data_small = {}

    for i, sample in enumerate(k_fold_data[j]['train_data'].keys()):

        if k_fold_data[j]['train_data'][sample]['human_entropy'] > 2.5:
            train_data_small[sample] = k_fold_data[j]['train_data'][sample]
    
    for i, sample in enumerate(k_fold_data[j]['test_data'].keys()):

        if k_fold_data[j]['test_data'][sample]['human_entropy'] > 2.5:
            test_data_small[sample] = k_fold_data[j]['test_data'][sample]
            
    data_entropy_split[j]['train_data'] = train_data_small
    data_entropy_split[j]['val_data'] = k_fold_data[j]['val_data']
    data_entropy_split[j]['test_data'] = test_data_small

for j in range(4):
    print(f'entropy split fold {j}')
    print(len(data_entropy_split[j]['train_data']))


def get_tvd_value_human_model(i, test_data, model, tokenizer):
    test_data = test_data[i]
    context = ' '.join(test_data['context'])

    # print(f'context test = {context}')
    # print(f'context data = {data[i]["context"]}')

    model_samples = get_model_samples(model, tokenizer, context, n_samples = 40)

    #Remove samples for which we failed to generate a complete word
    model_samples = [x for x in model_samples if x != 'Failed to generate word']

    l_words = [[item[0]]*int(item[1]) for item in data[i]['human_next_word_pred']]
    l_words = [item for row in l_words for item in row]
    human_samples = l_words.copy()

    # print(f'model samples = {model_samples}')
    # print(f'human samples = {human_samples}')

    support_human, probs_human = get_estimator(human_samples)
    support_model, probs_model = get_estimator(model_samples)

    common_support = get_common_support(support_human, support_model)

    support_human, probs_human = change_support(support_human, probs_human, common_support)
    support_model, probs_model = change_support(support_model, probs_model, common_support)

    tvd = get_tvd(probs_human, probs_model)

    return model_samples, tvd

def get_tvd_value_human_human(i):

    l_words = [[item[0]]*int(item[1]) for item in data[i]['human_next_word_pred']]
    l_words = [item for row in l_words for item in row]
    oracle1, oracle2 = get_oracle_elements(l_words)

    #Get the MLE estimates given the samples
    support_or1, probs_or1 = get_estimator(oracle1)
    support_or2, probs_or2 = get_estimator(oracle2)

    common_support = get_common_support(support_or1, support_or2)

    support_or1, probs_or1 = change_support(support_or1, probs_or1, common_support)
    support_or2, probs_or2 = change_support(support_or2, probs_or2, common_support)

    tvd = get_tvd(probs_or1, probs_or2)

    return l_words, tvd

# # evaluate baseline on train data 
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# tokenizer.pad_token = tokenizer.eos_token
# model = GPT2LMHeadModel.from_pretrained("gpt2")

# test_data =  k_fold_data[0]['train_data'] # SIEN
# tvd_human_model_values = []
# tvd_human_human_values = []

# train_data_stats = {}

# k = 0
# for i in test_data.keys():
#     model_samples, tvd_human_model = get_tvd_value_human_model(i, test_data, model, tokenizer)
#     human_samples, tvd_human_human = get_tvd_value_human_human(i) # SIEN

#     tvd_human_model_values.append(tvd_human_model)
#     tvd_human_human_values.append(tvd_human_human)

#     if k%50==0:
#         print(f'--------------------')
#         print(f'k = {k}')
#         print(f'tvd human model = {tvd_human_model}')
#         print(f'tvd human human = {tvd_human_human}')

    

#     k+= 1

# train_data_stats['tvd_human_human'] = tvd_human_human_values
# train_data_stats['tvd_human_model'] = tvd_human_model_values


# with open('tvd_values_baseline_trainset' + '.json', 'w') as fp:
#     json.dump(train_data_stats, fp)


# LOAD K-FOLD MODELS
kfold_finetuned_models = ['/home/scur1303/nlp2_sien/resutls_rq3/rq3_fold_0', '/home/scur1303/nlp2_sien/resutls_rq3/rq3_fold_1', '/home/scur1303/nlp2_sien/resutls_rq3/rq3_fold_2', '/home/scur1303/nlp2_sien/resutls_rq3/rq3_fold_3']
sample_stats = {}
kfold_tvd_stats = {0: {}, 1: {}, 2:{}, 3:{}}

for f in range(4):
    print(f'\nFOLD = {f}\n')
    tokenizer = GPT2Tokenizer.from_pretrained(kfold_finetuned_models[f])
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(kfold_finetuned_models[f]).to(device)

    test_data =  data_entropy_split[f]['test_data'] # SIEN
    tvd_human_model_values = []
    tvd_human_human_values = []
    model_samples_list = []
    human_samples_list = []

    k = 0
    for i in test_data.keys():
        model_samples, tvd_human_model = get_tvd_value_human_model(i, test_data, model, tokenizer)
        human_samples, tvd_human_human = get_tvd_value_human_human(i) # SIEN

        tvd_human_model_values.append(tvd_human_model)
        tvd_human_human_values.append(tvd_human_human)
        model_samples_list.append(model_samples)
        human_samples_list.append(human_samples)

        if k%50==0:
            print(f'--------------------')
            print(f'k = {k}')
            print(f'tvd human model = {tvd_human_model}')
            print(f'tvd human human = {tvd_human_human}')
        
        k+= 1
    
    kfold_tvd_stats[f]['tvd_human_human'] = tvd_human_human_values
    kfold_tvd_stats[f]['tvd_human_model'] = tvd_human_model_values

    kfold_tvd_stats[f]['human_samples'] = tvd_human_model_values
    kfold_tvd_stats[f]['model_samples'] = tvd_human_model_values
    

    
with open('ft_rq3_tvd_samples' + '.json', 'w') as fp:
    json.dump(kfold_tvd_stats, fp)

# BASELINE samples
# kfold_finetuned_models = ['/home/scur1303/nlp2_sien/resutls_finetuning_kfold/test_model_fold_0', '/home/scur1303/nlp2_sien/resutls_finetuning_kfold/test_model_fold_1', '/home/scur1303/nlp2_sien/resutls_finetuning_kfold/test_model_fold_2', '/home/scur1303/nlp2_sien/resutls_finetuning_kfold/test_model_fold_3']
# kfold_tvd_stats = {0: {}, 1: {}, 2: {}, 3: {}}
# sample_stats = {}
# for f in range(1):
#     print(f'\nFOLD = {f}\n')
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# tokenizer.pad_token = tokenizer.eos_token
# model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# test_data =  k_fold_data[0]['test_data'] # SIEN
# tvd_human_model_values = []
# tvd_human_human_values = []
# model_samples_list = []
# human_samples_list = []

# k = 0
# for i in test_data.keys():
#     model_samples, tvd_human_model = get_tvd_value_human_model(i, test_data, model, tokenizer)
#     human_samples, tvd_human_human = get_tvd_value_human_human(i) # SIEN

#     tvd_human_model_values.append(tvd_human_model)
#     tvd_human_human_values.append(tvd_human_human)
#     model_samples_list.append(model_samples)
#     human_samples_list.append(human_samples)

#     if k%50==0:
#         print(f'--------------------')
#         print(f'k = {k}')
#         print(f'tvd human model = {tvd_human_model}')
#         print(f'tvd human human = {tvd_human_human}')
    
#     k+= 1

# kfold_tvd_stats[f]['tvd_human_human'] = tvd_human_human_values
# kfold_tvd_stats[f]['tvd_human_model'] = tvd_human_model_values

# sample_stats['human_samples'] = human_samples_list
# sample_stats['model_samples'] = model_samples_list
# sample_stats['tvd_human_model'] = tvd_human_model_values
# sample_stats['tvd_human_human'] = tvd_human_human_values

# with open('baseline_samples_tvd' + '.json', 'w') as fp:
#     json.dump(sample_stats, fp)
