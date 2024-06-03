import torch
import pandas as pd
import numpy as np
import os
import sys
import re
from collections import Counter
import random
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from transformers import AdamW, get_linear_schedule_with_warmup

import numpy as np
import pandas as pd
import torch
import torch.distributions as td
import torch.utils.data as data
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers.trainer_utils import set_seed

import os
import re
import json


# generating k_fold data split 
data = torch.load('data_dictionary.pt')
i_values = [5, 15, 25, 35]
k_fold_data = {0:{}, 1: {}, 2:{}, 3:{}}
k = 0
for i in i_values:
    train_data_small = {k: v for k, v in data.items() if v['paragraph_num'] < i} # 0:i
    val_data_small = {k: v for k, v in data.items() if v['paragraph_num'] >= i and v['paragraph_num'] < i+10} # i : i+10
    train_data_small.update({k: v for k, v in data.items() if v['paragraph_num'] >= i+10 and v['paragraph_num'] < 45}) #i+10 : 45
    test_data = {k: v for k, v in data.items() if v['paragraph_num'] >= 45}

    k_fold_data[k]['train_data'] = train_data_small
    k_fold_data[k]['val_data'] = val_data_small
    k_fold_data[k]['test_data'] = test_data
    print(f'k_fold statistics \nk: {k}\ntrain: {len(train_data_small)}\nval: {len(val_data_small)}\ntest:{len(test_data)}\n')
    k += 1

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

#Switching to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class log_probs:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        self.bos_token = '<|endoftext|>'
        self.pad_token = '<|endoftext|>'

    def get_log_prob_from_loss(self, sentence):
        # print(f'sentence = {sentence}')
        """Receives a sentence and produces its probability under the model"""
        inputs =  self.tokenizer(f"{self.bos_token} {sentence}", return_tensors="pt")
        inputs.to(device)
        # with torch.no_grad():
        # print(inputs["input_ids"])
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        log_prob = - loss*( len(inputs["input_ids"][0]) - 1) #multiplying by length since the total loss
        #was the average of losses, 1/N Sigma(L_CE) and subtracting 1 to not account for the bos_token

        return log_prob
        # return log_prob.item()

    def get_log_probs_from_prob_dist(self, sentences):
        """Receives a list of sentences and returns a list of tensors with the probabilities of each sentence under the model"""

        #We pad to match the length of the longest sentence
        inputs = self.tokenizer(sentences, padding='longest', return_length=True, return_special_tokens_mask=True,
                                return_tensors="pt")
        inputs.to(device)

        #The observation per position is shifted with respect to the input
        obs_ids = torch.roll(inputs['input_ids'], -1, -1)

        # with torch.no_grad():
        outputs = self.model(inputs["input_ids"])
        output_logits = outputs.logits

        pT = td.Categorical(logits=output_logits)
        tok_log_probs = pT.log_prob(obs_ids).to(device)
        zeroes = torch.zeros(tok_log_probs.shape[0],tok_log_probs.shape[1]).to(device)

        #Log probabilities for non-padded tokens
        tok_log_probs_no_pad = torch.where(obs_ids == self.tokenizer.eos_token_id, zeroes, tok_log_probs)

        return tok_log_probs_no_pad.sum(-1) #the log_prob of each sentence

    def get_log_probs_from_loss_sien(self, sentences):
        #We pad to match the length of the longest sentence
        log_probs = []
        for sentence in sentences:
            log_probs.append(self.get_log_prob_from_loss(sentence))

        return torch.tensor(log_probs).to(device)


class cond_log_probs:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.log_probs_obj = log_probs(self.model, self.tokenizer)

    def get_log_prob_context(self, context):
        self.log_prob_context = self.log_probs_obj.get_log_prob_from_loss(context)
        return self.log_prob_context

    def conditional_log_prob_of_word_given_context(self, word, context):
        """Receives one word and a context and returns the log probability of the word given the context under the model"""
        context_and_word = context + ' ' + word
        log_prob_context_and_word = self.log_probs_obj.get_log_prob_from_loss(context_and_word)
        log_prob_conditional_on_word = log_prob_context_and_word - self.log_prob_context

        return(log_prob_conditional_on_word)

    def conditional_log_prob_of_word_list_given_context(self, words, context, log_prob_context):
        """Receives a list of words and a context and returns a list of log probabilities of the words given
           the context under the model"""
        bos_token = '<|endoftext|>'
        sentences = [f'{self.tokenizer.bos_token} {context} {x}' for x in words]
        log_probs_context_and_words = self.log_probs_obj.get_log_probs_from_prob_dist(sentences)

        # sentences = [f'{context} {x}' for x in words] #each sentence is the context and one of the words # SIEN
        #log_probs_context_and_words= self.log_probs_obj.get_log_probs_from_loss_sien(sentences) # sien

        log_probs_conditional_on_words = log_probs_context_and_words - log_prob_context

        # for i, prob in enumerate(log_probs_conditional_on_words):
        #     if prob > 0:
        #         print(f' log probability conditional > 0 {prob}')
        #         print(f'prob = {torch.exp(prob)}')
        #         print(f'log probs = {log_probs_context_and_words[i]} - {log_prob_context}')
        #         print(f'w = {words[i]}')
        #         print(f'context = {context}')
        #         print(f' q(v, c) = {torch.exp(log_probs_context_and_words[i])}')
        #         print(f' q(c) = {torch.exp(torch.tensor(log_prob_context))}')

        return log_probs_conditional_on_words  # sien
        # return([x.item() for x in log_probs_conditional_on_words])
def CE_variability(cond_log_probs_obj, context, words, human_probs):

    log_prob_context = cond_log_probs_obj.get_log_prob_context(context)
    log_probs_cond = cond_log_probs_obj.conditional_log_prob_of_word_list_given_context(words, context, log_prob_context)

    loss = torch.sum( - human_probs*log_probs_cond)

    cond_probs = torch.exp(log_probs_cond).detach().cpu().numpy()
    human_probs = human_probs.detach().cpu().numpy()

    return loss, cond_probs, human_probs



def train_one_fold(model, tokenizer, optimizer, scheduler, epochs, train_data_small, val_data_small, k, dict_parameters):
    training_stats = []
    train_losses = []
    train_loss_avg = []
    cond_log_probs_obj = cond_log_probs(model, tokenizer)
    i_list = [20, 100, 150, 200, 300, 400, 500]
    i_track_dict = {20: [], 100: [], 150: [], 200:[], 300:[], 400:[], 500:[]}
    print(f'Training fold {k}...')
    for epoch in range(epochs):
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))

        total_train_loss = 0
        model.train()

        i = 0
        for context_dict in train_data_small.values():

            context = ' '.join(context_dict['context'])
            words = [word for word, _ in context_dict['human_next_word_pred']]
            human_probs = torch.tensor([prob for _, prob in context_dict['human_next_word_pred']]).to(device)
            model.zero_grad()

            ce_loss, cond_probs, human_probs = CE_variability(cond_log_probs_obj, context, words, human_probs)

            i += 1
            if i%100 == 0:
                # print(f'avg loss {i} = {total_train_loss/i}')
                train_loss_avg.append(total_train_loss/i)
            ce_loss.backward()
            optimizer.step()
            scheduler.step()

            train_losses.append(ce_loss.item())
            total_train_loss += ce_loss.item()

            if i in i_list:
                i_track_dict[i].append((cond_probs.tolist(), human_probs.tolist()))

            # if i == 201:
            #     break

        avg_train_loss = total_train_loss/len(train_data_small)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        # print("  Training epoch took: {:}".format(training_time))

        print("")
        print("Running Validation...")


        total_val_loss = 0

        i=0

        for context_dict in val_data_small.values():

            context = ' '.join(context_dict['context'])
            words = [word for word, _ in context_dict['human_next_word_pred']]
            human_probs = torch.tensor([prob for _, prob in context_dict['human_next_word_pred']]).to(device)

            with torch.no_grad():

                ce_loss, cond_probs, human_probs = CE_variability(cond_log_probs_obj, context, words, human_probs)

            i += 1
            # if i%100 == 0:
            #     # print(f'avg loss {i} = {total_val_loss/i}')
            total_val_loss += ce_loss.item()


        avg_val_loss = total_val_loss/len(val_data_small)

        print("")
        print("  Average validation loss: {0:.2f}".format(avg_val_loss))

        training_stats.append({'epoch': epoch + 1,
                                'Training loss': avg_train_loss,
                                'Valid loss': avg_val_loss})

    print('')
    print('training complete!')
    print(training_stats)

    model_version = f'lmhead_model_fold_{k}'
    #Save training statistics (training and validation losses)
    with open('training_stats' + model_version + '.json', 'w') as fp:
        json.dump(training_stats, fp)
        json.dump(train_losses, fp)
        json.dump(i_track_dict, fp)
        

    #Saving model
    output_dir = f'./lmhead_model_fold_{k}/'

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save a trained model, configuration and tokenizer. They can then be reloaded using 'from_pretrained()'
    gpt2_provo_fine_tuned = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    gpt2_provo_fine_tuned.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open('training_parameters' + model_version + '.json', 'w') as fp:
        json.dump(dict_parameters, fp)


# for k in range(4):
k = 3
# ensuring all dropout is off
model = GPT2LMHeadModel.from_pretrained('gpt2', resid_pdrop = 0., embd_pdrop = 0., attn_pdrop = 0., summary_first_dropout = 0.)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

seed = 0
set_seed(seed)

# freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Choose which layers we would like to continue training.
# We only want to update the weights of the last layer

module_to_unfreeze = [model.transformer.h[11].ln_1, model.transformer.h[11].ln_2, model.transformer.h[10].ln_1, model.transformer.h[10].ln_2, model.transformer.ln_f, model.lm_head]

for module in module_to_unfreeze:
    for param in module.parameters():
        param.requires_grad = True

# print(f'model = {model}')

lr = 0.0001
# model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
epochs = 10 # set number of epochs

# Total number of training steps is [number of batches] x [number of epochs].
total_steps = len(train_data_small) * epochs

# Create the learning rate scheduler. This changes the learning rate as the training loop progresses
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 10, num_training_steps = total_steps)

model = model.to(device)

dict_parameters = {'epochs': epochs, 'lr': lr}
print('dict paramters')
train_data_small = k_fold_data[k]['train_data']
val_data_small = k_fold_data[k]['val_data']

train_one_fold(model, tokenizer, optimizer, scheduler, epochs, train_data_small, val_data_small, k, dict_parameters)