import numpy as np
import re
import itertools
from collections import Counter
import pdb

def add_noise(x, embedding_dim, random_type=None, word_keep=1.0, mean=1.0, weight=0.0, replace_map = None, grad_noise=None):
    seq_length = len(x[0])
    batch_size = len(x)
    noise = np.ones([batch_size, seq_length, embedding_dim])
    
    #turn out to be bad.
    #force left and right paddings' word embedding to be zero.
    #for i in range(batch_size):
    #    noise[i,:4,:] = 0.0
    #    for j in range(seq_length)[::-1]:
    #        if x[i][j] != 0:
    #            noise[i,j+1:,:] = 0.0
        
    if random_type in ['Bernoulli', 'Bernoulli-semantic', 'Bernoulli-adversarial']:
        noise = noise / word_keep
    elif random_type in ['Gaussian', 'Gaussian-adversarial', 'Bernoulli-word', 'Bernoulli-idf', 'Bernoulli-polary', 'Replace']:
        pass
    
    if random_type in ['Adversarial','Gaussian-adversarial','Adversarial-single','Gaussian-adversarial-single','Gaussian-adversarial-norm']:
        if len(grad_noise) > 0:
            # print("Matrix")
            return grad_noise
        else:
            return np.ones([batch_size, seq_length, embedding_dim])*grad_noise

    for bi in range(batch_size):
        if random_type == 'Bernoulli':
            noise[bi,:,:] = np.random.choice(2,size=(seq_length, embedding_dim), p=[1-word_keep, word_keep])
        if random_type == 'Gaussian':
            noise[bi,:,:] = np.random.normal(mean, weight, [seq_length, embedding_dim])
        if random_type == 'Bernoulli-word':
            # x = list(x)
            # x[bi] = x[bi] * np.random.choice(2, size=seq_length, p=[1-word_keep, word_keep])#change x by shallow copy
            noise[bi,:,:] *= np.random.choice(2, size=(seq_length, 1), p=[1-word_keep, word_keep])
        if random_type == 'Bernoulli-semantic':
            noise[bi,:,:] *= np.random.choice(2, size=(embedding_dim), p=[1-word_keep, word_keep])
        if random_type == 'Bernoulli-idf':
            pass
        if random_type == 'Replace':
            positions = np.random.choice(2, size=seq_length, p=[1-word_keep, word_keep])
            positions = np.argwhere(positions == 0)
            for pi in positions:
                #print(x[bi])
                #print(int(x[bi][pi]))
                
                if int(x[bi][pi]) < len(replace_map) and replace_map[int(x[bi][pi])] != None:
                    word_choices, probs = replace_map[int(x[bi][pi])]
                    choose_wid = np.random.choice(word_choices, p=probs)
                    x[bi][pi] = choose_wid
                else:
                    x[bi][pi] = 0
            #x[bi] = np.random.choice(2, size=(seq_length), p=[1-word_keep, word_keep])#change x by shallow copy
    return noise