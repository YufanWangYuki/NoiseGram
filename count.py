import errant

import torch
torch.cuda.empty_cache()
import random
import time
import os
import logging
import argparse
import sys
import numpy as np

from utils.dataset_1toN import Dataset_1toN
from utils.dataset_eval import Dataset_EVAL
from utils.dataset import Dataset
from utils.misc import save_config, validate_config
from utils.misc import get_memory_alloc, log_ckpts, plot_attention_transformer
from utils.misc import plot_alignment, check_device, combine_weights
from modules.checkpoint import Checkpoint
from models.Seq2seq import Seq2seq

logging.basicConfig(level=logging.INFO)
import pdb
import string


def count_edits(input, prediction,remove_punct=False):
    '''
    Count number of edits
    '''
    prediction = prediction.replace('\n', '')
    input = input.replace('\n', '')
    if prediction[-2:] == ' .':
        prediction = prediction[:-2]+'.'
    if input[-2:] == ' .':
        input = input[:-2]+'.'
    # pdb.set_trace()
    exclude = set(string.punctuation)
    if remove_punct:
        # remove punctuation
        input = ''.join(ch for ch in input if ch not in exclude)
        prediction = ''.join(ch for ch in prediction if ch not in exclude)
    annotator = errant.load('en')
    input = annotator.parse(input)
    prediction = annotator.parse(prediction)
    alignment = annotator.align(input, prediction)
    edits = annotator.merge(alignment)
    return len(edits)

def return_edits(input, prediction):
    '''
    Get edits
    '''
    if prediction[-2:] == ' .':
        prediction = prediction[:-2]+'.'
    if input[-2:] == ' .':
        input = input[:-2]+'.'
    annotator = errant.load('en')
    input = annotator.parse(input)
    prediction = annotator.parse(prediction)
    alignment = annotator.align(input, prediction)
    edits = annotator.merge(alignment)
    return edits

def load_arguments(parser):
    parser.add_argument('--base_path', type=str, required=True, help='base path')
    parser.add_argument('--input_path', type=str, required=True, help='input path')
    parser.add_argument('--output_file', type=str, required=True, help='output file name')
    return parser

def main():

	# load config
    # 
    parser = argparse.ArgumentParser(description='Seq2seq Evaluation')
    parser = load_arguments(parser)
    args = vars(parser.parse_args())
    config = validate_config(args)
    fp1 = open(config['base_path'], "r")
    fp2 = open(config['input_path'], "r")
    total_count = 0
    total_avg_count = 0
    line = 0

    while True:
        try:
            line += 1
            x = next(fp1)
            y = next(fp2)
            # pdb.set_trace()
            edits = count_edits(x, y)
            total_count += edits
            total_avg_count += (edits/len(x))
            # if count_edits(x, y) > 0:
            #     pdb.set_trace()
            # print(line)
            print('Temp Count: {0}, Idx: {1}'.format(total_count,line))
        except StopIteration:
            break   
    print(total_count)
    print(total_avg_count)
    file = config['output_file']
    with open(file, 'a+') as f:
        f.write(config['input_path']+': ')
        f.write(str(total_count) +' ' +str(total_avg_count) +'\n')   #加\n换行显示

if __name__ == '__main__':
	main()

