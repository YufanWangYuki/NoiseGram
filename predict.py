'''
Generate model predictions

Input file:
ID1 Sentence1
ID2 Sentence2
.
.
.

Output file:
ID1 Sentence1
ID2 Sentence2
.
.
.

If an adversarial sequence is appended at the input, then two files are created:
adv_with_adv_not_removed.pred
adv_with_adv_removed.pred

i.e. the second file has the adversarial sequence removed from the output prediction sentences
'''

import sys
import os
import argparse
import torch
from utils.gec_tools import get_sentences, correct
from models.Seq2seq import Seq2seq
from utils.eval_uni_attack import set_seeds
from utils.uni_attack import concatenate
from modules.checkpoint import Checkpoint
import pdb

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--IN', type=str, help='Path to input data')
    commandLineParser.add_argument('--MODEL', type=str, help='Path to Gramformer model')
    commandLineParser.add_argument('--OUT_BASE', type=str, help='Path to corrected output data - pass only base name, e.g. beam1_N4 or no_attack')
    commandLineParser.add_argument('--use_attack', type=int, default=0, help='Attack or not, 0 for no, 1 for yes')
    commandLineParser.add_argument('--phrase', type=str, default='', help='Universal adversarial phrase')
    commandLineParser.add_argument('--seed', type=int, default=1, help='reproducibility')
    commandLineParser.add_argument('--delim', type=str, default='', help='concatenation delimiter')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/predict.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n') 
    
    device = torch.device('cpu')
    set_seeds(args.seed)

    # Load Model
    # model = Seq2seq()
    # model.load_state_dict(torch.load(args.MODEL, map_location=torch.device('cpu')))
    # model.eval()
    latest_checkpoint_path = args.MODEL
    resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
    model = resume_checkpoint.model.to(device)
    model.eval()

    # Load input sentences
    identifiers, sentences = get_sentences(args.IN)

    # Correction (prediction) for each input sentence
    corrections = []

    if args.use_attack == 1 and args.phrase != '':
        corrections_adv_removed = []

    for i, sent in enumerate(sentences):
        print(f'On {i}/{len(sentences)}')
        if args.use_attack == 1 and args.phrase != '':
            attack_phrase = args.phrase + '.'
            sent = concatenate(sent, attack_phrase, delim=args.delim)
            # pdb.set_trace()
        correction = correct(model, sent)
        corrections.append(correction)
        if args.use_attack == 1 and args.phrase != '':
            correction_adv_removed = correction.replace(f', {args.phrase}.', '.')
            corrections_adv_removed.append(correction_adv_removed)        
    assert len(corrections) == len(identifiers), "Number of ids don't match number of predictions"

    # Save predictions
    if args.use_attack == 1 and args.phrase != '':
        print(args.OUT_BASE)
        file2 = f'{args.OUT_BASE}_with_adv_removed.pred'
        file1 = f'{args.OUT_BASE}_with_adv_not_removed.pred'
        with open(file2, 'w') as f:
            for id, sentence in zip(identifiers, corrections_adv_removed):
                f.write(f'{id} {sentence}\n')
    else:
        file1 = f'{args.OUT_BASE}.pred'

    with open(file1, 'w') as f:
        for id, sentence in zip(identifiers, corrections):
            f.write(f'{id} {sentence}\n')
