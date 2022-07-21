import argparse
import sys
import os
import scandir
import pdb

words_dir = "/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/LOGs/best_words/"
output_file = words_dir+"perp_vocab.txt"
# Get list of files in directory
files = [f.name for f in scandir.scandir(words_dir)]

words = []
for file in files:
    if 'vyas' not in file:
        continue
    print("Processing " + file)
    curr_path = words_dir+"/"+file
    with open(curr_path, 'r') as f:
        lines = f.readlines()
    for line in lines[2:]:
        word = line.strip().split(":")[0]
        words.append(word)
    
    print(len(word))

set_word = set(words)
print(len(set_word))
pdb.set_trace()
with open(output_file, 'w+') as f:
    for item in set_word:
        f.write('\n'+item)


