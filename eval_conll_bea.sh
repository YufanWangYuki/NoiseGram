#!/bin/bash
#$ -S /bin/bash

echo $HOSTNAME
unset LD_PRELOAD
echo export PATH=/home/alta/BLTSpeaking/exp-yw575/env/anaconda3/bin/:$PATH

# export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
export CUDA_VISIBLE_DEVICES=0
echo $CUDA_VISIBLE_DEVICES

# python 3.7
# pytorch 1.5
source ~/.bashrc
conda activate gec37
export PYTHONBIN=/home/alta/BLTSpeaking/exp-yw575/env/anaconda3/envs/gec37/bin/python3
export PYTHONPATH="${PYTHONPATH}:/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/"


for exp in CoNLL_orig_combine_seed_1 CoNLL_volta_Gaussian_mul_1.0_0.0__256_8_combine_seed_1
do
    dir=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/prediction_files/orig
    errant_parallel -orig ${dir}/conlltest.inc -cor ${dir}/${exp}.pred -out ${dir}/${exp}.m2
    errant_compare -hyp ${dir}/${exp}.m2 -ref ${dir}/official-2014.combined.m2 >> results/Fscore/v003_conll.txt
done