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
source activate /home/alta/BLTSpeaking/exp-yw575/env/anaconda3/envs/gec37
export PYTHONBIN=/home/alta/BLTSpeaking/exp-yw575/env/anaconda3/envs/gec37/bin/python3

base_path=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v001/eval-clc-test-beam-1/combine/translate.txt
input_path=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v001/eval-clc-test-beam-1/combine/1_Gaussian_mul_1.0_0.1/translate.txt

$PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/count.py \
    --base_path $base_path \
    --input_path $input_path
