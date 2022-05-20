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

# base_path=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v001/eval-clc-test-beam-1/combine_v2/2_Gaussian_mul_1.0_0.0/translate.txt
base_path=/home/alta/BLTSpeaking/exp-ytl28/projects/gec-pretrained/exp-t5-written/lib/gec-train-bpe-written/prep/test.src
output_file=count_orig.txt

# for word_keep in $(seq 0.1 0.1 1)
# do
input_path=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v001/eval-clc-test-beam-1/combine_v3/orig/translate.txt
echo $input_path
$PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/count.py \
    --base_path $base_path \
    --input_path $input_path \
    --output_file $output_file
# done
