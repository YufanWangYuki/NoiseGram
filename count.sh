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
base_path=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v001/eval-clc-test-beam-1/combine_v3/orig/translate.txt
# base_path=/home/alta/BLTSpeaking/exp-ytl28/projects/gec-pretrained/exp-pretrained/models/gramformer/eval_clc-orig/beam1

# base_path=/home/alta/BLTSpeaking/exp-ytl28/projects/gec-pretrained/exp-t5-written/lib/gec-train-bpe-written/prep/test.src
# base_path=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v001/eval-clc-test-beam-1/combine_v3_gramformer/orig/translate.txt
# base_path=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v002/Gaussian_mul_1.0_1.5_256_2_002/eval-clc-test-beam-1/combine_Gau_1.5/orig/translate.txt
output_file=results/adv/count_adv_trans.txt

# for ntype in Gaussian-adversarial Adversarial
# do
for ntype in Gaussian-adversarial Adversarial
do
for weight in 0.0 0.001 0.005 0.01 0.05 0.1 0.5 1 1.5
do
# input_path=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v001/eval-clc-test-beam-1/combine_v3/orig/translate.txt
# input_path=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v001/eval-clc-test-beam-1/combine_v3/2_Gaussian_mul_1.0_${weight}/translate.txt
# input_path=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v001/eval-clc-test-beam-1/combine_v3/2_Bernoulli_mul_${word_keep}/translate.txt
# input_path=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v001/eval-clc-test-beam-1/combine_v3/2_Gaussian_add_0.0_${weight}/translate.txt
# input_path=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v001/eval-clc-test-beam-1/combine_v3_gramformer_edie/2_Gaussian_mul_1.0_${weight}/translate.txt
# input_path=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/gramformer/eval_clc-orig/beam1
# input_path=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v001/eval-clc-test-beam-1/combine_v3_gramformer/2_Gaussian_mul_1.0_$weight/translate.txt
# input_path=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v001/eval-clc-test-beam-1/combine_v3_gramformer/2_Gaussian_add_0.0_$weight/translate.txt

# input_path=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v001/eval-clc-test-beam-1/combine_v3_gramformer/2_Bernoulli_mul_${word_keep}/translate.txt
# input_path=/home/alta/BLTSpeaking/exp-ytl28/projects/gec-pretrained/exp-t5-written/lib/gec-train-bpe-written/prep/test.tgt
# input_path=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v001/eval-clc-test-beam-1/combine_v3_gramformer/2_${ntype}_mul_${weight}/translate.txt
# input_path=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v002/Gaussian_mul_1.0_1.5_256_2_002/eval-clc-test-beam-1/combine_Gau_1.5/2_Gaussian_mul_1.0_$weight/translate.txt
input_path=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v001/eval-clc-test-beam-1/combine_v3/2_${ntype}_mul_${weight}/translate.txt
echo $base_path >> $output_file
echo $input_path >> $output_file
echo $input_path
$PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/count.py \
    --base_path $base_path \
    --input_path $input_path \
    --output_file $output_file
done
done

# qsub -cwd -j yes -o 'LOGs/count/count_single.log' -P esol -l hostname='*' -l qp=cuda-low -l gpuclass='*' -l osrel='*' count.sh 1 1
