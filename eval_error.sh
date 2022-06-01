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
source activate /home/alta/BLTSpeaking/exp-yw575/env/anaconda3/envs/gec37
export PYTHONBIN=/home/alta/BLTSpeaking/exp-yw575/env/anaconda3/envs/gec37/bin/python3
export PYTHONPATH="${PYTHONPATH}:/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/"

# for name in comma_N1 comma_N2 comma_N3 comma_N4 comma_N5 comma_N6 comma_N7 comma_N8 comma_N9
# do
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED prediction_files/Gaussian_mul_1.0_1.5_256_2_002/orig/Gaussian_mul_1.0_1.5_256_2_002_combine_seed_1.pred \
#     --OUT edit_dist_files/Gaussian_mul_1.0_1.5_256_2_002/orig.txt


$PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
    --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
    --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
    --PRED prediction_files/Gaussian_mul_1.0_1.5_256_2_002/attacks/comma_N1_with_adv_not_removed.pred \
    --OUT edit_dist_files/Gaussian_mul_1.0_1.5_256_2_002/attacks/comma_N1_with_adv_not_removed.txt \
    --phrase 'xl' \
    --delim ','
# done