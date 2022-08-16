#!/bin/bash
#$ -S /bin/bash

echo $HOSTNAME
unset LD_PRELOAD
echo export PATH=/home/mifs/yw575/env/anaconda3/envs/gec37/bin:$PATH

# export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
export CUDA_VISIBLE_DEVICES=0
echo $CUDA_VISIBLE_DEVICES

# python 3.7
# pytorch 1.5
source activate /home/mifs/yw575/env/anaconda3/envs/gec37
export PYTHONBIN=/home/mifs/yw575/env/anaconda3/envs/gec37/bin/python3
export PYTHONPATH="${PYTHONPATH}:/home/mifs/yw575/NoiseGram/"

# ===================================================================================
input=/home/mifs/yw575/dataset/test-fce.src 
outdir=prediction_files
seed=1

exp=orig
model=/home/mifs/yw575/models/Gaussian_mul_1.0_0.1_256_2_002/checkpoints-combine/
for checkpoint in combine
do
    output=$outdir/GramGau
    mkdir output
    $PYTHONBIN /home/mifs/yw575/NoiseGram/predict.py \
        --IN $input \
        --MODEL $model/$checkpoint \
        --OUT_BASE $output \
        --seed $seed \
        --use_attack 0
done