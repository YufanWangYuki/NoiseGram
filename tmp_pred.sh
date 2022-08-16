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
input=/home/mifs/yw575/dataset/test_id.src 
outdir=prediction_files
seed=1

exp=orig

for exp in GramGau GramAdv GramMeanAdv_1 GramMeanAdv_2 GramMeanAdv_3
do
for checkpoint in combine
do
    output=$outdir/$exp
    model=/home/mifs/yw575/models/$exp/checkpoints-combine/
    mkdir output
    $PYTHONBIN /home/mifs/yw575/NoiseGram/predict.py \
        --IN $input \
        --MODEL $model/$checkpoint \
        --OUT_BASE $output \
        --seed $seed \
        --use_attack 0
    
    output=$outdir/${exp}_perp_N5
    $PYTHONBIN /home/mifs/yw575/NoiseGram/predict.py \
        --IN $input \
        --MODEL $model/$checkpoint \
        --OUT_BASE $output \
        --seed $seed \
        --use_attack 1 \
        --phrase 'trifecta haiku utah intransigent penicillin' \
        --delim '.' 
done
done