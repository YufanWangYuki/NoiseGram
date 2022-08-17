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
corr=/home/mifs/yw575/dataset/test_id.tgt
preddir=prediction_files
outdir=prediction_files/for_errant
seed=1

# ------ [Generation for reranker eval set] ----------
exp=GramGau
model=/home/mifs/yw575/models/Gaussian_mul_1.0_0.1_256_2_002/checkpoints-combine/
for exp in GramGau GramAdv GramMeanAdv_1 GramMeanAdv_2 GramMeanAdv_3
do
for checkpoint in combine
do 
    pred=$preddir/${exp}.pred
    output=$outdir/${exp}_${checkpoint}_seed_${seed}
    echo pred
    $PYTHONBIN /home/mifs/yw575/NoiseGram/utils/align_preds.py \
        --INC $input \
        --PRED $pred \
        --CORR $corr \
        --BASE $output \
        --seed $seed
        # --remove_punct yes
done
done

