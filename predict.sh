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
export PYTHONPATH="${PYTHONPATH}:/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/"

# ===================================================================================


# ------ [Generation for reranker eval set] ----------
# FCE
exp=Adversarial_mul_1.0_0.1_2_001
model=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/$exp/checkpoints

input=/home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc 
outdir=prediction_files/
seed=1

for checkpoint in 2022_05_17_01_52_19 2022_05_17_02_32_19 2022_05_17_03_12_32 2022_05_17_04_33_38
do 
    output=$outdir/${exp}_${checkpoint}_seed_${seed}
    $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
        IN $input \
        MODEL $model/$checkpoint/model.pt \
        OUT_BASE $output \
        --seed $seed \
        --use_attack 0
done
