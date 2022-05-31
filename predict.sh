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
input=/home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc 
outdir=prediction_files/v002/
seed=1

# ------ [Generation for reranker eval set] ----------
exp=Gaussian_mul_1.0_1.5_256_2_002
model=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v002/$exp/checkpoints
for checkpoint in 2022_05_27_14_07_55  2022_05_28_02_45_55  2022_05_28_06_48_16  2022_05_29_00_44_45  2022_05_29_22_26_15
do 
    output=$outdir/${exp}_${checkpoint}_seed_${seed}
    $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
        --IN $input \
        --MODEL $model/$checkpoint \
        --OUT_BASE $output \
        --seed $seed \
        --use_attack 0
done

model=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v002/$exp/checkpoints-combine
checkpoint=combine
output=$outdir/${exp}_${checkpoint}_seed_${seed}
$PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
        --IN $input \
        --MODEL $model/$checkpoint \
        --OUT_BASE $output \
        --seed $seed \
        --use_attack 0