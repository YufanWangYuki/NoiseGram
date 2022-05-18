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

# ===================================================================================
corr=/home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr
input=/home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc 
preddir=prediction_files/for_errant
seed=1

exp=Adversarial_mul_1.0_0.1_2_001
for checkpoint in 2022_05_17_01_52_19
do 
    # Adversarial_mul_1.0_0.1_2_001_2022_05_17_01_52_19_seed_1.pred
    pred=$preddir/${exp}_${checkpoint}_seed_${seed}.pred
    errant_parallel -orig $input -cor $pred -out ${exp}_${checkpoint}_edits-pred.m2
    errant_parallel -orig $input -cor $corr -out ${exp}_${checkpoint}_edits-corr.m2
done



