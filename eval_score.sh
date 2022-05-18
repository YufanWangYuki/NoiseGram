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
preddir=prediction_files/for_errant
outdir=prediction_files/m2
seed=1

exp=Adversarial_mul_1.0_0.1_2_001
for checkpoint in 2022_05_17_01_52_19
do 
    # Adversarial_mul_1.0_0.1_2_001_2022_05_17_01_52_19_seed_1.pred
    input=$preddir/${exp}_${checkpoint}_seed_${seed}.inc
    pred=$preddir/${exp}_${checkpoint}_seed_${seed}.pred
    corr=$preddir/${exp}_${checkpoint}_seed_${seed}.corr
    errant_parallel -orig $input -cor $pred -out $outdir/${exp}_${checkpoint}_edits-pred.m2
    errant_parallel -orig $input -cor $corr -out $outdir/${exp}_${checkpoint}_edits-corr.m2
done
echo ${exp}_${checkpoint} >> results/Fscore.txt
errant_compare -hyp $outdir/${exp}_${checkpoint}_edits-pred.m2 -ref $outdir/${exp}_${checkpoint}_edits-corr.m2 >> results/Fscore.txt

# errant_compare -hyp prediction_files/m2/Adversarial_mul_1.0_0.1_2_001_2022_05_17_01_52_19_edits-pred.m2 -ref prediction_files/m2/Adversarial_mul_1.0_0.1_2_001_2022_05_17_01_52_19_edits-corr.m2



