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
preddir=prediction_files/for_errant
outdir=prediction_files/m2
seed=1

exp=GramGau
for checkpoint in combine
do 
    input=$preddir/${exp}_${checkpoint}_seed_${seed}.inc
    pred=$preddir/${exp}_${checkpoint}_seed_${seed}.pred
    corr=$preddir/${exp}_${checkpoint}_seed_${seed}.corr
    errant_parallel -orig $input -cor $pred -out $outdir/${exp}_${checkpoint}_edits-pred.m2
    errant_parallel -orig $input -cor $corr -out $outdir/${exp}_${checkpoint}_edits-corr.m2

    echo ${exp}_${checkpoint} >> results/Fscore.txt
    errant_compare -hyp $outdir/${exp}_${checkpoint}_edits-pred.m2 -ref $outdir/${exp}_${checkpoint}_edits-corr.m2 >> results/Fscore.txt
    echo ${exp}_${checkpoint}
done


