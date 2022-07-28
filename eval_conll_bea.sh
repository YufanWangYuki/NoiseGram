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
conda activate gec37
export PYTHONBIN=/home/alta/BLTSpeaking/exp-yw575/env/anaconda3/envs/gec37/bin/python3
export PYTHONPATH="${PYTHONPATH}:/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/"


for exp in v005
do
    dir=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/prediction_files/orig
    pred=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/prediction_files/for_errant/conll/$exp
    errant_parallel -orig ${dir}/conlltest.inc -cor ${pred}/clean_CoNLL_GramAdvMean_mul_0.1.pred -out ${pred}/clean_${exp}_GramAdvMean_mul_0.1.m2
    echo clean_${exp}_GramAdvMean_mul_0.1 >> results/Fscore/v002_conll.txt
    errant_compare -hyp ${pred}/clean_${exp}_GramAdvMean_mul_0.1.m2 -ref ${dir}/official-2014.combined.m2 >> results/Fscore/v002_conll.txt
done

# errant_compare -hyp edits-pred.m2 -ref /home/alta/BLTSpeaking/exp-vr313/GEC/data/CoNLL-14/conll14st-test-data/noalt/official-2014.combined.m2

# for exp in CoNLL_orig CoNLL_volta_Gaussian_mul_1.0_0.0__256_8
# do
#     checkpoint=combine
#     dir=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/prediction_files/for_errant/orig/$exp
#     input=$dir/${checkpoint}_seed_1.inc
#     pred=$dir/${checkpoint}_seed_1.pred
#     corr=$dir/${checkpoint}_seed_1.corr
#     outdir=prediction_files/m2/orig

#     errant_parallel -orig $input -cor $pred -out $outdir/${exp}_${checkpoint}_edits-pred.m2
#     errant_parallel -orig $input -cor $corr -out $outdir/${exp}_${checkpoint}_edits-corr.m2
#     echo ${exp}_${checkpoint} >> results/Fscore/v003_conll.txt
#     errant_compare -hyp $outdir/${exp}_${checkpoint}_edits-pred.m2 -ref $outdir/${exp}_${checkpoint}_edits-corr.m2 >> results/Fscore/v003_conll.txt
#     echo ${exp}_${checkpoint}
# done