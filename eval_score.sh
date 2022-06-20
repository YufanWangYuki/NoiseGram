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
# preddir=prediction_files/for_errant
outdir=prediction_files/m2
seed=1

# exp=Adversarial_mul_1.0_0.1_2_001
# for checkpoint in 2022_05_17_01_52_19
# do 
#     # Adversarial_mul_1.0_0.1_2_001_2022_05_17_01_52_19_seed_1.pred
#     input=$preddir/${exp}_${checkpoint}_seed_${seed}.inc
#     pred=$preddir/${exp}_${checkpoint}_seed_${seed}.pred
#     corr=$preddir/${exp}_${checkpoint}_seed_${seed}.corr
#     errant_parallel -orig $input -cor $pred -out $outdir/${exp}_${checkpoint}_edits-pred.m2
#     errant_parallel -orig $input -cor $corr -out $outdir/${exp}_${checkpoint}_edits-corr.m2

#     echo ${exp}_${checkpoint} >> results/Fscore.txt
#     errant_compare -hyp $outdir/${exp}_${checkpoint}_edits-pred.m2 -ref $outdir/${exp}_${checkpoint}_edits-corr.m2 >> results/Fscore.txt
#     echo ${exp}_${checkpoint}
# done


# for checkpoint in 2022_05_17_02_32_19 2022_05_17_03_12_32 2022_05_17_04_33_38
# do 
#     input=$preddir/${exp}_${checkpoint}_seed_${seed}.inc
#     pred=$preddir/${exp}_${checkpoint}_seed_${seed}.pred
#     corr=$preddir/${exp}_${checkpoint}_seed_${seed}.corr
#     errant_parallel -orig $input -cor $pred -out $outdir/${exp}_${checkpoint}_edits-pred.m2
#     errant_parallel -orig $input -cor $corr -out $outdir/${exp}_${checkpoint}_edits-corr.m2

#     echo ${exp}_${checkpoint} >> results/Fscore.txt
#     errant_compare -hyp $outdir/${exp}_${checkpoint}_edits-pred.m2 -ref $outdir/${exp}_${checkpoint}_edits-corr.m2 >> results/Fscore.txt
#     echo ${exp}_${checkpoint}
# done

# exp=Gaussian-adversarial_mul_1.0_0.1_2_001
# model=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/$exp/checkpoints
# for checkpoint in 2022_05_18_04_56_12 2022_05_18_05_22_07 2022_05_18_07_55_14 2022_05_18_08_08_46 2022_05_18_08_34_54
# do 
#     input=$preddir/${exp}_${checkpoint}_seed_${seed}.inc
#     pred=$preddir/${exp}_${checkpoint}_seed_${seed}.pred
#     corr=$preddir/${exp}_${checkpoint}_seed_${seed}.corr
#     errant_parallel -orig $input -cor $pred -out $outdir/${exp}_${checkpoint}_edits-pred.m2
#     errant_parallel -orig $input -cor $corr -out $outdir/${exp}_${checkpoint}_edits-corr.m2

#     echo ${exp}_${checkpoint} >> results/Fscore.txt
#     errant_compare -hyp $outdir/${exp}_${checkpoint}_edits-pred.m2 -ref $outdir/${exp}_${checkpoint}_edits-corr.m2 >> results/Fscore.txt
#     echo ${exp}_${checkpoint}
# done

# exp=Gaussian_mul_1.0_1.8_2_001
# model=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/$exp/checkpoints
# for checkpoint in 2022_05_18_01_41_19  2022_05_18_01_58_04  2022_05_18_02_14_48  2022_05_18_02_31_56  2022_05_18_04_11_27
# do 
#     input=$preddir/${exp}_${checkpoint}_seed_${seed}.inc
#     pred=$preddir/${exp}_${checkpoint}_seed_${seed}.pred
#     corr=$preddir/${exp}_${checkpoint}_seed_${seed}.corr
#     errant_parallel -orig $input -cor $pred -out $outdir/${exp}_${checkpoint}_edits-pred.m2
#     errant_parallel -orig $input -cor $corr -out $outdir/${exp}_${checkpoint}_edits-corr.m2

#     echo ${exp}_${checkpoint} >> results/Fscore.txt
#     errant_compare -hyp $outdir/${exp}_${checkpoint}_edits-pred.m2 -ref $outdir/${exp}_${checkpoint}_edits-corr.m2 >> results/Fscore.txt
#     echo ${exp}_${checkpoint}
# done

# exp=Gaussian_mul_1.0_1.5_256_2_002
# preddir=prediction_files/for_errant/$exp
# outdir=prediction_files/m2/$exp
# for checkpoint in combine
# do 
#     input=$preddir/${checkpoint}_seed_${seed}.inc
#     pred=$preddir/${checkpoint}_seed_${seed}.pred
#     corr=$preddir/${checkpoint}_seed_${seed}.corr
#     errant_parallel -orig $input -cor $pred -out $outdir/${exp}_${checkpoint}_edits-pred.m2
#     errant_parallel -orig $input -cor $corr -out $outdir/${exp}_${checkpoint}_edits-corr.m2

#     echo ${exp}_${checkpoint} >> results/Fscore/${exp}_Fscore.txt
#     errant_compare -hyp $outdir/${exp}_${checkpoint}_edits-pred.m2 -ref $outdir/${exp}_${checkpoint}_edits-corr.m2 >> results/Fscore/${exp}_Fscore.txt
#     echo ${exp}_${checkpoint}
# done
# for checkpoint in 2022_05_27_14_07_55 2022_05_28_02_45_55 2022_05_28_06_48_16 2022_05_29_00_44_45 2022_05_29_22_26_15
# do 
#     input=$preddir/${checkpoint}_seed_${seed}.inc
#     pred=$preddir/${checkpoint}_seed_${seed}.pred
#     corr=$preddir/${checkpoint}_seed_${seed}.corr
#     errant_parallel -orig $input -cor $pred -out $outdir/${exp}_${checkpoint}_edits-pred.m2
#     errant_parallel -orig $input -cor $corr -out $outdir/${exp}_${checkpoint}_edits-corr.m2

#     echo ${exp}_${checkpoint} >> results/Fscore/${exp}_Fscore.txt
#     errant_compare -hyp $outdir/${exp}_${checkpoint}_edits-pred.m2 -ref $outdir/${exp}_${checkpoint}_edits-corr.m2 >> results/Fscore/${exp}_Fscore.txt
#     echo ${exp}_${checkpoint}
# done


# for exp in Gaussian-adversarial_mul_1.0_0.01_16_1_002 Gaussian-adversarial_mul_1.0_1_16_1_002 Gaussian-adversarial_mul_1.0_0.001_16_1_002 Gaussian-adversarial_mul_1.0_0.1_16_1_002
# do
#     preddir=prediction_files/for_errant/${exp}
#     checkpoint=combine
#     outdir=prediction_files/m2/adv_fine
#     mkdir $outdir
#     input=$preddir/${checkpoint}_seed_${seed}.inc
#     pred=$preddir/${checkpoint}_seed_${seed}.pred
#     corr=$preddir/${checkpoint}_seed_${seed}.corr
#     errant_parallel -orig $input -cor $pred -out $outdir/${exp}_${checkpoint}_edits-pred.m2
#     errant_parallel -orig $input -cor $corr -out $outdir/${exp}_${checkpoint}_edits-corr.m2

#     echo ${exp}_${checkpoint} >> results/Fscore/adv_fine_Fscore.txt
#     errant_compare -hyp $outdir/${exp}_${checkpoint}_edits-pred.m2 -ref $outdir/${exp}_${checkpoint}_edits-corr.m2 >> results/Fscore/adv_fine_Fscore.txt
#     echo ${exp}_${checkpoint}
# done

# for exp in Gaussian_mul_1.0_0.1_256_2_002
# do
# outdir=prediction_files/m2/${exp}
# mkdir $outdir
# for checkpoint in combine 2022_06_04_13_20_01 2022_06_04_17_24_51 2022_06_05_01_43_16 2022_06_05_05_52_26 2022_06_05_10_02_02
# do
#     preddir=prediction_files/for_errant/${exp}    
#     input=$preddir/${checkpoint}_seed_${seed}.inc
#     pred=$preddir/${checkpoint}_seed_${seed}.pred
#     corr=$preddir/${checkpoint}_seed_${seed}.corr
#     errant_parallel -orig $input -cor $pred -out $outdir/${exp}_${checkpoint}_edits-pred.m2
#     errant_parallel -orig $input -cor $corr -out $outdir/${exp}_${checkpoint}_edits-corr.m2

#     echo ${exp}_${checkpoint} >> results/Fscore/${exp}_Fscore.txt
#     errant_compare -hyp $outdir/${exp}_${checkpoint}_edits-pred.m2 -ref $outdir/${exp}_${checkpoint}_edits-corr.m2 >> results/Fscore/${exp}_Fscore.txt
#     echo ${exp}_${checkpoint}
# done
# done

outdir=prediction_files/m2/v005
mkdir prediction_files/m2/v005
for exp in orig
do
checkpoint=combine
dir=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/prediction_files/for_errant/v005/$exp
input=$dir/combine_seed_1.inc
pred=$dir/combine_seed_1.pred
corr=$dir/combine_seed_1.corr
errant_parallel -orig $input -cor $pred -out $outdir/${exp}_combine_edits-pred.m2
errant_parallel -orig $input -cor $corr -out $outdir/${exp}_combine_edits-corr.m2
    echo ${exp}_${checkpoint} >> results/Fscore/v005_Fscore.txt
    errant_compare -hyp $outdir/${exp}_${checkpoint}_edits-pred.m2 -ref $outdir/${exp}_${checkpoint}_edits-corr.m2 >> results/Fscore/v005_Fscore.txt
    echo ${exp}_${checkpoint}
done

