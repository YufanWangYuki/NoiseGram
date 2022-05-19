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
outdir=prediction_files
seed=1

# ------ [Generation for reranker eval set] ----------
exp=Adversarial_mul_1.0_0.1_2_001
model=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/$exp/checkpoints
# for checkpoint in 2022_05_17_01_52_19
# do 
#     output=$outdir/${exp}_${checkpoint}_seed_${seed}
#     $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model/$checkpoint \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 0
# done

# for checkpoint in 2022_05_17_02_32_19 2022_05_17_03_12_32 2022_05_17_04_33_38
# do 
#     output=$outdir/${exp}_${checkpoint}_seed_${seed}
#     $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model/$checkpoint \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 0
# done

# exp=Gaussian-adversarial_mul_1.0_0.1_2_001
# model=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/$exp/checkpoints
# for checkpoint in 2022_05_18_04_56_12 2022_05_18_05_22_07 2022_05_18_07_55_14 2022_05_18_08_08_46 2022_05_18_08_34_54
# do 
#     output=$outdir/${exp}_${checkpoint}_seed_${seed}
#     $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model/$checkpoint \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 0
# done

# exp=Gaussian_mul_1.0_1.8_2_001
# model=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/$exp/checkpoints
# for checkpoint in 2022_05_18_01_41_19  2022_05_18_01_58_04  2022_05_18_02_14_48  2022_05_18_02_31_56  2022_05_18_04_11_27
# do 
#     output=$outdir/${exp}_${checkpoint}_seed_${seed}
#     $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model/$checkpoint \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 0
# done
exp=Bernoulli_mul_1.0_0.7_2_001
model=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/$exp/checkpoints
# for checkpoint in 2022_05_19_02_14_40 2022_05_19_02_48_10
# do 
#     output=$outdir/${exp}_${checkpoint}_seed_${seed}
#     $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model/$checkpoint \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 0
# done

for checkpoint in 2022_05_19_04_59_18  2022_05_19_05_33_33  2022_05_19_07_14_27
    output=$outdir/${exp}_${checkpoint}_seed_${seed}
    $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
        --IN $input \
        --MODEL $model/$checkpoint \
        --OUT_BASE $output \
        --seed $seed \
        --use_attack 0
done