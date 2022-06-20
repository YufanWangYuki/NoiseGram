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
# exp=Gaussian_mul_1.0_1.5_256_2_002
# model=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v002/$exp/checkpoints
# for checkpoint in 2022_05_27_14_07_55  2022_05_28_02_45_55  2022_05_28_06_48_16  2022_05_29_00_44_45  2022_05_29_22_26_15
# do 
#     output=$outdir/${exp}_${checkpoint}_seed_${seed}
#     $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model/$checkpoint \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 0
# done

exp=orig
model=/home/alta/BLTSpeaking/exp-ytl28/projects/gec-pretrained/exp-t5-written/models/v001/checkpoints-combine/
for checkpoint in combine
do
    output=$outdir/orig
    mkdir output
    $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
        --IN $input \
        --MODEL $model/$checkpoint \
        --OUT_BASE $output \
        --seed $seed \
        --use_attack 0
done

# exp=Gaussian_mul_1.0_0.1_256_2_002
# model=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v002/$exp/checkpoints
# for checkpoint in 2022_06_04_13_20_01 2022_06_04_17_24_51 2022_06_05_01_43_16 2022_06_05_05_52_26 2022_06_05_10_02_02
# do
#     output=$outdir/v002/${exp}_${checkpoint}_seed_${seed}
#     $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model/$checkpoint \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 0
# done

# model=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v002/$exp/checkpoints-combine
# checkpoint=combine
# output=$outdir/${exp}_${checkpoint}_seed_${seed}
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model/$checkpoint \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 0

# exp=volta_Gaussian-adversarial_mul_1.0_0.1_256_8
# model=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v003/${exp}/checkpoints-combine
# checkpoint=combine
# outdir=prediction_files/v003_volta_Gaussian-adversarial_mul_1.0_0.1_256_8
# output=$outdir/${checkpoint}_seed_${seed}
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model/$checkpoint \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 0


# exp=volta_Gaussian-adversarial_mul_1.0_0.1_256_8
# checkpoint=combine
# outdir=prediction_files/v005
# mkdir $outdir
# for exp in volta_Gaussian-adversarial_add_0.0_1000_256_8 volta_Gaussian-adversarial_add_0.0_100_256_8 volta_Gaussian-adversarial-norm_add_0.0_100_256_8 volta_Gaussian-adversarial-norm_add_0.0_10_256_8 
# do
# model=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v005/${exp}/checkpoints-combine
# output=$outdir/${exp}_${checkpoint}_seed_${seed}
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model/$checkpoint \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 0
# done

# ------------------------------------Attack
# for exp in orig
# do
# model=/home/alta/BLTSpeaking/exp-ytl28/projects/gec-pretrained/exp-t5-written/models/v001/checkpoints-combine/combine/
# mkdir prediction_files/v005/$exp
# mkdir prediction_files/v005/$exp/attacks
# outdir=prediction_files/v005/$exp/attacks
# output=$outdir/full_N1
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 1 \
#         --phrase 'ink' \
#         --delim '.'

# output=$outdir/full_N2
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 1 \
#         --phrase 'ink l' \
#         --delim '.'
# output=$outdir/full_N3
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 1 \
#         --phrase 'ink l xml' \
#         --delim '.'

# output=$outdir/full_N4
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 1 \
#         --phrase 'ink l xml mv' \
#         --delim '.'   

# output=$outdir/full_N5
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 1 \
#         --phrase 'ink l xml mv sub' \
#         --delim '.'    

# output=$outdir/full_N6
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 1 \
#         --phrase 'ink l xml mv sub xu' \
#         --delim '.'    

# output=$outdir/full_N7
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 1 \
#         --phrase 'ink l xml mv sub xu bec' \
#         --delim '.'   

# output=$outdir/full_N8
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 1 \
#         --phrase 'ink l xml mv sub xu bec l' \
#         --delim '.'   

# output=$outdir/full_N9
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 1 \
#         --phrase 'ink l xml mv sub xu bec l sub' \
#         --delim '.'     
# done



# output=prediction_files/$exp/comma_N1
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 1 \
#         --phrase 'xl' \
#         --delim ','

# output=prediction_files/$exp/comma_N2
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 1 \
#         --phrase 'xl ii' \
#         --delim ','

# output=prediction_files/$exp/comma_N3
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 1 \
#         --phrase 'xl ii xl' \
#         --delim ','

# output=prediction_files/$exp/comma_N4
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 1 \
#         --phrase 'xl ii xl lbs' \
#         --delim ','

# output=prediction_files/$exp/comma_N5
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 1 \
#         --phrase 'xl ii xl lbs sub' \
#         --delim ','

# output=prediction_files/$exp/comma_N6
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 1 \
#         --phrase 'xl ii xl lbs sub xl' \
#         --delim ','

# output=prediction_files/$exp/comma_N7
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 1 \
#         --phrase 'xl ii xl lbs sub xl dp' \
#         --delim ','

# output=prediction_files/$exp/comma_N8
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 1 \
#         --phrase 'xl ii xl lbs sub xl dp lbs' \
#         --delim ','

# output=prediction_files/$exp/comma_N9
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 1 \
#         --phrase 'xl ii xl lbs sub xl dp lbs lc' \
#         --delim ','

# model=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v002/adv_fine
# checkpoint=combine

# for exp in Adversarial_mul_1.0_0.001_16_1_002
# do
#     loadir=$model/$exp/checkpoints-combine
#     output=prediction_files/adv_fine/$exp
#     $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $loadir/$checkpoint \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 0
# done

# for exp in Adversarial_mul_1.0_1_16_1_002 Gaussian-adversarial_mul_1.0_0.01_16_1_002 Gaussian-adversarial_mul_1.0_1_16_1_002 Adversarial_mul_1.0_0.01_16_1_002 Gaussian-adversarial_mul_1.0_0.001_16_1_002 Gaussian-adversarial_mul_1.0_0.1_16_1_002
# do
#     loadir=$model/$exp/checkpoints-combine
#     output=prediction_files/adv_fine/$exp
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $loadir/$checkpoint \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 0
# done