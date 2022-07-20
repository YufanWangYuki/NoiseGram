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

# exp=orig
# model=/home/alta/BLTSpeaking/exp-ytl28/projects/gec-pretrained/exp-t5-written/models/v001/checkpoints-combine/
# for checkpoint in combine
# do
#     output=$outdir/orig
#     mkdir output
#     $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model/$checkpoint \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 0
# done

# exp=volta_Gaussian_mul_1.0_0.0__256_8
# model=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/orig/volta_Gaussian_mul_1.0_0.0__256_8/checkpoints
# outdir=prediction_files/orig
# for checkpoint in 2022_07_07_21_19_25 2022_07_08_01_27_30 2022_07_08_17_37_21
# do
#     output=$outdir/${exp}_${checkpoint}_seed_${seed}
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

# checkpoint=combine
# outdir=prediction_files/v005
# mkdir $outdir
# for exp in volta_Gaussian-adversarial_mul_1.0_0.1_1_1_256_8
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

# checkpoint=combine
# for exp in volta_Gaussian_mul_1.0_0.0__256_8
# do
# # model=/home/alta/BLTSpeaking/exp-ytl28/projects/gec-pretrained/exp-t5-written/models/v001/checkpoints-combine
# model=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/orig/volta_Gaussian_mul_1.0_0.0__256_8/checkpoints-combine 
# outdir=prediction_files/orig
# input=/home/alta/BLTSpeaking/exp-vr313/GEC/data/CoNLL-14/conll14st-test-data/noalt/input_sentences.txt
# output=$outdir/CoNLL_${exp}_${checkpoint}_seed_${seed}
# # $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
# #         --IN $input \
# #         --MODEL $model/$checkpoint \
# #         --OUT_BASE $output \
# #         --seed $seed \
# #         --use_attack 0

# outdir=prediction_files/orig/attacks
# mkdir outdir       
# output=$outdir/${exp}_full_N5_CoNLL
# # $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
# #         --IN $input \
# #         --MODEL $model/$checkpoint \
# #         --OUT_BASE $output \
# #         --seed $seed \
# #         --use_attack 1 \
# #         --phrase 'ink l xml mv sub' \
# #         --delim '.' 

# output=$outdir/${exp}_perp_N5_CoNLL
# # $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
# #         --IN $input \
# #         --MODEL $model/$checkpoint \
# #         --OUT_BASE $output \
# #         --seed $seed \
# #         --use_attack 1 \
# #         --phrase 'trifecta haiku utah intransigent penicillin' \
# #         --delim '.' 

# outdir=prediction_files/orig
# input=/home/alta/BLTSpeaking/exp-vr313/GEC/data/BEA2019/bea2019test.txt
# output=$outdir/BEA_${exp}_${checkpoint}_seed_${seed}
# # $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
# #         --IN $input \
# #         --MODEL $model/$checkpoint \
# #         --OUT_BASE $output \
# #         --seed $seed \
# #         --use_attack 0

# outdir=prediction_files/orig/attacks
# mkdir outdir       
# output=$outdir/${exp}_full_N5_BEA
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model/$checkpoint \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 1 \
#         --phrase 'ink l xml mv sub' \
#         --delim '.' 

# output=$outdir/${exp}_perp_N5_BEA
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model/$checkpoint \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 1 \
#         --phrase 'trifecta haiku utah intransigent penicillin' \
#         --delim '.' 
# done

# # ------------------------------------Attack
# for exp in volta_Gaussian-adversarial_add_0.0_100_256_8
# do
# model=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v005/${exp}/checkpoints-combine/combine/
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

# ------------------------------------Perp Attack
for exp in volta_Gaussian-adversarial_mul_1.0_0.1_256_8
do
model=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v003/volta_Gaussian-adversarial_mul_1.0_0.1_256_8/checkpoints-combine/combine/
mkdir prediction_files/v003/$exp
mkdir prediction_files/v003/$exp/attacks_perp
outdir=prediction_files/v003/$exp/attacks_perp
# # output=$outdir/full_N1
# # $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
# #         --IN $input \
# #         --MODEL $model \
# #         --OUT_BASE $output \
# #         --seed $seed \
# #         --use_attack 1 \
# #         --phrase 'chutzpah' \
# #         --delim '.'

# output=$outdir/full_N2
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 1 \
#         --phrase 'chutzpah vb' \
#         --delim '.'

output=$outdir/full_N3
$PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
        --IN $input \
        --MODEL $model \
        --OUT_BASE $output \
        --seed $seed \
        --use_attack 1 \
        --phrase 'chutzpah vb ditka' \
        --delim '.'

# output=$outdir/full_N4
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 1 \
#         --phrase 'chutzpah ii bibb en' \
#         --delim '.'   

# output=$outdir/full_N5
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 1 \
#         --phrase 'chutzpah ii bibb en fyi' \
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
done

# ------------------------------------Perp Attack
# for exp in volta_Gaussian-adversarial_mul_1.0_0.1_256_8
# do
# model=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v003/volta_Gaussian-adversarial_mul_1.0_0.1_256_8/checkpoints-combine/combine/
# mkdir prediction_files/v003/$exp
# mkdir prediction_files/v003/$exp/attacks_perp_old
# outdir=prediction_files/v003/$exp/attacks_perp_old
# output=$outdir/full_N1
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 1 \
#         --phrase 'trifecta' \
#         --delim '.'

# output=$outdir/full_N2
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 1 \
#         --phrase 'trifecta haiku' \
#         --delim '.'
# output=$outdir/full_N3
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 1 \
#         --phrase 'trifecta haiku utah' \
#         --delim '.'

# output=$outdir/full_N4
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 1 \
#         --phrase 'trifecta haiku utah intransigent' \
#         --delim '.'   

# output=$outdir/full_N5
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 1 \
#         --phrase 'trifecta haiku utah intransigent penicillin' \
#         --delim '.'    

# output=$outdir/full_N6
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 1 \
#         --phrase 'trifecta haiku utah intransigent penicillin baseline' \
#         --delim '.'    

# output=$outdir/full_N7
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 1 \
#         --phrase 'trifecta haiku utah intransigent penicillin baseline exploratory' \
#         --delim '.'   

# output=$outdir/full_N8
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
#         --IN $input \
#         --MODEL $model \
#         --OUT_BASE $output \
#         --seed $seed \
#         --use_attack 1 \
#         --phrase 'trifecta haiku utah intransigent penicillin baseline exploratory bioengineering' \
#         --delim '.'   

# # output=$outdir/full_N9
# # $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/predict.py \
# #         --IN $input \
# #         --MODEL $model \
# #         --OUT_BASE $output \
# #         --seed $seed \
# #         --use_attack 1 \
# #         --phrase 'ink l xml mv sub xu bec l sub' \
# #         --delim '.'     
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

# qsub -cwd -j yes -o 'LOGs/pred_conll.log' -P esol -l hostname='*' -l qp=cuda-low -l gpuclass='*' -l osrel='*' predict.sh 1 1
# qsub -cwd -j yes -o 'LOGs/pred_perp.log' -P esol -l hostname='*' -l qp=cuda-low -l gpuclass='*' -l osrel='*' predict.sh 1 1
# qsub -cwd -j yes -o 'LOGs/pred_new_conll.log' -P esol -l hostname='*' -l qp=cuda-low -l gpuclass='*' -l osrel='*' predict.sh 1 1
# qsub -cwd -j yes -o 'LOGs/pred_new_bea.log' -P esol -l hostname='*' -l qp=cuda-low -l gpuclass='*' -l osrel='*' predict.sh 1 1