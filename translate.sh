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

# ===================================================================================
model=models/v002/Gaussian_mul_1.0_0.1_256_2_002

# ckpt=2022_02_01_15_37_10
ckpt=combine

# ------ [Generation for reranker eval set] ----------
# CLC
fname=eval-clc-test
# fname=checkpoints
ftst=/home/alta/BLTSpeaking/exp-ytl28/projects/gec-pretrained/exp-t5-written/lib/gec-train-bpe-written/prep/test.src
# ftst=/home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.train16.inc
# ftst=/home/alta/BLTSpeaking/exp-ytl28/projects/lib/gec-tst-collate/test-clc-orig.src

max_tgt_len=100

# ------ decode --------
# MODE
# 1: save combiend ckpts
# 2. save model dict
# 3: translate - <sent>
# 4: translate with verbo - <sid> <score> <sent>
eval_mode=3
use_gpu='True'

batch_size=16
mode='beam-1'
# batch_size=1
# mode='beam-50'
# mode='combine'

# ----------------------- [noise] ---------------------------
noise=1 #2 is for using the noise
ntype=Gaussian #Gaussian, Bernoulli, Gaussian-adversarial
nway=mul
mean=1.0
weight=0.0
word_keep=1.0

# ----- [dir names] -----
loaddir=/home/alta/BLTSpeaking/exp-ytl28/projects/gec-pretrained/exp-t5-written/models/v001/checkpoints-combine/combine
echo 'MODE '$eval_mode

if [[ $eval_mode -eq 1 ]]
    then
    
    outdir=$model/$fname-"$mode"/combine/
    combine_path=$model/checkpoints/
    echo 'COMBINE: '$loaddir
else
    if [[ $ckpt == 'combine' ]] # [combined ckpt]
        then
        # outdir=$model/$fname-"$mode"/combine/${noise}_${ntype}_${nway}_${mean}_${weight}
        outdir=$model/$fname-"$mode"/combine
        # loaddir=$model/checkpoints-combine/combine
    else # [single ckpt]
        outdir=$model/$fname-"$mode"/$ckpt/${noise}_${ntype}_${nway}_${mean}_${weight}
        # loaddir=$model/checkpoints/$ckpt
    fi
    combine_path='None'
    echo 'LOAD: '$loaddir
fi


# loaddir=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v002/Gaussian_mul_1.0_1.5_256_2_002/checkpoints-combine/combine
# loaddir=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v002/Gaussian_mul_1.0_0.1_256_2_002/checkpoints-combine/combine
# noise=1
# mkdir $model/$fname-"$mode"
# outdir=$model/$fname-"$mode"/orig
#     # outdir=$model/$fname-"$mode"/combine_v3/${noise}_${ntype}_${nway}_${word_keep}
#     echo 'OUT: '$outdir
#     $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/translate.py \
#         --test_path_src $ftst \
#         --load $loaddir \
#         --test_path_out $outdir \
#         --max_tgt_len $max_tgt_len \
#         --batch_size $batch_size \
#         --mode $mode \
#         --use_gpu $use_gpu \
#         --eval_mode $eval_mode \
#         --combine_path $combine_path \
#         --noise $noise \

# noise=2
# for weight in $(seq 0.0 0.1 2.5)
# do
#     # outdir=$model/$fname-"$mode"/combine_v2/${noise}_${ntype}_${nway}_${mean}_${weight}
#     outdir=$model/$fname-"$mode"/${noise}_${ntype}_${nway}_${mean}_${weight}
#     # outdir=$model/$fname-"$mode"/combine_Gau_1.5/orig
#     # outdir=$model/$fname-"$mode"/combine_v3/${noise}_${ntype}_${nway}_${word_keep}
#     echo 'OUT: '$outdir
#     $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/translate.py \
#         --test_path_src $ftst \
#         --load $loaddir \
#         --test_path_out $outdir \
#         --max_tgt_len $max_tgt_len \
#         --batch_size $batch_size \
#         --mode $mode \
#         --use_gpu $use_gpu \
#         --eval_mode $eval_mode \
#         --combine_path $combine_path \
#         --noise $noise \
#         --ntype $ntype \
#         --nway $nway \
#         --mean $mean \
#         --weight $weight \
#         --word_keep $word_keep
# done

# for weight in $(seq 0.0 0.1 2.5)
# for word_keep in $(seq 0.1 0.1 1)
# for weight in 1.1
# do
#     # outdir=$model/$fname-"$mode"/combine_v2/${noise}_${ntype}_${nway}_${mean}_${weight}
#     # outdir=$model/$fname-"$mode"/combine_v3_gramformer_edie/${noise}_${ntype}_${nway}_${mean}_${weight}
#     # outdir=$model/$fname-"$mode"/combine_v3_gramformer/${noise}_${ntype}_${nway}_${word_keep}
#     outdir=$model/$fname-"$mode"/combine_v3_gramformer/${noise}_${ntype}_${nway}_${mean}_${weight}
#     # outdir=$model/$fname-"$mode"/combine_v3_gramformer/orig
#     echo 'OUT: '$outdir
#     $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/translate.py \
#         --test_path_src $ftst \
#         --test_path_out $outdir \
#         --max_tgt_len $max_tgt_len \
#         --batch_size $batch_size \
#         --mode $mode \
#         --use_gpu $use_gpu \
#         --eval_mode $eval_mode \
#         --combine_path $combine_path \
#         --noise $noise \
#         --ntype $ntype \
#         --nway $nway \
#         --mean $mean \
#         --weight $weight \
#         --word_keep $word_keep
# done


# ------- Combine ---------------
eval_mode=1
model=models/v005
fname=checkpoints
mode='combine'
for exp in volta_Gaussian-adversarial_add_0.0_1000_256_8 volta_Gaussian-adversarial_add_0.0_100_256_8 volta_Gaussian-adversarial-norm_add_0.0_100_256_8 volta_Gaussian-adversarial-norm_add_0.0_10_256_8 
do
combine_path=$model/$exp/checkpoints/
outdir=$model/$exp/checkpoints-combine/combine/
$PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/translate.py \
        --test_path_src $ftst \
        --test_path_out $outdir \
        --max_tgt_len $max_tgt_len \
        --batch_size $batch_size \
        --mode $mode \
        --use_gpu $use_gpu \
        --eval_mode $eval_mode \
        --combine_path $combine_path \
        --noise $noise \
        --ntype $ntype \
        --nway $nway \
        --mean $mean \
        --weight $weight \
        --word_keep $word_keep
done
# for exp in Gaussian-adversarial_mul_1.0_0.01_16_1_002
# do
# combine_path=$model/$exp/checkpoints/
# outdir=$model/$exp/checkpoints-combine/combine/
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/translate.py \
#         --test_path_src $ftst \
#         --test_path_out $outdir \
#         --max_tgt_len $max_tgt_len \
#         --batch_size $batch_size \
#         --mode $mode \
#         --use_gpu $use_gpu \
#         --eval_mode $eval_mode \
#         --combine_path $combine_path \
#         --noise $noise \
#         --ntype $ntype \
#         --nway $nway \
#         --mean $mean \
#         --weight $weight \
#         --word_keep $word_keep
# done

# for exp in Gaussian-adversarial_mul_1.0_1_16_1_002 Gaussian-adversarial_mul_1.0_0.001_16_1_002 Gaussian-adversarial_mul_1.0_0.1_16_1_002
# do
# combine_path=$model/$exp/checkpoints/
# outdir=$model/$exp/checkpoints-combine/combine/
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/translate.py \
#         --test_path_src $ftst \
#         --test_path_out $outdir \
#         --max_tgt_len $max_tgt_len \
#         --batch_size $batch_size \
#         --mode $mode \
#         --use_gpu $use_gpu \
#         --eval_mode $eval_mode \
#         --combine_path $combine_path \
#         --noise $noise \
#         --ntype $ntype \
#         --nway $nway \
#         --mean $mean \
#         --weight $weight \
#         --word_keep $word_keep
# done

# for exp in Adversarial_mul_1.0_0.001_16_1_002 Adversarial_mul_1.0_0.01_16_1_002 Adversarial_mul_1.0_1_16_1_002 
# do
# combine_path=$model/$exp/checkpoints/
# outdir=$model/$exp/checkpoints-combine/combine/
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/translate.py \
#         --test_path_src $ftst \
#         --test_path_out $outdir \
#         --max_tgt_len $max_tgt_len \
#         --batch_size $batch_size \
#         --mode $mode \
#         --use_gpu $use_gpu \
#         --eval_mode $eval_mode \
#         --combine_path $combine_path \
#         --noise $noise \
#         --ntype $ntype \
#         --nway $nway \
#         --mean $mean \
#         --weight $weight \
#         --word_keep $word_keep
# done


# ------- Dict ---------------
# eval_mode=2
# load_dir=models/v002/Gaussian_mul_1.0_1.5_256_2_002/checkpoints-combine/combine
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/translate.py \
#         --test_path_src $ftst \
#         --test_path_out $outdir \
#         --max_tgt_len $max_tgt_len \
#         --batch_size $batch_size \
#         --mode $mode \
#         --use_gpu $use_gpu \
#         --eval_mode $eval_mode \
#         --combine_path $combine_path \
#         --noise $noise \
#         --ntype $ntype \
#         --nway $nway \
#         --mean $mean \
#         --weight $weight \
#         --word_keep $word_keep \
#         --load $load_dir


# qsub -cwd -j yes -o 'LOGs/combine.log' -P esol -l hostname='*' -l qp=cuda-low -l gpuclass='*' -l osrel='*' translate.sh 1 1