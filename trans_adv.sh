#!/bin/bash
#$ -S /bin/bash

echo $HOSTNAME
unset LD_PRELOAD
# export PATH=/home/alta/BLTSpeaking/exp-ytl28/env/anaconda3/bin/:$PATH
echo export PATH=/home/alta/BLTSpeaking/exp-yw575/env/anaconda3/bin/:$PATH


export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
# export CUDA_VISIBLE_DEVICES=0
echo $CUDA_VISIBLE_DEVICES

# python 3.7
# pytorch 1.5
# source activate /home/alta/BLTSpeaking/exp-ytl28/env/anaconda3/envs/py38-pt15-cuda10
# export PYTHONBIN=/home/alta/BLTSpeaking/exp-ytl28/env/anaconda3/envs/py38-pt15-cuda10/bin/python3
source activate /home/alta/BLTSpeaking/exp-yw575/env/anaconda3/envs/gec37
export PYTHONBIN=/home/alta/BLTSpeaking/exp-yw575/env/anaconda3/envs/gec37/bin/python3

# ===================================================================================
# ------------------------ DIR --------------------------
orig_path=/home/alta/BLTSpeaking/exp-ytl28/projects/gec-pretrained/exp-t5-written
# train_path_src=$orig_path/lib/gec-train-bpe-written/prep/train.src
# train_path_tgt=$orig_path/lib/gec-train-bpe-written/prep/train.tgt
# dev_path_src=$orig_path/lib/gec-train-bpe-written/prep/dev.src
# dev_path_tgt=$orig_path/lib/gec-train-bpe-written/prep/dev.tgt

max_src_len=64
max_tgt_len=64

# ------------------------ TRAIN --------------------------
# [SCHEDULE 1]
# lr_init=0.00001
# lr_peak=0.0007
# lr_warmup_steps=4000

# [SCHEDULE 2] - slower warmup for tuned models
lr_init=0.00001
lr_peak=0.0005
lr_warmup_steps=4000

grab_memory='True'
random_seed=25

# [inactive when dev not given]
max_count_no_improve=30
max_count_num_rollback=0 # 0:no roll back no lr reduce
keep_num=5

# --------------
batch_size=256
minibatch_split=8 #8 for million
# minibatch_split=8 #8 for million
# minibatch_split=16 #8 for million
num_epochs=100

checkpoint_every=5000 # ~10k if 2M, batch - 256
print_every=1000

grab_memory='False'
loaddir=/home/alta/BLTSpeaking/exp-ytl28/projects/gec-pretrained/exp-t5-written/models/v001/checkpoints-combine/combine
savedir=models/temp/
load_mode='null' # 'resume' | 'restart' | 'null'

# ----------------------- [debug] ---------------------------
train_path_src=$orig_path/lib/gec-train-bpe-written/prep/test.src
train_path_tgt=$orig_path/lib/gec-train-bpe-written/prep/test.tgt
dev_path_src=$orig_path/lib/gec-train-bpe-written/prep/toy.src
dev_path_tgt=$orig_path/lib/gec-train-bpe-written/prep/toy.tgt
num_epochs=1
# minibatch_split=1
# batch_size=2
# checkpoint_every=10
# print_every=2

# ----------------------- [noise] ---------------------------
ntype=Gaussian-adversarial #Gaussian, Bernoulli, Gaussian-adversarial, Adversarial, Gaussian-adversarial-single, Adversarial-single
nway=add
mean=0.0
weight=0.1
savedir=models/v005/eval-clc-test-beam-1/temp/

# ===================================================================================
for ntype in Gaussian-adversarial
do
for weight in 0.0 0.1 1.0 10.0 100.0 1000.0
do
for alpha in 100000000
do
savedir=models/v005/eval-clc-test-beam-1/temp/
$PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/trans_adv.py \
	--train_path_src $train_path_src \
	--train_path_tgt $train_path_tgt \
	--dev_path_src $dev_path_src \
	--dev_path_tgt $dev_path_tgt \
	--max_src_len $max_src_len \
	--max_tgt_len $max_tgt_len \
	\
	--lr_peak $lr_peak \
	--lr_init $lr_init \
	--lr_warmup_steps $lr_warmup_steps \
	\
	--batch_size $batch_size \
	--minibatch_split $minibatch_split \
	--num_epochs $num_epochs \
	\
	--load $loaddir \
	--load_mode $load_mode \
	--save $savedir \
	\
	--random_seed $random_seed \
	--max_grad_norm 1.0 \
	--checkpoint_every $checkpoint_every \
	--print_every $print_every \
	--max_count_no_improve $max_count_no_improve \
	--max_count_num_rollback $max_count_num_rollback \
	--keep_num $keep_num \
	--grab_memory $grab_memory \
	--use_gpu True \
	--gpu_id $CUDA_VISIBLE_DEVICES \
	--ntype $ntype \
	--nway $nway \
	--mean $mean \
	--weight $weight
done
done

# for ntype in Gaussian-adversarial-single Adversarial-single
# do
# for weight in 0.0 0.001 0.005 0.01 0.05 0.1 0.5 1 1.5
# do
# savedir=models/v001/eval-clc-test-beam-1/adv_updated/${ntype}_${nway}_${mean}_${weight}_${batch_size}_${minibatch_split}_002/
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/trans_adv.py \
# 	--train_path_src $train_path_src \
# 	--train_path_tgt $train_path_tgt \
# 	--dev_path_src $dev_path_src \
# 	--dev_path_tgt $dev_path_tgt \
# 	--max_src_len $max_src_len \
# 	--max_tgt_len $max_tgt_len \
# 	\
# 	--lr_peak $lr_peak \
# 	--lr_init $lr_init \
# 	--lr_warmup_steps $lr_warmup_steps \
# 	\
# 	--batch_size $batch_size \
# 	--minibatch_split $minibatch_split \
# 	--num_epochs $num_epochs \
# 	\
# 	--load $loaddir \
# 	--load_mode $load_mode \
# 	--save $savedir \
# 	\
# 	--random_seed $random_seed \
# 	--max_grad_norm 1.0 \
# 	--checkpoint_every $checkpoint_every \
# 	--print_every $print_every \
# 	--max_count_no_improve $max_count_no_improve \
# 	--max_count_num_rollback $max_count_num_rollback \
# 	--keep_num $keep_num \
# 	--grab_memory $grab_memory \
# 	--use_gpu True \
# 	--gpu_id $CUDA_VISIBLE_DEVICES \
# 	--ntype $ntype \
# 	--nway $nway \
# 	--mean $mean \
# 	--weight $weight
# done
# done

# qsub -cwd -j yes -o 'LOGs/adv_trans_2.log' -P esol -l hostname='*' -l qp=cuda-low -l gpuclass='*' -l osrel='*' trans_adv.sh 1 1
# qsub -cwd -j yes -o 'LOGs/adv_single_2.log' -P esol -l hostname='*' -l qp=cuda-low -l gpuclass='*' -l osrel='*' trans_adv.sh 1 1
# qsub -cwd -j yes -o 'LOGs/adv_003_trans.log' -P esol -l hostname='*' -l qp=cuda-low -l gpuclass='*' -l osrel='*' trans_adv.sh 1 1
# qsub -cwd -j yes -o 'LOGs/adv_003_single.log' -P esol -l hostname='*' -l qp=cuda-low -l gpuclass='*' -l osrel='*' trans_adv.sh 1 1


# qsub -cwd -j yes -o 'LOGs/adv_004_add.log' -P esol -l hostname='*' -l qp=cuda-low -l gpuclass='*' -l osrel='*' trans_adv.sh 1 1

# qsub -cwd -j yes -o 'LOGs/v005/adv_005_add.log' -P esol -l hostname='*' -l qp=cuda-low -l gpuclass='*' -l osrel='*' trans_adv.sh 1 1