#!/bin/bash
#$ -S /bin/bash

echo $HOSTNAME
unset LD_PRELOAD
echo export PATH=/home/alta/BLTSpeaking/exp-yw575/env/anaconda3/bin/:$PATH


export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
# export CUDA_VISIBLE_DEVICES=0
echo $CUDA_VISIBLE_DEVICES

# python 3.7
# pytorch 1.5
source activate /home/alta/BLTSpeaking/exp-yw575/env/anaconda3/envs/gec37
export PYTHONBIN=/home/alta/BLTSpeaking/exp-yw575/env/anaconda3/envs/gec37/bin/python3

# ===================================================================================
# ------------------------ DIR --------------------------
train_path_src=/home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.train16.inc
train_path_tgt=/home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.train16.corr
dev_path_src=/home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.dev16.inc
dev_path_tgt=/home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.dev16.corr

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
num_epochs=100

checkpoint_every=5000 # ~10k if 2M, batch - 256
print_every=1000

grab_memory='False'
loaddir=/home/alta/BLTSpeaking/exp-ytl28/projects/gec-pretrained/exp-t5-written/models/v001/checkpoints-combine/combine/
load_mode='null' # 'resume' | 'restart' | 'null'

# ----------------------- [debug] ---------------------------
# orig_path=/home/alta/BLTSpeaking/exp-ytl28/projects/gec-pretrained/exp-t5-written
# train_path_src=$orig_path/lib/gec-train-bpe-written/prep/dev.src
# train_path_tgt=$orig_path/lib/gec-train-bpe-written/prep/dev.tgt
# dev_path_src=$orig_path/lib/gec-train-bpe-written/prep/toy.src
# dev_path_tgt=$orig_path/lib/gec-train-bpe-written/prep/toy.tgt
# num_epochs=5
# minibatch_split=1
# batch_size=2
# checkpoint_every=10
# print_every=2

# ----------------------- [noise] ---------------------------
ntype=Gaussian-adversarial #Adversarial, Bernoulli, Gaussian-adversarial, Adversarial
nway=mul
mean=1.0
weight=0.1
savedir=models/fce_train/${ntype}_${nway}_${mean}_${weight}_${batch_size}_${minibatch_split}_002/
# ===================================================================================
$PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/train.py \
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
	--weight $weight \

# Run below command to submit this script as an array job
# qsub -cwd -j yes -o 'LOGs/train_adv_v1.log' -P esol -l hostname='*' -l qp=cuda-low -l gpuclass='Volta' -l osrel='*' debug.sh 1 1

# Adv 0.1 mul
# qsub -cwd -j yes -o 'LOGs/train_adv_v2.log' -P esol -l hostname='*' -l qp=cuda-low -l gpuclass='*' -l osrel='*' debug.sh 1 1

# Gaussian mul 1.8 batch size = 2
# qsub -cwd -j yes -o 'LOGs/train_adv_v3.log' -P esol -l hostname='*' -l qp=cuda-low -l gpuclass='*' -l osrel='*' debug.sh 1 1

# Gasusian-adv 0.1 mul batch size = 2
# qsub -cwd -j yes -o 'LOGs/train_adv_v4.log' -P esol -l hostname='*' -l qp=cuda-low -l gpuclass='*' -l osrel='*' debug.sh 1 1

# Bernoulli 0.7 mul batch size = 2
# qsub -cwd -j yes -o 'LOGs/train_adv_v5.log' -P esol -l hostname='*' -l qp=cuda-low -l gpuclass='*' -l osrel='*' debug.sh 1 1

# Orig batch size = 2
# qsub -cwd -j yes -o 'LOGs/fce_train/train_orig.log' -P esol -l hostname='*' -l qp=cuda-low -l gpuclass='*' -l osrel='*' debug.sh 1 1
# Orig batch size = 256
# qsub -cwd -j yes -o 'LOGs/fce_train/train_orig_256.log' -P esol -l hostname='*' -l qp=cuda-low -l gpuclass='*' -l osrel='*' debug.sh 1 1

# Gau 1.75 mul 256
# qsub -cwd -j yes -o 'LOGs/fce_train/train_gau_mul_1.75_256.log' -P esol -l hostname='*' -l qp=cuda-low -l gpuclass='*' -l osrel='*' debug.sh 1 1

# Adv 0.1 mul 256
# qsub -cwd -j yes -o 'LOGs/fce_train/train_adv_0.1_256.log' -P esol -l hostname='*' -l qp=cuda-low -l gpuclass='*' -l osrel='*' debug.sh 1 1