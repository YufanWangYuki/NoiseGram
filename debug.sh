#!/bin/bash
#$ -S /bin/bash

echo $HOSTNAME
unset LD_PRELOAD
PATH=/opt/intel/composer_xe_2013_sp1.2.144/bin/intel64:/opt/intel/composer_xe_2013_sp1.2.144/mpirt/bin/intel64:/opt/intel/composer_xe_2013_sp1.2.144/debugger/gdb/intel64_mic/py27/bin:/opt/intel/composer_xe_2013_sp1.2.144/debugger/gdb/intel64/py27/bin:/opt/intel/composer_xe_2013_sp1.2.144/bin/intel64:/opt/intel/composer_xe_2013_sp1.2.144/bin/intel64_mic:/opt/intel/composer_xe_2013_sp1.2.144/debugger/gui/intel64:/home/mifs/yw575/bin:/home/mifs/yw575/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games: No such file or directory
echo export PATH=/home/alta/BLTSpeaking/exp-yw575/env/anaconda3/bin/:$PATH

# export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
export CUDA_VISIBLE_DEVICES=0
echo $CUDA_VISIBLE_DEVICES

# python 3.7
source activate /home/alta/BLTSpeaking/exp-yw575/env/anaconda3/envs/gec37
export PYTHONBIN=/home/alta/BLTSpeaking/exp-yw575/env/anaconda3/envs/gec37/bin/python3

# /home/alta/BLTSpeaking/exp-ytl28/env/anaconda3/bin/:/opt/intel/composer_xe_2013_sp1.2.144/bin/intel64:/opt/intel/composer_xe_2013_sp1.2.144/mpirt/bin/intel64:/opt/intel/composer_xe_2013_sp1.2.144/debugger/gdb/intel64_mic/py27/bin:/opt/intel/composer_xe_2013_sp1.2.144/debugger/gdb/intel64/py27/bin:/opt/intel/composer_xe_2013_sp1.2.144/bin/intel64:/opt/intel/composer_xe_2013_sp1.2.144/bin/intel64_mic:/opt/intel/composer_xe_2013_sp1.2.144/debugger/gui/intel64:/home/mifs/yw575/bin:/home/mifs/yw575/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games: No such file or directory

# ===================================================================================
# ------------------------ DIR --------------------------
orig_path=/home/alta/BLTSpeaking/exp-ytl28/projects/gec-pretrained/exp-t5-written
train_path_src=$orig_path/lib/gec-train-bpe-written/prep/train.src
train_path_tgt=$orig_path/lib/gec-train-bpe-written/prep/train.tgt
dev_path_src=$orig_path/lib/gec-train-bpe-written/prep/dev.src
dev_path_tgt=$orig_path/lib/gec-train-bpe-written/prep/dev.tgt

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
minibatch_split=2 #8 for million
num_epochs=100

checkpoint_every=5000 # ~10k if 2M, batch - 256
print_every=1000

grab_memory='False'
loaddir='None'
savedir=models/v001/
load_mode='null' # 'resume' | 'restart' | 'null'

# ----------------------- [debug] ---------------------------
orig_path=/home/alta/BLTSpeaking/exp-ytl28/projects/gec-pretrained/exp-t5-written
train_path_src=$orig_path/lib/gec-train-bpe-written/prep/dev.src
train_path_tgt=$orig_path/lib/gec-train-bpe-written/prep/dev.tgt
dev_path_src=$orig_path/lib/gec-train-bpe-written/prep/toy.src
dev_path_tgt=$orig_path/lib/gec-train-bpe-written/prep/toy.tgt
num_epochs=5
minibatch_split=1
batch_size=2
checkpoint_every=10
print_every=2

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

# Run below command to submit this script as an array job
# qsub -cwd -j yes -P esol -l qp=low -o LOGs/train.txt -t 1-5 -l not_host="air113|air116" train.sh 1
