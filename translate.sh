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
model=models/v001

# ckpt=2022_02_01_15_37_10
ckpt=combine

# ------ [Generation for reranker eval set] ----------
# FCE
fname=eval-clc-test
ftst=/home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.train16.inc

max_tgt_len=100

# ------ decode --------
# MODE
# 1: save combiend ckpts
# 2. save model dict
# 3: translate - <sent>
# 4: translate with verbo - <sid> <score> <sent>
eval_mode=3
use_gpu='True'

batch_size=50
mode='beam-1'
# batch_size=1
# mode='beam-50'


# ----------------------- [noise] ---------------------------
ntype=Gaussian
nway=mul
mean=1.0
weight=0.1

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
        outdir=$model/$fname-"$mode"/combine/
        # loaddir=$model/checkpoints-combine/combine
    else # [single ckpt]
        outdir=$model/$fname-"$mode"/$ckpt/
        # loaddir=$model/checkpoints/$ckpt
    fi
    combine_path='None'
    echo 'LOAD: '$loaddir
    echo 'OUT: '$outdir
fi

# ----- run ------
$PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/translate.py \
    --test_path_src $ftst \
    --load $loaddir \
    --test_path_out $outdir \
    --max_tgt_len $max_tgt_len \
    --batch_size $batch_size \
    --mode $mode \
    --use_gpu $use_gpu \
    --eval_mode $eval_mode \
    --combine_path $combine_path \
    --ntype $ntype \
	--nway $nway \
	--mean $mean \
	--weight $weight \
