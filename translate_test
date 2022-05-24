#!/bin/bash
#$ -S /bin/bash

echo $HOSTNAME
unset LD_PRELOAD
export PATH=/home/alta/BLTSpeaking/exp-ytl28/env/anaconda3/bin/:$PATH

# export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
export CUDA_VISIBLE_DEVICES=0
echo $CUDA_VISIBLE_DEVICES

# python 3.7
# pytorch 1.5
source activate /home/alta/BLTSpeaking/exp-ytl28/env/anaconda3/envs/py38-pt15-cuda10
export PYTHONBIN=/home/alta/BLTSpeaking/exp-ytl28/env/anaconda3/envs/py38-pt15-cuda10/bin/python3

# ===================================================================================
# ----- [MODEL] ------
model='gramformer'
# model='vennify'
# model='flexudy'

beam_width=1
device='cuda'

# ------[DATA] ------
# FCE-text
fname=eval_clc-orig
# ftst=./lib/gec-tst-collate/test-clc-orig.src
ftst=/home/alta/BLTSpeaking/exp-ytl28/projects/lib/gec-tst-collate/test-clc-orig.src

# FCE
# fname=eval_clc-spellcorr
# ftst=./lib/gec-tst-collate-v2/test-clc-spellcorr.src.derasp

# NICT
# fname=eval_nict-flt
# ftst=./lib/gec-tst-collate-v2/test-nict-flt.src.derasp
# fname=eval_nict-dsf
# ftst=./lib/gec-tst-collate-v2/test-nict-dsf.src.derasp
# fname=eval_nict-autoflt
# ftst=./lib/gec-tst-collate-v2/test-nict-autoflt.src.derasp

# ELIT
# fname=eval_elitrls2-man-flt
# ftst=./lib/gec-tst-collate-v2/test-elit-man-flt.src.derasp
# fname=eval_elitrls2-man-dsf
# ftst=./lib/gec-tst-collate-v2/test-elit-man-dsf.src.derasp
# fname=eval_elitrls2-man-autoflt
# ftst=./lib/gec-tst-collate-v2/test-elit-man-autoflt.src.derasp
# fname=eval_elitrls2-asr-dsf
# ftst=./lib/gec-tst-collate-v2/test-elit-asr-dsf.src.derasp
# fname=eval_elitrls2-asr-autoflt
# ftst=./lib/gec-tst-collate-v2/test-elit-asr-autoflt.src.derasp

# ----- [DIR] ------
outdir=./models/"$model"/"$fname"
mkdir -p $outdir
fout="$outdir"/beam"$beam_width"

# ===================================================================================
$PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/translate_gramformer.py \
    --test_path_src $ftst \
    --test_path_out $fout \
    --beam_width $beam_width \
    --device $device
