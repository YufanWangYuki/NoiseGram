#!/bin/bash
#$ -S /bin/bash

#Check Number of Args
if [[ $# -ne 1 ]]; then
   echo "Usage: $0 N"
   echo "  e.g: $0 1"
   exit 2
fi

N=$1
let JSON=N-1
CURR=just
POS=RB

source ~/.bashrc
conda activate conda_env36

export OMP_NUM_THREADS=1 # export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 # export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 # export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 # export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 # export NUMEXPR_NUM_THREADS=1


python /home/alta/BLTSpeaking/exp-vr313/GEC/SubTunedGramformerAttack/uni_attack.py /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.train16.inc /home/alta/BLTSpeaking/exp-ytl28/projects/gec-pretrained/exp-t5-written/models/v001/checkpoints-combine/combine/model-dict.pt /home/alta/BLTSpeaking/exp-vr313/GEC/SubTunedGramformerAttack/experiments/vocab_files/$POS.json $CURR /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/universal_attack_logs/v003best/evade_perp_beam1/temp/words$SGE_TASK_ID.txt --prev_attack /home/alta/BLTSpeaking/exp-vr313/GEC/SubTunedGramformerAttack/experiments/sub_dicts/$POS/k$JSON.json --search_size 50 --start $SGE_TASK_ID --num_points 25000

# Run below command to submit this script as an array job
# qsub -cwd -j yes -P esol -l qp=low -o LOGs/run-array_RB_temp.txt -t 1-20 -l not_host="air113|air116" run_search_RB.sh 1
