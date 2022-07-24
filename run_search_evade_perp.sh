#!/bin/bash
#$ -S /bin/bash


# export PATH=/home/alta/BLTSpeaking/exp-yw575/env/anaconda3/bin/:$PATH
source ~/.bashrc
# conda activate /home/alta/BLTSpeaking/exp-yw575/env/anaconda3/envs/gec37
conda activate gec37
# export PYTHONBIN=/home/alta/BLTSpeaking/exp-yw575/env/anaconda3/envs/gec37/bin/python3
# export PYTHONPATH="${PYTHONPATH}:/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/"

# export CUDA_VISIBLE_DEVICES=0
# echo $CUDA_VISIBLE_DEVICES

export OMP_NUM_THREADS=1 # export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 # export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 # export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 # export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 # export NUMEXPR_NUM_THREADS=1


# 44928 chutzpah ii bibb en fyi
# chutzpah vb ditka 0.4290416971470373
# chutzpah ii bibb en 

python /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/uni_attack_evade_perplexity.py /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.train16.inc /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/models/v003/volta_Gaussian-adversarial_mul_1.0_0.1_256_8/checkpoints-combine/combine/ /home/alta/BLTSpeaking/grd-graphemic-vr313/speech_processing/adversarial_attack/word2vec/test_words.txt /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/universal_attack_logs/v003best/evade_perp_beam1/k6/words$SGE_TASK_ID.txt --prev_attack="chutzpah vb clap shu wring" --search_size=200 --start=$SGE_TASK_ID --num_points=500 --perp_thresh=243

# SGE_TASK_ID=1
# python /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/uni_attack_evade_perplexity.py /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.train16.inc /home/alta/BLTSpeaking/exp-ytl28/projects/gec-pretrained/exp-t5-written/models/v001/checkpoints-combine/combine/ /home/alta/BLTSpeaking/grd-graphemic-vr313/speech_processing/adversarial_attack/word2vec/test_words.txt /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/universal_attack_logs/evade_perp_beam1_orig/k1/words$SGE_TASK_ID.txt --prev_attack= --search_size=200 --start=$SGE_TASK_ID --num_points=500 --perp_thresh=243

# Run below command to submit this script as an array job
# qsub -cwd -j yes -P esol -l qp=low -o LOGs/run-array-evade/run-array-evade_perp_orig_v3.txt -t 50-224 -l not_host="air113|air112" run_search_evade_perp.sh

# qsub -cwd -j yes -P esol -l qp=low -o LOGs/run-array-evade/run-array-evade_perp_V003.txt -t 20-224 -l not_host="air113|air116" run_search_evade_perp.sh

# qsub -cwd -j yes -P esol -l qp=low -o LOGs/run-array-evade/run-array-evade_perp_V003.txt -t 1-19 -l not_host="air113|air116" run_search_evade_perp.sh

# qsub -cwd -j yes -P esol -l qp=low -o LOGs/run-array-evade/run-array-evade_perp_V003_k2.txt -t 1-224 -l not_host="air113|air116" run_search_evade_perp.sh

# qsub -cwd -j yes -P esol -l qp=low -o LOGs/run-array-evade/run-array-evade_perp_V003_k3.txt -t 1-224 -l not_host="air113|air116" run_search_evade_perp.sh

# qsub -cwd -j yes -P esol -l qp=low -o LOGs/run-array-evade/run-array-evade_perp_V003_k4.txt -t 1-224 -l not_host="air113|air116" run_search_evade_perp.sh

# qsub -cwd -j yes -P esol -l qp=low -o LOGs/run-array-evade/run-array-evade_perp_V003_k5.txt -t 1-224 -l not_host="air113|air116" run_search_evade_perp.sh

# qsub -cwd -j yes -P esol -l qp=low -o LOGs/run-array-evade/run-array-evade_perp_V003_k6.txt -t 1-224 -l not_host="air113|air116" run_search_evade_perp.sh

# 0-3 *1000 = 4000 million