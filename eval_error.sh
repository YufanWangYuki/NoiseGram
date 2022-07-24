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
source ~/.bashrc
source activate /home/alta/BLTSpeaking/exp-yw575/env/anaconda3/envs/gec37
export PYTHONBIN=/home/alta/BLTSpeaking/exp-yw575/env/anaconda3/envs/gec37/bin/python3
export PYTHONPATH="${PYTHONPATH}:/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/"

# for name in comma_N1 comma_N2 comma_N3 comma_N4 comma_N5 comma_N6 comma_N7 comma_N8 comma_N9
# do
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED prediction_files/$exp/orig/$exp_combine_seed_1.pred \
#     --OUT edit_dist_files/$exp/orig.txt


# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED prediction_files/$exp/attacks/comma_N1_with_adv_not_removed.pred \
#     --OUT edit_dist_files/$exp/attacks/comma_N1_with_adv_not_removed.txt \
#     --phrase 'xl' \
#     --delim ','
# done

# name=comma_N2
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED prediction_files/$exp/attacks/${name}_with_adv_not_removed.pred \
#     --OUT edit_dist_files/$exp/attacks/${name}_with_adv_not_removed.txt \
#     --phrase 'xl ii' \
#     --delim ','

# name=comma_N3
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED prediction_files/$exp/attacks/${name}_with_adv_not_removed.pred \
#     --OUT edit_dist_files/$exp/attacks/${name}_with_adv_not_removed.txt \
#     --phrase 'xl ii xl' \
#     --delim ','

# name=comma_N4
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED prediction_files/$exp/attacks/${name}_with_adv_not_removed.pred \
#     --OUT edit_dist_files/$exp/attacks/${name}_with_adv_not_removed.txt \
#     --phrase 'xl ii xl lbs' \
#     --delim ','

# name=comma_N5
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED prediction_files/$exp/attacks/${name}_with_adv_not_removed.pred \
#     --OUT edit_dist_files/$exp/attacks/${name}_with_adv_not_removed.txt \
#     --phrase 'xl ii xl lbs sub' \
#     --delim ','

# name=comma_N6
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED prediction_files/$exp/attacks/${name}_with_adv_not_removed.pred \
#     --OUT edit_dist_files/$exp/attacks/${name}_with_adv_not_removed.txt \
#     --phrase 'xl ii xl lbs sub xl' \
#     --delim ','

# name=comma_N7
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED prediction_files/$exp/attacks/${name}_with_adv_not_removed.pred \
#     --OUT edit_dist_files/$exp/attacks/${name}_with_adv_not_removed.txt \
#     --phrase 'xl ii xl lbs sub xl dp' \
#     --delim ','

# name=comma_N8
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED prediction_files/$exp/attacks/${name}_with_adv_not_removed.pred \
#     --OUT edit_dist_files/$exp/attacks/${name}_with_adv_not_removed.txt \
#     --phrase 'xl ii xl lbs sub xl dp lbs' \
#     --delim ','

# name=comma_N9
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED prediction_files/$exp/attacks/${name}_with_adv_not_removed.pred \
#     --OUT edit_dist_files/$exp/attacks/${name}_with_adv_not_removed.txt \
#     --phrase 'xl ii xl lbs sub xl dp lbs lc' \
#     --delim ','

# mkdir edit_dist_files/v005
# for exp in volta_Gaussian-adversarial_add_0.0_100_256_8
# do
# mkdir edit_dist_files/v005/$exp
# mkdir edit_dist_files/v005/$exp/attacks_new
# outdir=edit_dist_files/v005/$exp/attacks_new
# name=full_N1
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED prediction_files/v005/$exp/attacks/${name}_with_adv_not_removed.pred \
#     --OUT ${outdir}/${name}_with_adv_not_removed.txt \
#     --phrase 'ink' \
#     --delim '.'

# name=full_N2
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED prediction_files/v005/$exp/attacks/${name}_with_adv_not_removed.pred \
#     --OUT ${outdir}/${name}_with_adv_not_removed.txt \
#     --phrase 'ink l' \
#     --delim '.'

# name=full_N3
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED prediction_files/v005/$exp/attacks/${name}_with_adv_not_removed.pred \
#     --OUT ${outdir}/${name}_with_adv_not_removed.txt \
#     --phrase 'ink l xml' \
#     --delim '.'

# name=full_N4
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED prediction_files/v005/$exp/attacks/${name}_with_adv_not_removed.pred \
#     --OUT ${outdir}/${name}_with_adv_not_removed.txt \
#     --phrase 'ink l xml mv' \
#     --delim '.'

# name=full_N5
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED prediction_files/v005/$exp/attacks/${name}_with_adv_not_removed.pred \
#     --OUT ${outdir}/${name}_with_adv_not_removed.txt \
#     --phrase 'ink l xml mv sub' \
#     --delim '.'

# name=full_N6
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED prediction_files/v005/$exp/attacks/${name}_with_adv_not_removed.pred \
#     --OUT ${outdir}/${name}_with_adv_not_removed.txt \
#     --phrase 'ink l xml mv sub xu' \
#     --delim '.'

# name=full_N7
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED prediction_files/v005/$exp/attacks/${name}_with_adv_not_removed.pred \
#     --OUT ${outdir}/${name}_with_adv_not_removed.txt \
#     --phrase 'ink l xml mv sub xu bec' \
#     --delim '.'

# name=full_N8
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED prediction_files/v005/$exp/attacks/${name}_with_adv_not_removed.pred \
#     --OUT ${outdir}/${name}_with_adv_not_removed.txt \
#     --phrase 'ink l xml mv sub xu bec l' \
#     --delim '.'

# name=full_N9
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED prediction_files/v005/$exp/attacks/${name}_with_adv_not_removed.pred \
#     --OUT ${outdir}/${name}_with_adv_not_removed.txt \
#     --phrase 'ink l xml mv sub xu bec l sub' \
#     --delim '.'


# name=orig
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED prediction_files/v005/${exp}_combine_seed_1.pred \
#     --OUT ${outdir}/${name}_with_adv_not_removed.txt
    
# done

# name=orig
# exp=Gaussian_mul_1.0_0.1_256_2_002
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED prediction_files/$exp/orig_combine_seed_1.pred \
#     --OUT edit_dist_files/$exp/attacks/${name}_with_adv_not_removed.txt


# mkdir edit_dist_files/v003

# # /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/prediction_files/v003/volta_Gaussian-adversarial_mul_1.0_0.1_256_8/attacks_perp/
for exp in volta_Gaussian-adversarial_mul_1.0_0.1_256_8
do
    pred_dir=prediction_files/v003/$exp/attacks_perp
    out_dir=edit_dist_files/v003/$exp/attacks_perp
#     mkdir edit_dist_files/v003/$exp
#     mkdir edit_dist_files/v003/$exp/attacks_perp
# name=full_N1
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED ${pred_dir}/${name}_with_adv_not_removed.pred \
#     --OUT ${out_dir}/${name}_with_adv_not_removed.txt \
#     --phrase 'chutzpah' \
#     --delim '.'

# name=full_N2
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED ${pred_dir}/${name}_with_adv_not_removed.pred \
#     --OUT ${out_dir}/${name}_with_adv_not_removed.txt \
#     --phrase 'chutzpah vb' \
#     --delim '.'

# name=full_N3
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED ${pred_dir}/${name}_with_adv_not_removed.pred \
#     --OUT ${out_dir}/${name}_with_adv_not_removed_v2.txt \
#     --phrase 'chutzpah vb clap' \
#     --delim '.'

# name=full_N4
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED ${pred_dir}/${name}_with_adv_not_removed.pred \
#     --OUT ${out_dir}/${name}_with_adv_not_removed_v2.txt \
#     --phrase 'chutzpah vb clap shu' \
#     --delim '.'

name=full_N5
$PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
    --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
    --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
    --PRED ${pred_dir}/${name}_with_adv_not_removed.pred \
    --OUT ${out_dir}/${name}_with_adv_not_removed.txt \
    --phrase 'chutzpah vb clap shu donato' \
    --delim '.'

# chutzpah ii bibb en fyi
# name=full_N6
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED ${pred_dir}/${name}_with_adv_not_removed.pred \
#     --OUT ${out_dir}/${name}_with_adv_not_removed.txt \
#     --phrase 'ink l xml mv sub xu' \
#     --delim '.'

# name=full_N7
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED ${pred_dir}/${name}_with_adv_not_removed.pred \
#     --OUT ${out_dir}/${name}_with_adv_not_removed.txt \
#     --phrase 'ink l xml mv sub xu bec' \
#     --delim '.'

# name=full_N8
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED ${pred_dir}/${name}_with_adv_not_removed.pred \
#     --OUT ${out_dir}/${name}_with_adv_not_removed.txt \
#     --phrase 'ink l xml mv sub xu bec l' \
#     --delim '.'

# name=full_N9
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED ${pred_dir}/${name}_with_adv_not_removed.pred \
#     --OUT ${out_dir}/${name}_with_adv_not_removed.txt \
#     --phrase 'ink l xml mv sub xu bec l sub' \
#     --delim '.'

# name=orig
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED prediction_files/$exp/orig_combine_seed_1.pred \
#     --OUT edit_dist_files/$exp/attacks/${name}_with_adv_not_removed.txt
done


# for exp in volta_Gaussian-adversarial_mul_1.0_0.1_256_8
# do
# pred_dir=prediction_files/v003/$exp/attacks_perp_old
# out_dir=edit_dist_files/v003/$exp/attacks_perp_old
# mkdir edit_dist_files/v003/$exp
# mkdir edit_dist_files/v003/$exp/attacks_perp_old
# name=full_N1
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED ${pred_dir}/${name}_with_adv_not_removed.pred \
#     --OUT ${out_dir}/${name}_with_adv_not_removed.txt \
#     --phrase 'trifecta' \
#     --delim '.'

# name=full_N2
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED ${pred_dir}/${name}_with_adv_not_removed.pred \
#     --OUT ${out_dir}/${name}_with_adv_not_removed.txt \
#    --phrase 'trifecta haiku' \
#     --delim '.'

# name=full_N3
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED ${pred_dir}/${name}_with_adv_not_removed.pred \
#     --OUT ${out_dir}/${name}_with_adv_not_removed.txt \
#     --phrase 'trifecta haiku utah' \
#     --delim '.'

# name=full_N4
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED ${pred_dir}/${name}_with_adv_not_removed.pred \
#     --OUT ${out_dir}/${name}_with_adv_not_removed.txt \
#     --phrase 'trifecta haiku utah intransigent' \
#     --delim '.'

# name=full_N5
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED ${pred_dir}/${name}_with_adv_not_removed.pred \
#     --OUT ${out_dir}/${name}_with_adv_not_removed.txt \
#     --phrase 'trifecta haiku utah intransigent penicillin' \
#     --delim '.'

# name=full_N6
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED ${pred_dir}/${name}_with_adv_not_removed.pred \
#     --OUT ${out_dir}/${name}_with_adv_not_removed.txt \
#     --phrase 'trifecta haiku utah intransigent penicillin baseline' \
#     --delim '.'

# name=full_N7
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED ${pred_dir}/${name}_with_adv_not_removed.pred \
#     --OUT ${out_dir}/${name}_with_adv_not_removed.txt \
#     --phrase 'trifecta haiku utah intransigent penicillin baseline exploratory' \
#     --delim '.'

# name=full_N8
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED ${pred_dir}/${name}_with_adv_not_removed.pred \
#     --OUT ${out_dir}/${name}_with_adv_not_removed.txt \
#     --phrase 'trifecta haiku utah intransigent penicillin baseline exploratory bioengineering' \
#     --delim '.'

# name=full_N9
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED ${pred_dir}/${name}_with_adv_not_removed.pred \
#     --OUT ${out_dir}/${name}_with_adv_not_removed.txt \
#     --phrase 'ink l xml mv sub xu bec l sub' \
#     --delim '.'

# name=orig
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED prediction_files/$exp/orig_combine_seed_1.pred \
#     --OUT edit_dist_files/$exp/attacks/${name}_with_adv_not_removed.txt
# done

# name=orig
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.inc \
#     --REF /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.test.corr \
#     --PRED prediction_files/v005/${exp}_combine_seed_1.pred \
#     --OUT edit_dist_files/v005/$exp/attacks/${name}_with_adv_not_removed.txt
# done
# qsub -cwd -j yes -o 'LOGs/count_err.log' -P esol -l hostname='*' -l qp=cuda-low -l gpuclass='*' -l osrel='*' eval_error.sh 1 1
# qsub -cwd -j yes -o 'LOGs/count_err_mul.log' -P esol -l hostname='*' -l qp=cuda-low -l gpuclass='*' -l osrel='*' eval_error.sh 1 1
# qsub -cwd -j yes -o 'LOGs/count_err_add_100.log' -P esol -l hostname='*' -l qp=cuda-low -l gpuclass='*' -l osrel='*' eval_error.sh 1 1



# mkdir edit_dist_files/orig
# for exp in CoNLL_volta_Gaussian_mul_1.0_0.0__256_8_combine_seed_1
# do
#     dir=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/prediction_files/orig/attacks
#     # input=/home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/prediction_files/orig/conlltest.inc
#     input=/home/alta/BLTSpeaking/exp-vr313/GEC/data/CoNLL-14/conll14st-test-data/noalt/input_sentences.txt
    
#     mkdir edit_dist_files/orig/$exp
#     outdir=edit_dist_files/orig/$exp
#     # volta_Gaussian_mul_1.0_0.0__256_8_perp_N5_CoNLL_with_adv_removed.pred

#     name=full_N5
#     ref=$dir/clean_volta_Gaussian_mul_1.0_0.0__256_8_full_N5_CoNLL_with_adv_not_removed.pred
# $PYTHONBIN /home/alta/BLTSpeaking/exp-yw575/GEC/NoiseGram/eval_error.py \
#     --SOURCE $input\
#     --REF $ref\
#     --PRED $ref \
#     --OUT ${outdir}/${name}_with_adv_not_removed.txt \
#     --phrase 'ink l xml mv sub' \
#     --delim '.'

# done