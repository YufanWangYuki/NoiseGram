
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

for num in 1
do
$PYTHONBIN get_best.py \
    /home/alta/BLTSpeaking/exp-vr313/GEC/TunedGramformerAttack/universal_attack_logs/v003best/evade_perp_beam1/k${num}/ \
    /home/alta/BLTSpeaking/exp-yw575/GEC/AttackGram/LOGs/best_words/best_words_${num}.txt
done