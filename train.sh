#!/bin/bash
#$ -S /bin/bash

python train_residue_detector.py /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.train16.inc /home/alta/BLTSpeaking/exp-ytl28/projects/gec-pretrained/exp-t5-written/models/v001/checkpoints-combine/combine/model-dict.pt LOGs/train_residue_detector_beam1_N1.log residue_detectors/residue_detector_beam1_N1.th --attack_phrase='ink' --N=1 --epochs=20 --cpu=yes

python train_residue_detector.py /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.train16.inc /home/alta/BLTSpeaking/exp-ytl28/projects/gec-pretrained/exp-t5-written/models/v001/checkpoints-combine/combine/model-dict.pt LOGs/train_residue_detector_beam1_N2.log residue_detectors/residue_detector_beam1_N2.th --attack_phrase='ink l' --N=2 --epochs=20 --cpu=yes

python train_residue_detector.py /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.train16.inc /home/alta/BLTSpeaking/exp-ytl28/projects/gec-pretrained/exp-t5-written/models/v001/checkpoints-combine/combine/model-dict.pt LOGs/train_residue_detector_beam1_N3.log residue_detectors/residue_detector_beam1_N3.th --attack_phrase='ink l xml' --N=3 --epochs=20 --cpu=yes

python train_residue_detector.py /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.train16.inc /home/alta/BLTSpeaking/exp-ytl28/projects/gec-pretrained/exp-t5-written/models/v001/checkpoints-combine/combine/model-dict.pt LOGs/train_residue_detector_beam1_N4.log residue_detectors/residue_detector_beam1_N4.th --attack_phrase='ink l xml mv' --N=4 --epochs=20 --cpu=yes

python train_residue_detector.py /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.train16.inc /home/alta/BLTSpeaking/exp-ytl28/projects/gec-pretrained/exp-t5-written/models/v001/checkpoints-combine/combine/model-dict.pt LOGs/train_residue_detector_beam1_N5.log residue_detectors/residue_detector_beam1_N5.th --attack_phrase='ink l xml mv sub' --N=5 --epochs=20 --cpu=yes

python train_residue_detector.py /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.train16.inc /home/alta/BLTSpeaking/exp-ytl28/projects/gec-pretrained/exp-t5-written/models/v001/checkpoints-combine/combine/model-dict.pt LOGs/train_residue_detector_beam1_N6.log residue_detectors/residue_detector_beam1_N6.th --attack_phrase='ink l xml mv sub xu' --N=6 --epochs=20 --cpu=yes

python train_residue_detector.py /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.train16.inc /home/alta/BLTSpeaking/exp-ytl28/projects/gec-pretrained/exp-t5-written/models/v001/checkpoints-combine/combine/model-dict.pt LOGs/train_residue_detector_beam1_N7.log residue_detectors/residue_detector_beam1_N7.th --attack_phrase='ink l xml mv sub xu bec' --N=7 --epochs=20 --cpu=yes

python train_residue_detector.py /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.train16.inc /home/alta/BLTSpeaking/exp-ytl28/projects/gec-pretrained/exp-t5-written/models/v001/checkpoints-combine/combine/model-dict.pt LOGs/train_residue_detector_beam1_N8.log residue_detectors/residue_detector_beam1_N8.th --attack_phrase='ink l xml mv sub xu bec l' --N=8 --epochs=20 --cpu=yes

python train_residue_detector.py /home/alta/CLC/LNRC/exams/FCEsplit-public/v3/fce-public.train16.inc /home/alta/BLTSpeaking/exp-ytl28/projects/gec-pretrained/exp-t5-written/models/v001/checkpoints-combine/combine/model-dict.pt LOGs/train_residue_detector_beam1_N9.log residue_detectors/residue_detector_beam1_N9.th --attack_phrase='ink l xml mv sub xu bec l sub' --N=9 --epochs=20 --cpu=yes
