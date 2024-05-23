#!/bin/bash

cd ../src/ || exit

python train.py --model=swin --problem=denoise --noise_type=2 --noise_scale=2.0 --device=1 --epochs=100
python train.py --model=swin --problem=denoise --noise_type=2 --noise_scale=2.0 --device=1 --epochs=100 --attack=fgsm

python train.py --model=swin --problem=firstbreak --noise_type=2 --noise_scale=2.0 --device=1 --epochs=50 --pretrained=swin_denoise_noisetype_2_noisescale_2.0_dataclip_False_attack_none_pretrained_False.pkl
python train.py --model=swin --problem=firstbreak --noise_type=2 --noise_scale=2.0 --device=1 --epochs=50 --attack=fgsm --pretrained=swin_denoise_noisetype_2_noisescale_2.0_dataclip_False_attack_fgsm_pretrained_False.pkl

#python train.py --model=swin --problem=firstbreak --noise_type=0 --noise_scale=0.25 --device=1 --epochs=50 --pretrained=swin_denoise_noisetype_0_noisescale_0.25_dataclip_True_attack_none_pretrained_False.pkl
#python train.py --model=swin --problem=firstbreak --noise_type=0 --noise_scale=0.5 --device=1 --epochs=50 --pretrained=swin_denoise_noisetype_0_noisescale_0.5_dataclip_True_attack_none_pretrained_False.pkl
###
##python train.py --model=swin --problem=firstbreak --noise_type=1 --noise_scale=0.25 --device=1 --epochs=50 --attack=fgsm
##python train.py --model=swin --problem=firstbreak --noise_type=1 --noise_scale=0.5 --device=1 --epochs=50 --attack=fgsm
#python train.py --model=swin --problem=firstbreak --noise_type=1 --noise_scale=0.25 --device=1 --epochs=50 --pretrained=swin_denoise_noisetype_1_noisescale_0.25_dataclip_True_attack_none_pretrained_False.pkl
#python train.py --model=swin --problem=firstbreak --noise_type=1 --noise_scale=0.5 --device=1 --epochs=50 --pretrained=swin_denoise_noisetype_1_noisescale_0.5_dataclip_True_attack_none_pretrained_False.pkl
##
##python train.py --model=swin --problem=firstbreak --noise_type=2 --noise_scale=0.25 --device=1 --epochs=50 --attack=fgsm
##python train.py --model=swin --problem=firstbreak --noise_type=2 --noise_scale=0.5 --device=1 --epochs=50 --attack=fgsm
#python train.py --model=swin --problem=firstbreak --noise_type=2 --noise_scale=0.25 --device=1 --epochs=50 --pretrained=swin_denoise_noisetype_2_noisescale_0.25_dataclip_True_attack_none_pretrained_False.pkl
#python train.py --model=swin --problem=firstbreak --noise_type=2 --noise_scale=0.5 --device=1 --epochs=50 --pretrained=swin_denoise_noisetype_2_noisescale_0.5_dataclip_True_attack_none_pretrained_False.pkl
##
#python train.py --model=swin --problem=firstbreak --noise_type=3 --noise_scale=0.25 --device=1 --epochs=50 --pretrained=swin_denoise_noisetype_3_noisescale_0.25_dataclip_True_attack_none_pretrained_False.pkl
##python train.py --model=swin --problem=firstbreak --noise_type=3 --noise_scale=0.5 --device=1 --epochs=50
#python train.py --model=swin --problem=firstbreak --noise_type=3 --noise_scale=1.0 --device=1 --epochs=50 --pretrained=swin_denoise_noisetype_3_noisescale_1.0_dataclip_True_attack_none_pretrained_False.pkl
#python train.py --model=swin --problem=firstbreak --noise_type=3 --noise_scale=2.0 --device=1 --epochs=50 --pretrained=swin_denoise_noisetype_3_noisescale_2.0_dataclip_True_attack_none_pretrained_False.pkl
#
#python train.py --model=swin --problem=firstbreak --noise_type=4 --noise_scale=0.25 --device=1 --epochs=50 --attack=fgsm
#python train.py --model=swin --problem=firstbreak --noise_type=4 --noise_scale=0.5 --device=1 --epochs=50 --attack=fgsm
#python train.py --model=swin --problem=firstbreak --noise_type=4 --noise_scale=1.0 --device=1 --epochs=50 --pretrained=swin_denoise_noisetype_2_noisescale_1.0_dataclip_True_attack_none_pretrained_False.pkl
#python train.py --model=swin --problem=firstbreak --noise_type=4 --noise_scale=2.0 --device=1 --epochs=50 --pretrained=swin_denoise_noisetype_2_noisescale_2.0_dataclip_True_attack_none_pretrained_False.pkl
##
#python train.py --model=swin --problem=firstbreak --noise_type=0 --noise_scale=0.25 --device=1 --epochs=50 --attack fgsm
#python train.py --model=swin --problem=firstbreak --noise_type=0 --noise_scale=0.5 --device=1 --epochs=50 --attack fgsm
#python train.py --model=swin --problem=firstbreak --noise_type=0 --noise_scale=1.0 --device=1 --epochs=50 --attack fgsm
#python train.py --model=swin --problem=firstbreak --noise_type=0 --noise_scale=2.0 --device=1 --epochs=50 --attack fgsm
##
#python train.py --model=swin --problem=firstbreak --noise_type=1 --noise_scale=0.25 --device=1 --epochs=50 --attack fgsm
#python train.py --model=swin --problem=firstbreak --noise_type=1 --noise_scale=0.5 --device=1 --epochs=50 --attack fgsm
#python train.py --model=swin --problem=firstbreak --noise_type=1 --noise_scale=1.0 --device=1 --epochs=50 --attack fgsm
#python train.py --model=swin --problem=firstbreak --noise_type=1 --noise_scale=2.0 --device=1 --epochs=50 --attack fgsm
##
#python train.py --model=swin --problem=firstbreak --noise_type=2 --noise_scale=0.25 --device=1 --epochs=50 --attack fgsm
#python train.py --model=swin --problem=firstbreak --noise_type=2 --noise_scale=0.5 --device=1 --epochs=50 --attack fgsm
#python train.py --model=swin --problem=firstbreak --noise_type=2 --noise_scale=1.0 --device=1 --epochs=50 --attack fgsm
#python train.py --model=swin --problem=firstbreak --noise_type=2 --noise_scale=2.0 --device=1 --epochs=50 --attack fgsm
##
#python train.py --model=swin --problem=firstbreak --noise_type=3 --noise_scale=0.25 --device=1 --epochs=50 --attack fgsm
#python train.py --model=swin --problem=firstbreak --noise_type=3 --noise_scale=0.5 --device=1 --epochs=50 --attack fgsm
#python train.py --model=swin --problem=firstbreak --noise_type=3 --noise_scale=1.0 --device=1 --epochs=50 --attack fgsm
#python train.py --model=swin --problem=firstbreak --noise_type=3 --noise_scale=2.0 --device=1 --epochs=50 --attack fgsm