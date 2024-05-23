#!/bin/bash

cd ../src/ || exit

python train.py --model=restormer --problem=denoise --noise_type=2 --noise_scale=2.0 --device=0 --epochs=50 --attack=fgsm
python train.py --model=restormer --problem=firstbreak --noise_type=2 --noise_scale=2.0 --device=0 --epochs=30 --pretrained=restormer_denoise_noisetype_2_noisescale_2.0_dataclip_False_attack_fgsm_pretrained_False.pkl

#python train.py --model=restormer --problem=denoise --noise_type=0 --noise_scale=0.25 --device=0 --epochs=50
#python train.py --model=restormer --problem=denoise --noise_type=0 --noise_scale=0.5 --device=0 --epochs=50
#python train.py --model=restormer --problem=denoise --noise_type=0 --noise_scale=1.0 --device=0 --epochs=50
#python train.py --model=restormer --problem=denoise --noise_type=0 --noise_scale=2.0 --device=0 --epochs=50
##
#python train.py --model=restormer --problem=denoise --noise_type=1 --noise_scale=0.25 --device=0 --epochs=50
#python train.py --model=restormer --problem=denoise --noise_type=1 --noise_scale=0.5 --device=0 --epochs=50
#python train.py --model=restormer --problem=denoise --noise_type=1 --noise_scale=1.0 --device=0 --epochs=50
#python train.py --model=restormer --problem=denoise --noise_type=1 --noise_scale=2.0 --device=0 --epochs=50
##
#python train.py --model=restormer --problem=denoise --noise_type=2 --noise_scale=0.25 --device=0 --epochs=50
#python train.py --model=restormer --problem=denoise --noise_type=2 --noise_scale=0.5 --device=0 --epochs=50
#python train.py --model=restormer --problem=denoise --noise_type=2 --noise_scale=1.0 --device=0 --epochs=50
#python train.py --model=restormer --problem=denoise --noise_type=2 --noise_scale=2.0 --device=0 --epochs=50
##
#python train.py --model=restormer --problem=denoise --noise_type=3 --noise_scale=0.25 --device=0 --epochs=50
#python train.py --model=restormer --problem=denoise --noise_type=3 --noise_scale=0.5 --device=0 --epochs=50
#python train.py --model=restormer --problem=denoise --noise_type=3 --noise_scale=1.0 --device=0 --epochs=50
#python train.py --model=restormer --problem=denoise --noise_type=3 --noise_scale=2.0 --device=0 --epochs=50
##
#python train.py --model=restormer --problem=denoise --noise_type=0 --noise_scale=0.25 --device=0 --epochs=50 --attack fgsm
#python train.py --model=restormer --problem=denoise --noise_type=0 --noise_scale=0.5 --device=0 --epochs=50 --attack fgsm
#python train.py --model=restormer --problem=denoise --noise_type=0 --noise_scale=1.0 --device=0 --epochs=50 --attack fgsm
#python train.py --model=restormer --problem=denoise --noise_type=0 --noise_scale=2.0 --device=0 --epochs=50 --attack fgsm
##
#python train.py --model=restormer --problem=denoise --noise_type=1 --noise_scale=0.25 --device=0 --epochs=50 --attack fgsm
#python train.py --model=restormer --problem=denoise --noise_type=1 --noise_scale=0.5 --device=0 --epochs=50 --attack fgsm
#python train.py --model=restormer --problem=denoise --noise_type=1 --noise_scale=1.0 --device=0 --epochs=50 --attack fgsm
#python train.py --model=restormer --problem=denoise --noise_type=1 --noise_scale=2.0 --device=0 --epochs=50 --attack fgsm
##
#python train.py --model=restormer --problem=denoise --noise_type=2 --noise_scale=0.25 --device=0 --epochs=50 --attack fgsm
#python train.py --model=restormer --problem=denoise --noise_type=2 --noise_scale=0.5 --device=0 --epochs=50 --attack fgsm
#python train.py --model=restormer --problem=denoise --noise_type=2 --noise_scale=1.0 --device=0 --epochs=50 --attack fgsm
#python train.py --model=restormer --problem=denoise --noise_type=2 --noise_scale=2.0 --device=0 --epochs=50 --attack fgsm
##
#python train.py --model=restormer --problem=denoise --noise_type=3 --noise_scale=0.25 --device=0 --epochs=50 --attack fgsm
#python train.py --model=restormer --problem=denoise --noise_type=3 --noise_scale=0.5 --device=0 --epochs=50 --attack fgsm
#python train.py --model=restormer --problem=denoise --noise_type=3 --noise_scale=1.0 --device=0 --epochs=50 --attack fgsm
#python train.py --model=restormer --problem=denoise --noise_type=3 --noise_scale=2.0 --device=0 --epochs=50 --attack fgsm