#!/bin/bash

#python train.py --model=unet --problem=firstbreak --noise_type=-1 --noise_scale=0.0 --device=1 --epochs=15
#
python train.py --model=unet --problem=denoise --noise_type=0 --noise_scale=0.25 --device=1 --epochs=50 --attack=fgsm
python train.py --model=unet --problem=denoise --noise_type=0 --noise_scale=0.5 --device=1 --epochs=50 --attack=fgsm
python train.py --model=unet --problem=denoise --noise_type=0 --noise_scale=1.0 --device=1 --epochs=50 --attack=fgsm
python train.py --model=unet --problem=denoise --noise_type=0 --noise_scale=2.0 --device=1  --epochs=50 --attack=fgsm
#
python train.py --model=unet --problem=denoise --noise_type=1 --noise_scale=0.25 --device=1 --epochs=50 --attack=fgsm
python train.py --model=unet --problem=denoise --noise_type=1 --noise_scale=0.5 --device=1 --epochs=50 --attack=fgsm
python train.py --model=unet --problem=denoise --noise_type=1 --noise_scale=1.0 --device=1 --epochs=50 --attack=fgsm
python train.py --model=unet --problem=denoise --noise_type=1 --noise_scale=2.0 --device=1  --epochs=50 --attack=fgsm
#
python train.py --model=unet --problem=denoise --noise_type=2 --noise_scale=0.25 --device=1 --epochs=50 --attack=fgsm
python train.py --model=unet --problem=denoise --noise_type=2 --noise_scale=0.5 --device=1 --epochs=50 --attack=fgsm
python train.py --model=unet --problem=denoise --noise_type=2 --noise_scale=1.0 --device=1 --epochs=50 --attack=fgsm
python train.py --model=unet --problem=denoise --noise_type=2 --noise_scale=2.0 --device=1  --epochs=50 --attack=fgsm
#
python train.py --model=unet --problem=denoise --noise_type=3 --noise_scale=0.25 --device=1 --epochs=50 --attack=fgsm
python train.py --model=unet --problem=denoise --noise_type=3 --noise_scale=0.5 --device=1 --epochs=50 --attack=fgsm
python train.py --model=unet --problem=denoise --noise_type=3 --noise_scale=1.0 --device=1 --epochs=50 --attack=fgsm
python train.py --model=unet --problem=denoise --noise_type=3 --noise_scale=2.0 --device=1  --epochs=50 --attack=fgsm