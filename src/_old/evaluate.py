import glob
from metrics import ConfusionMatrix, RMSE
import torch, torchvision
from models.build import build_model
from noiseadding import build_noise_transforms
from data import get_dataset, get_train_val_dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from skimage import transform

METADATA = '../metadata/'
EVALDATA = os.path.join(METADATA, 'evaluation/')

class Evaluator:
    def __init__(self, weight_file, device="cpu", activation=None, batch_size=8, skip=0):
        """
        Basic class to evaluate model
        :param weight_file: weight file to use
        :param device: specify device
        :param activation: specify activation in last layer
        :param batch_size: batch_size to use in validation. Ignored in real data.
        :param skip: skip prefix in weight_file name
        """
        self.weight_file = weight_file[:-4] if weight_file[-4:] == '.pkl' else weight_file
        self.batch_size = batch_size
        self.model_type, self.problem = weight_file[skip:].split('_')[:2]
        print(weight_file[skip:].split('_')[:2])
        if self.problem == 'firstbreak':
            self.metrics = ConfusionMatrix(2, ["empty", "firstbreak"])
            self.loss_type = nn.CrossEntropyLoss
        else:
            self.metrics = RMSE()
            self.loss_type = nn.MSELoss
        self.model = build_model(self.model_type, self.problem, activation=activation)
        self.device = torch.device(device)
        self.model.to(device)
        self.save_path = os.path.join(METADATA, self.weight_file + '.pkl')
        self.model.load_state_dict(torch.load(self.save_path))
        self.model.eval()
        self.figdir = os.path.join(EVALDATA, 'figures/')

    def predict_numpy(self, input, target, target_size=(224, 224)):
        input_ = transform.resize(input, target_size, preserve_range=True).astype(input.dtype)
        input_ /= np.abs(input_).max()
        input_ = torch.from_numpy(input_)[None,None,...].to(self.device)
        if target is not None:
            target_ = transform.resize(target, target_size, preserve_range=True).astype(target.dtype)
            target_ = torch.from_numpy(target_)[None,None,...].to(self.device)
        else:
            target_ = None
        with torch.no_grad():
            pred_ = self.model(input_)
        return input_.cpu().numpy(), pred_.cpu().numpy(), target_.cpu().numpy()

    def prepare_loader(self, noise_type=-1, noise_scale=0.25):
        """
        prepare loader
        :param noise_type: select noise_type: [-1,..., 6] and noise_scale: [0.25,0.5,1.0,2.0]
        :param noise_scale:
        :return:
        """
        noise_transforms = build_noise_transforms(noise_type=noise_type, scale=noise_scale)
        denoise_dataset = get_dataset(self.problem, noise_transforms=noise_transforms)
        _, val_dataset = get_train_val_dataset(denoise_dataset, generator=torch.Generator().manual_seed(42), valid_split=0.01)
        loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        return loader

    def prepare_real_loader(self):
        dataset = get_dataset('real')
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        return loader

    def evaluate_model(self, eval_type='synthetic', noise_type=-1, noise_scale=0.25, **eval_args):
        if eval_type == 'synthetic':
            loader = self.prepare_loader(noise_type, noise_scale)
            eval = self.evaluate
        else:
            loader = self.prepare_real_loader()
            eval = self.eval_real
        return eval(loader, prefix=f'{noise_type}_{noise_scale}_', **eval_args)

    def evaluate(self, loader, plot=True, to_file=True, prefix='', output=False, **att_args):
        self.metrics.reset()
        for i, (sample) in enumerate(loader):
            x, y = sample['input'].to(self.device), sample['target'].to(self.device)
            with torch.no_grad():
                y_pred = self.model(x)
                if self.problem == 'firstbreak':
                    y_pred = torch.argmax(y_pred, dim=1)  # get the most likely prediction
            self.metrics.add_batch(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
            print('_', end='')
        if output:
            print(f'Metrics({self.metrics.name}) : {self.metrics.get()}')
        if plot:
            self.plot_predictions(x, y, y_pred, to_file=to_file, prefix=prefix)
        return self.metrics.get()

    def plot_noise_palette(self):
        fig, axes = plt.subplots(7,4, figsize=[10,17])
        for i, noise_type in enumerate(range(-1, 6)):
            for j, noise_scale in enumerate([0.25, 0.5, 1.0, 2.0]):
                loader = self.prepare_loader(noise_type, noise_scale)
                sample = iter(loader).__next__()
                x = sample['input'].to(self.device)
                axes[i,j].imshow(x[0,0], cmap='seismic')
                axes[i,j].axis('off')
        fname = os.path.join(self.figdir, 'noise_pallette.jpg')
        fig.savefig(fname)
        plt.close(fig)

    def evaluate_robustness(self, plot=True, to_file=True, only_linear=False, fast=False, **eval_args):
        self.robustness = np.ones([7, 4]) * -1
        ### how many noise types to use
        if only_linear:
            n=4
        elif fast:
            n=3
        else:
            n=6
        for i, noise_type in enumerate(range(-1, n)):
            for j, noise_scale in enumerate([0.25, 0.5, 1.0, 2.0]):
                if i == 0 and j != 0:
                    continue
                self.robustness[i, j] = self.evaluate_model(noise_type=noise_type, noise_scale=noise_scale, plot=plot, **eval_args)
                print(noise_type, noise_scale, self.robustness[i, j])
        fname = os.path.join(EVALDATA, 'robustness_' + self.weight_file + '.npy')
        np.save(fname, self.robustness)
        if plot:
            self.plot_robustness(to_file=to_file)

    def plot_robustness(self, from_file=False, filename=None, to_file=True):
        if filename is None:
            filename = 'robustness_' + self.weight_file
        if from_file:
            robustness = np.load(os.path.join(EVALDATA, filename+'.npy'))
        else:
            robustness = self.robustness
        if robustness.shape[:1] == 4:
            indices = ['gauss+color', '+linear', '+fft', '+hyperbolic',]
        else:
            indices = ['clean', 'gauss+color', '+linear', '+fft', '+hyperbolic', '+bandpass', '+trace']
        df_cm = pd.DataFrame(robustness, index=[i for i in indices],
                             columns=[i for i in [1, 2, 4, 8]])
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True, cbar=False, vmin=robustness[robustness>0].min())
        plt.xlabel('noise to signal')
        plt.ylabel('noise types')
        if to_file:
            fname = os.path.join(self.figdir, "robust/", filename + '.jpg')
            plt.savefig(fname)
            plt.close()
        else:
            plt.show()

    def eval_real(self, loader, plot=True, to_file=True, **kwargs):
        self.metrics.reset()
        for i, (sample) in enumerate(loader):
            x, y = sample['input'].to(self.device), sample['target'].to(self.device)
            with torch.no_grad():
                y_pred = self.model(x)
                if self.problem == 'firstbreak':
                    y_pred = torch.argmax(y_pred, dim=1)  # get the most likely prediction
            print('_', end='')
        np.save(EVALDATA + 'real_x.npy', x.detach().cpu().numpy())
        np.save(EVALDATA + 'real_y.npy', y.detach().cpu().numpy())
        np.save(EVALDATA + 'real_y_pred.npy', y_pred.detach().cpu().numpy())
        if plot:
            self.plot_real(x, y, y_pred, to_file=to_file)
        return None

    def plot_real(self, *pargs, **kwargs):
        self.plot_predictions(*pargs, **kwargs, prefix='real_')

    def plot_predictions(self, x, y, y_pred, to_file=True, prefix=''):
        num = 3
        fig, axes = plt.subplots(num, 1, figsize=[9*num, 18])
        fig.suptitle('model=' + self.weight_file)
        axes[0].imshow(torchvision.utils.make_grid(x, padding=0)[0][None, ...].permute((1, 2, 0)), cmap='seismic')
        axes[0].set_title("input")
        if self.problem == 'firstbreak':
            y = y.unsqueeze(1)
            y_pred = y_pred.unsqueeze(1)
        axes[1].imshow(
            torchvision.utils.make_grid(y.float(), padding=0)[0][None, ...].permute((1, 2, 0)),
            cmap='seismic')
        axes[1].set_title("target")
        axes[2].imshow(torchvision.utils.make_grid(y_pred.float(), padding=0)[0][None, ...].permute((1, 2, 0)),
                   cmap='seismic')
        axes[2].set_title("prediction")
        if to_file:
            fname = os.path.join(self.figdir, "preds/", prefix + self.weight_file + '.jpg')
            fig.savefig(fname)
            plt.close(fig)
        else:
            fig.show()

    def save_robustness(self):
        pass

def evaluate_models(model_type, problem, **eval_args):
    for i, noise_type in enumerate(range(-1, 6)):
        for j, noise_scale in enumerate([0.25, 0.5, 1.0, 2.0]):
            fname = f'{model_type}_{problem}_noisetype_{noise_type}_noisescale_{noise_scale}_dataclip_True_pretrained_False'
            if not os.path.exists(os.path.join(METADATA, fname + '.pkl')):
                continue
            else:
                "Model not found!"
                print(fname)
            print("Model was found!")
            print(fname)
            eval = Evaluator(fname)
            eval.evaluate_robustness(**eval_args)
            print("Succeed!")
            print(fname)

def evaluate_all_models(check_before=True, **eval_args):
    fnames = []
    for model_type in ['unet', 'swin', 'restormer']:
        fnames += glob.glob(os.path.join(METADATA, model_type) + '*')
    for fname in fnames:
        fname = os.path.basename(fname)
        print(f'Analysing {fname}')
        if not os.path.exists(os.path.join(METADATA, fname)):
            continue
        output = os.path.join(EVALDATA, 'robustness_' + fname[:-4] + '.npy')
        if os.path.exists(output) and check_before:
            print(f"{fname} already exists; skip!")
            continue
        eval = Evaluator(fname)
        eval.evaluate_robustness(**eval_args)
        print(f'Model {fname} was successfully analyzed!')

if __name__ == "__main__":
    #eval.plot_robustness(from_file=True)
    # for model_type in ['swin', 'unet']:
    #     for problem in ['denoise', 'firstbreak']:
    #         for attack in ['none', 'fgsm']:
    #             for eval_attack in [None, fgsm]:
    #                 evaluate_models(model_type, problem, attack_type=attack, attack=eval_attack)
    #eval.plot_noise_palette()
    # print(eval.model_type, eval.problem)
    #evaluate_all_models(attack=fgsm)
    #######################################
    eval = Evaluator('restormer_firstbreak_noisetype_0_noisescale_2.0_dataclip_True_attack_none_pretrained_True.pkl')
    eval.evaluate_model('real')
    eval = Evaluator('restormer_firstbreak_noisetype_0_noisescale_2.0_dataclip_True_attack_fgsm_pretrained_True.pkl')
    eval.evaluate_model('real')
    eval = Evaluator('swin_firstbreak_noisetype_3_noisescale_2.0_dataclip_True_attack_none_pretrained_True.pkl')
    eval.evaluate_model('real')
    eval = Evaluator('swin_firstbreak_noisetype_3_noisescale_2.0_dataclip_True_attack_fgsm_pretrained_True.pkl')
    eval.evaluate_model('real')
    eval = Evaluator('unet_firstbreak_noisetype_3_noisescale_2.0_dataclip_True_attack_none_pretrained_False.pkl')
    eval.evaluate_model('real')
    eval = Evaluator('unet_firstbreak_noisetype_3_noisescale_2.0_dataclip_True_attack_fgsm_pretrained_False.pkl')
    eval.evaluate_model('real')
    #######################################
    # eval = Evaluator('restormer_denoise_noisetype_0_noisescale_2.0_dataclip_True_attack_none_pretrained_False.pkl')
    # eval.evaluate_model('real')
    # eval = Evaluator('restormer_denoise_noisetype_0_noisescale_2.0_dataclip_True_attack_fgsm_pretrained_False.pkl')
    # eval.evaluate_model('real')
    # eval = Evaluator('restormer_denoise_noisetype_0_noisescale_1.0_dataclip_True_attack_none_pretrained_False.pkl')
    # eval.evaluate_model('real')
    # eval = Evaluator('restormer_denoise_noisetype_0_noisescale_1.0_dataclip_True_attack_fgsm_pretrained_False.pkl')
    # eval.evaluate_model('real')
    # eval = Evaluator('restormer_denoise_noisetype_1_noisescale_1.0_dataclip_True_attack_none_pretrained_False.pkl')
    # eval.evaluate_model('real')
    # eval = Evaluator('restormer_denoise_noisetype_1_noisescale_1.0_dataclip_True_attack_fgsm_pretrained_False.pkl')
    # eval.evaluate_model('real')
    #######################################
    # eval = Evaluator('swin_firstbreak_noisetype_0_noisescale_1.0_dataclip_True_attack_none_pretrained_True.pkl')
    # eval.evaluate_model(noise_type=0, noise_scale=1.0)
    # eval = Evaluator('ibextestswin_firstbreak_noisetype_0_noisescale_1.0_dataclip_True_attack_none_pretrained_True.pkl', skip=len('ibextest'))
    # eval.evaluate_model(noise_type=0, noise_scale=1.0)
    # eval.evaluate_model(eval_type='real')
    # for eval_attack in [None, fgsm]:
    #     eval.evaluate_robustness(plot=True, attack=eval_attack)
    #eval.evaluate_model(plot=True)