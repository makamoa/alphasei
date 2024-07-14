import torch
import os 
import matplotlib.pyplot as plt
import numpy as np
from dataset.SegyDataset import SegyDataset
from dataset.NpyDataset import NpyDataset
from metrics import RegressionMetrics, ClassificationMetrics
from transformation import TransformWrapper
from loss import buildL
from torch.utils.data import DataLoader
import json
import warnings
from models import buildM
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn

device = "cpu"
if torch.cuda.is_available():
	device = torch.device("cuda")   
	print("Running on: Cuda")
else:
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine.")
    else:
        device = torch.device("mps")
        print ("Running on mps")


def dict_without_key(d, key):
    """
    Return a copy of the dictionary without the specified key.

    Args:
        d (dict): The original dictionary.
        key (str): The key to remove.

    Returns:
        dict: The dictionary without the specified key.
    """
    new_d = d.copy()
    new_d.pop(key)
    return new_d

class Clamp():
    def __call__(self, sample):
        """
        Clamp the input and target to the range [0.0, 1.0].

        Args:
            sample (dict): A sample containing 'input' and 'target'.

        Returns:
            dict: The transformed sample with clamped input and target.
        """
        sample['input'] = torch.clamp(sample['input'], 0.0, 1.0)
        sample['target'] = torch.clamp(sample['target'], 0.0, 1.0)
        return sample
    
class ToTensor(object):
    def __call__(self, sample):
        """
        Convert the input and target from numpy arrays to tensors.

        Args:
            sample (dict): A sample containing 'input' and 'target'.

        Returns:
            dict: The transformed sample with input and target as tensors.
        """
        input, target = sample['input'], sample['target']
        return {'input': torch.from_numpy(input), 'target': torch.from_numpy(target)}

    
def plot_losses(train_losses, val_losses, fname, scale='linear', title='Loss per Epoch'):
    """
    Plots the training and validation losses across epochs and saves the plot as an image file with name - fname(function argument). 

    Args:
        train_losses (list): List of training losses for each epoch.
        val_losses (list): List of validation losses for each epoch.
        fname (str): Name of the file to save the plot (without extension).

    Returns:
        None
    """

	# Create 'plots' directory if it doesn't exist
    if not os.path.isdir('plots/runs'):
        os.mkdir('plots/runs')

    # Plotting training and validation losses
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    
    # range of y-axis values
    if scale == 'log':
        plt.yscale('log')
    else:
        plt.yscale('linear')

    # Saving the plot as an image file in 'plots' directory
    plt.savefig("./plots/" + fname + ".png")
    
def get_model (model, config):
    """
    Given a model name and a configuration file, return the model object.
    """
    with open (config, 'r') as f:
        config = json.load(f)
        return buildM.build_model(model, config)

def train_config(model, config_file):
    """
    Given a configuration file, return the configuration dictionary.
    """
    with open(config_file, 'r') as f:
        config = json.load(f)

    loss = get_loss(config)
    opt = get_optimizer(model, config)
    sch = get_scheduler(opt, config)
    metrics = get_metrics(config)
    (td, tl) = get_transforms(config)
    
    # if epochs is not specified
    if 'epochs' not in config:
        warnings.warn("Number of epochs not specified. Defaulting to 10.")
        epochs = 10
    else:
        epochs = config['epochs']
    
    return loss, epochs, opt, sch, metrics, (td, tl)

def get_loss(config):
    """
    Given a configuration dictionary, return the loss function.
    """
    
    ls = config['loss']
    return buildL.build_loss(ls['type'], ls)

def get_optimizer(model, config):
    """
    Given a model and a configuration dictionary, return the optimizer.
    
    Args:
        model (torch.nn.Module): The model to optimize.
        config (dict): A dictionary containing optimizer configuration.
    
    Available optimizers:
        - Adam
        - AdamW
        - SGD
        - RMSprop
        - Adagrad
        - Adadelta
        
    """
    
    optimizer_name = config['optimizer'].lower()
    
    optimizer_params = dict(config.get('optimizer_params', {}))
    
    lr = optimizer_params.pop('lr', 0.001)
    
    valid_optimizers = {
        'adam': optim.Adam,
        'adamw': optim.AdamW, 
        'sgd': optim.SGD,
        'rmsprop': optim.RMSprop,
        'adagrad': optim.Adagrad,
        'adadelta': optim.Adadelta
    }
    assert optimizer_name in valid_optimizers, f"Invalid optimizer name. Choose from {list(valid_optimizers.keys())}"
    
    optimizer_class = valid_optimizers[optimizer_name]
    
    if optimizer_name == 'sgd':
        momentum = optimizer_params.pop('momentum', 0.9)
        return optimizer_class(model.parameters(), lr=lr, momentum=momentum, **optimizer_params)
    else:
        return optimizer_class(model.parameters(), lr=lr, **optimizer_params)

def get_scheduler(optimizer, config):
    """
    Given an optimizer and a configuration dictionary, return the scheduler.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer to schedule.
        config (dict): A dictionary containing scheduler configuration.
    
    Avialable schedulers:
        - StepLR
        - MultiStepLR
        - ExponentialLR
        - CosineAnnealingLR
        - CyclicLR
    
    Returns:
        torch.optim.lr_scheduler._LRScheduler: The configured scheduler.
    """
    scheduler_name = config['scheduler'].lower()
    scheduler_params = dict(config.get('scheduler_params', {}))
    
    valid_schedulers = {
        'steplr': lr_scheduler.StepLR,
        'multisteplr': lr_scheduler.MultiStepLR,
        'exponentiallr': lr_scheduler.ExponentialLR,
        'cosineannealinglr': lr_scheduler.CosineAnnealingLR,
        'cycliclr': lr_scheduler.CyclicLR
    }
    
    if scheduler_name not in valid_schedulers:
        raise ValueError(f"Invalid scheduler name. Choose from {list(valid_schedulers.keys())}")
    
    scheduler_class = valid_schedulers[scheduler_name]
    
    if scheduler_name == 'steplr':
        step_size = scheduler_params.pop('step_size', 10)
        gamma = scheduler_params.pop('gamma', 0.1)
        return scheduler_class(optimizer, step_size=step_size, gamma=gamma, **scheduler_params)
    elif scheduler_name == 'multisteplr':
        milestones = scheduler_params.pop('milestones', [30, 60, 90])
        gamma = scheduler_params.pop('gamma', 0.1)
        return scheduler_class(optimizer, milestones=milestones, gamma=gamma, **scheduler_params)
    elif scheduler_name == 'exponentiallr':
        gamma = scheduler_params.pop('gamma', 0.95)
        return scheduler_class(optimizer, gamma=gamma, **scheduler_params)
    elif scheduler_name == 'cosineannealinglr':
        T_max = scheduler_params.pop('T_max', 10)
        eta_min = scheduler_params.pop('eta_min', 0)
        return scheduler_class(optimizer, T_max=T_max, eta_min=eta_min, **scheduler_params)
    elif scheduler_name == 'cycliclr':
        base_lr = scheduler_params.pop('base_lr', 0.001)
        max_lr = scheduler_params.pop('max_lr', 0.01)
        step_size_up = scheduler_params.pop('step_size_up', 2000)
        return scheduler_class(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up, **scheduler_params)
    
def get_metrics(config):
    """
    Given a configuration dictionary, return the metrics.
    """
    
    metric_type = config['metrics']
    config = config.get('metrics_params', {})
    if metric_type == 'regression':
        if config.get('data_range', None) is None:
            raise ValueError("Data Range not specified for regression metrics.")
        if config.get('full', True):
            data_range = config['data_range']
            (win_sz, normalize) = RegressionMetrics.RegressionMetricsFull(data_range).__init__.__defaults__
            win_sz = config.get('window_size', win_sz)
            normalize = config.get('normalize', normalize)
            return RegressionMetrics.RegressionMetricsFull(data_range, win_sz, normalize)
        else:
            data_range = config['data_range']
            (win_sz, normalize) = RegressionMetrics.RegressionMetricsLight(data_range).__init__.__defaults__
            win_sz = config.get('window_size', win_sz)
            normalize = config.get('normalize', normalize)
            return RegressionMetrics.RegressionMetricsLight(data_range, win_sz, normalize)
    elif metric_type == 'classification':
        if config.get('num_classes', None) is None:
            raise ValueError("Number of classes not specified for classification metrics.")
        num_classes = config['num_classes']
        return ClassificationMetrics.ClassificationMetrics(num_classes)
    else: 
        raise ValueError(f"Invalid metric type. Choose from ['regression', 'classification']")

def get_transforms(config):
    """
    Given a configuration dictionary, return the data transformations and label transformations.
    """
    dt_config = config.get('data_transforms', None)
    lt_config = config.get('label_transforms', None)
    
    dt = []
    
    if dt_config is None:
        warnings.warn("Data Transforms not specified. Defaulting to ToTensor.")
        dt = []
    else: 
        dt = TransformWrapper.build_transforms(dt_config)
    
    lt = []
    if lt_config is None:
        warnings.warn("Label Transforms not specified. Defaulting to ToTensor.")
        lt = []
    else: 
        lt = TransformWrapper.build_transforms(lt_config)
    
    return dt, lt
            
def get_loaders(type, config, td, tl):
    """
    Given a configuration file, load the dataset and return a train and validation DataLoader object.
    """
    with open(config, 'r') as f:
        config = json.load(f)
    
    
    if type == 'segy':
        return create_segy_loader(config, td, tl)
    elif type == 'npy':
        return create_npy_loader(config, td, tl)
    
def create_segy_loader(config, td ,tl):
    """
    Given a configuration dictionary, load the segy dataset and return the DataLoader object.
    """
    train_dataset = SegyDataset.from_config(config['train'])
    val_dataset = SegyDataset.from_config(config['val'])
    
    train_dataset.add_transforms(td, tl)
    val_dataset.add_transforms(td, tl)
    
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    
    if 'collate' in config and 'collate_args' in config:
        collate_args = config['collate_args']
        collate_train = train_dataset.create_collate_fn(config['collate'], **collate_args)
        collate_val = val_dataset.create_collate_fn(config['collate'], **collate_args)
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_train, num_workers=num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_val, num_workers=num_workers)
    else: 
        warnings.warn("Collate function not specified. Defaulte can't handle variable length sequences.")
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_dataloader, val_dataloader

def create_npy_loader(config, td, tl):
    """
    Given a configuration dictionary, load the npy dataset and return the DataLoader object.
    """
    # print (config, config['train'], config['val'])
    train_dataset = NpyDataset.from_config(config['train'])
    val_dataset = NpyDataset.from_config(config['val'])
    
    train_dataset.add_transforms(td, tl)
    val_dataset.add_transforms(td, tl)
    
    num_workers = config['num_workers']
    batch_size = config['batch_size']
    
    print(type(train_dataset), type(val_dataset))
    
    print (train_dataset, val_dataset)
    
    if 'collate' in config and 'collate_args' in config:
        collate_args = config['collate_args']
        collate_train = train_dataset.create_collate_fn(config['collate'], **collate_args)
        collate_val = val_dataset.create_collate_fn(config['collate'], **collate_args)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_train, num_workers=num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_val, num_workers=num_workers)
    else: 
        warnings.warn("Collate function not specified. Defaulte can't handle variable length sequences.")
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
    return train_dataloader, val_dataloader