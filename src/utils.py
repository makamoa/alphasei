import torch
import os 
import matplotlib.pyplot as plt
import numpy as np
from dataset import SegyDataset, NpyDataset
from models import *
from metrics import *
from loss import *
import torch.utils.data.dataloader as DataLoader
import json
import warnings

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

class Scale():
    def __init__(self, problem='regr'):
        """
        Initialize the Scale transform.

        Args:
            problem (str): The problem type ('regr' for regression, 'segment' for segmentation). Defaults to 'regr'.
        """
        self.problem = problem

    def __call__(self, sample):
        """
        Scale the input and target values to the range [0, 1].

        Args:
            sample (dict): A sample containing 'input' and 'target'.

        Returns:
            dict: The transformed sample with scaled input and target.
        """
        sample['input'] = sample['input'] / 255.
        if self.problem == 'regr':
            sample['target'] = sample['target'] / 255.
        return sample
    
class ChangeType():
    def __init__(self, problem='regr'):
        """
        Initialize the ChangeType transform.

        Args:
            problem (str): The problem type ('regr' for regression, 'segment' for segmentation). Defaults to 'regr'.
        """
        self.problem = problem

    def __call__(self, sample):
        """
        Change the data type of the input and target.

        Args:
            sample (dict): A sample containing 'input' and 'target'.

        Returns:
            dict: The transformed sample with changed data types.
        """
        sample['input'] = sample['input'].astype(np.float32)
        if self.problem == 'regr':
            sample['target'] = sample['target'].astype(np.float32)
        else:
            sample['target'] = sample['target'].astype(np.int)
        return sample
    
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
    if not os.path.isdir('plots'):
        os.mkdir('plots')

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
    Given a model name and a configuration dictionary, return the model object.
    """
    valid_models = []
    assert model in valid_models, f"Invalid model name. Choose from {valid_models}"
    
    # TODO: Initialize the model and return it
    pass

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
    
    valid_losses = []
    assert config['loss'] in valid_losses, f"Invalid loss name. Choose from {valid_losses}"
    
    # TODO: Initialize the loss function and return it
    pass

def get_optimizer(model, config):
    """
    Given a model and a configuration dictionary, return the optimizer.
    """
    valid_optimizers = []
    assert config['optimizer'] in valid_optimizers, f"Invalid optimizer name. Choose from {valid_optimizers}"
    
    # TODO: Initialize the optimizer and return it
    pass

def get_scheduler(optimizer, config):
    """
    Given an optimizer and a configuration dictionary, return the scheduler.
    """
    valid_schedulers = []
    assert config['scheduler'] in valid_schedulers, f"Invalid scheduler name. Choose from {valid_schedulers}"
    
    # TODO: Initialize the scheduler and return it
    pass

def get_metrics(config):
    """
    Given a configuration dictionary, return the metrics.
    """
    valid_metrics = []
    # check if all the metrics are valid
    for metric in config['metrics']:
        assert metric in valid_metrics, f"Invalid metric name. Choose from {valid_metrics}"
    
    # TODO: Initialize the metrics and return them
    pass

def get_transforms(config):
    """
    Given a configuration dictionary, return the data transformations and label transformations.
    """
    valid_transforms = []
    # check if all the transforms are valid
    for transform in config['transforms']:
        assert transform in valid_transforms, f"Invalid transform name. Choose from {valid_transforms}"
            
    
def get_loaders(type, config, td, tl, **kwargs):
    """
    Given a configuration file, load the dataset and return a train and validation DataLoader object.
    """
    with open(config, 'r') as f:
        config = json.load(f)
    
    if type == 'segy':
        return create_segy_loader(config, td, tl, **kwargs)
    elif type == 'npy':
        return create_npy_loader(config, td, tl, **kwargs)
    
    
def create_segy_loader(config, td ,tl, **kwargs):
    """
    Given a configuration dictionary, load the segy dataset and return the DataLoader object.
    """
    train_dataset = SegyDataset.from_config(config['train'])
    val_dataset = SegyDataset.from_config(config['val'])
    
    train_dataset.add_transforms(td, tl)
    val_dataset.add_transforms(td, tl)
    
    collate_train = train_dataset.create_collate_fn(config['collate'], **kwargs)
    collate_val = val_dataset.create_collate_fn(config['collate'], **kwargs)
    
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_train, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_val, num_workers=num_workers)
    
    return train_dataloader, val_dataloader

def create_npy_loader(config, td, tl, **kwargs):
    """
    Given a configuration dictionary, load the npy dataset and return the DataLoader object.
    """
    train_dataset = NpyDataset.from_config(config['train'])
    val_dataset = NpyDataset.from_config(config['val'])
    
    train_dataset.add_transforms(td, tl)
    val_dataset.add_transforms(td, tl)
    
    collate_train = train_dataset.create_collate_fn(config['collate'], **kwargs)
    collate_val = val_dataset.create_collate_fn(config['collate'], **kwargs)
    
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_train, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_val, num_workers=num_workers)
    
    return train_dataloader, val_dataloader