import torch
import torch.nn as nn
import torch.nn.functional as F

def build_loss(loss, config):
    """
    Given a loss name and a configuration dictionary, return the loss function.
    """
    
    loss = loss.lower()
    
    valid_losses = ['l1', 'l2', 'huber_loss', 'crossentropy']
    assert loss in valid_losses, f"Invalid loss name. Choose from {valid_losses}"
    
    if loss == 'l1':
        return nn.L1Loss()
    elif loss == 'l2':
        return nn.MSELoss()
    elif loss == 'huber_loss':
        return nn.SmoothL1Loss()
    elif loss == 'crossentropy':
        return nn.CrossEntropyLoss()
    