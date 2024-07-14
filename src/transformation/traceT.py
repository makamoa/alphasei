import torch

"""
A collection of trace (1D) transformation functions.
In this module, we define the following functions:
- `normalize`: Normalize the input tensor.
- `rescale`: Rescale the input tensor.
- `quantile`: Quantile normalize the input tensor.
- `clip`: Clip the input tensor.
- `log`: Apply the logarithm to the input tensor.
- `exp`: Apply the exponential function to the input tensor.
- `sqrt`: Apply the square root to the input tensor.
- `square`: Apply the square function to the input tensor.
"""

def normalize(tensor, mean=None, std=None):
    """Normalize the input tensor.
    Args:
        tensor (torch.Tensor): The input tensor.
        mean (float or torch.Tensor): The mean value.
        std (float or torch.Tensor): The standard deviation.
    Returns:
        torch.Tensor: The normalized tensor.
    """
    if mean is None:
        mean = tensor.mean()
    if std is None:
        std = tensor.std()
    return (tensor - mean) / std

def rescale (tensor, min = 0, max = 1):
    """Rescale the input tensor.
    Args:
        tensor (torch.Tensor): The input tensor.
        min (float): The minimum value.
        max (float): The maximum value.
    Returns:
        torch.Tensor: The rescaled tensor.
    """
    return min + (max - min) * (tensor - tensor.min()) / (tensor.max() - tensor.min())

def clib (tensor, min = None, max = None):
    """Clip the input tensor.
    Args:
        tensor (torch.Tensor): The input tensor.
        min (float): The minimum value.
        max (float): The maximum value.
    Returns:
        torch.Tensor: The clipped tensor.
    """
    if min is not None:
        tensor = torch.clamp(tensor, min = min)
    if max is not None:
        tensor = torch.clamp(tensor, max = max)
    return tensor

def log(tensor, eps=1e-6):
    """Apply the logarithm to the input tensor.
    Args:
        tensor (torch.Tensor): The input tensor.
        eps (float): The epsilon value.
    Returns:
        torch.Tensor: The transformed tensor.
    """
    return torch.log(tensor + eps)

def exp(tensor):
    """Apply the exponential function to the input tensor.
    Args:
        tensor (torch.Tensor): The input tensor.
    Returns:
        torch.Tensor: The transformed tensor.
    """
    return torch.exp(tensor)

def sqrt(tensor):
    """Apply the square root to the input tensor.
    Args:
        tensor (torch.Tensor): The input tensor.
    Returns:
        torch.Tensor: The transformed tensor.
    """
    return torch.sqrt(tensor)

def square(tensor):
    """Apply the square function to the input tensor.
    Args:
        tensor (torch.Tensor): The input tensor.
    Returns:
        torch.Tensor: The transformed tensor.
    """
    return tensor ** 2

def quantile(tensor, q=0.5):
    """Quantile normalize the input tensor.
    Args:
        tensor (torch.Tensor): The input tensor.
        q (float): The quantile value.
    Returns:
        torch.Tensor: The quantile normalized tensor.
    """
    return (tensor - tensor.quantile(q)) / tensor.quantile(1 - q)

