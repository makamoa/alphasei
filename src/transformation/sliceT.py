import torch 
import torchvision.transforms.functional as F




"""
A collection of slice (2D) transformation functions.
In this module, we define the following functions:
- `crop`: Crop the input tensor.
- `flip`: Flip the input tensor.
- `rotate`: Rotate the input tensor.
- `translate`: Translate the input tensor.
- `scale`: Scale the input tensor.
- `normalize`: Normalize the input tensor.
"""

def crop(tensor, x1, y1, x2, y2):
    """Crop the input tensor.
    Args:
        tensor (torch.Tensor): The input tensor.
        x1 (int): The x-coordinate of the top-left corner.
        y1 (int): The y-coordinate of the top-left corner.
        x2 (int): The x-coordinate of the bottom-right corner.
        y2 (int): The y-coordinate of the bottom-right corner.
    Returns:
        torch.Tensor: The cropped tensor.
    """
    return tensor[..., y1:y2, x1:x2]

def flip(tensor, dim):
    """Flip the input tensor.
    Args:
        tensor (torch.Tensor): The input tensor.
        dim (int): The dimension to flip.
    Returns:
        torch.Tensor: The flipped tensor.
    """
    return torch.flip(tensor, dims=(dim,))

def rotate(tensor, angle):
    """Rotate the input tensor.
    Args:
        tensor (torch.Tensor): The input tensor.
        angle (float): The angle of rotation.
        mode (str): The mode of interpolation.
    Returns:
        torch.Tensor: The rotated tensor.
    """
    return F.affine(tensor, angle, translate=[0, 0], scale=1.0, shear=0.0)

def translate(tensor, dx, dy):
    """Translate the input tensor.
    Args:
        tensor (torch.Tensor): The input tensor.
        dx (int): The horizontal shift.
        dy (int): The vertical shift.
        mode (str): The mode of interpolation.
    Returns:
        torch.Tensor: The translated tensor.
    """
    return F.affine(tensor, angle=0.0, translate=[dx, dy], scale=1.0, shear=0.0)

def scale(tensor, factor):
    """Scale the input tensor.
    Args:
        tensor (torch.Tensor): The input tensor.
        factor (float): The scaling factor.
        mode (str): The mode of interpolation.
    Returns:
        torch.Tensor: The scaled tensor.
    """
    return F.affine(tensor, angle=0.0, translate=[0, 0], scale=factor, shear=0.0)

def normalize(tensor, mean = None, std=  None):
    """Normalize the input tensor.
    Args:
        tensor (torch.Tensor): The input tensor.
        mean (float): The mean value.
        std (float): The standard deviation value.
    Returns:
        torch.Tensor: The normalized tensor.
    """
    if mean is None:
        mean = tensor.mean()
    if std is None:
        std = tensor.std()
    
    return (tensor - mean) / std