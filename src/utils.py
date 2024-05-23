import torch
import numpy as np
from skimage import io, transform
from torchvision import transforms
from skimage.transform import AffineTransform, warp

# Mean and standard deviation for normalization
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

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

class RandomHorizontalFlip():
    def __init__(self, prob=0.5):
        """
        Initialize the RandomHorizontalFlip transform.

        Args:
            prob (float): Probability of flipping the image. Defaults to 0.5.
        """
        self.prob = prob

    def __call__(self, sample):
        """
        Apply the transform to the sample.

        Args:
            sample (dict): A sample containing 'input' and 'target'.

        Returns:
            dict: The transformed sample.
        """
        x = sample['input']
        y = sample['target']
        if np.random.rand(1) < self.prob:
            sample['input'] = torch.flip(x, dims=(-1,))
            sample['target'] = torch.flip(y, dims=(-1,))
        return sample

class ScaleNormalize():
    def __init__(self, type='input'):
        """
        Initialize the ScaleNormalize transform.

        Args:
            type (str): Type of normalization ('input' or 'target'). Defaults to 'input'.
        """
        self.type = type

    def __call__(self, sample):
        """
        Apply the transform to the sample.

        Args:
            sample (dict): A sample containing 'input' and 'target'.

        Returns:
            dict: The transformed sample.
        """
        x = sample[self.type]
        if x.dtype == int:
            return sample
        x /= np.abs(x).max()
        return sample

class RandomShift():
    def __init__(self, mean=1, std=1e-5, shift_mean=0, shift_std=1e-3):
        """
        Initialize the RandomShift transform.

        Args:
            mean (float): Mean value for scaling. Defaults to 1.
            std (float): Standard deviation for scaling. Defaults to 1e-5.
            shift_mean (float): Mean value for shifting. Defaults to 0.
            shift_std (float): Standard deviation for shifting. Defaults to 1e-3.
        """
        self.mean = mean
        self.std = std
        self.shift_mean = shift_mean
        self.shift_std = shift_std

    def __call__(self, sample):
        """
        Apply the transform to the sample.

        Args:
            sample (dict): A sample containing 'input' and 'target'.

        Returns:
            dict: The transformed sample.
        """
        image, target = sample['input'], sample['target']
        wx, wy = image.shape[:2]
        scale = np.random.uniform(low=self.mean - np.sqrt(3) * self.std, high=self.mean + np.sqrt(3) * self.std)
        shift = scale * np.random.uniform(low=self.shift_mean - np.sqrt(3) * self.shift_std, high=self.shift_mean + np.sqrt(3) * self.shift_std)
        scaling = AffineTransform(scale=(1, scale), translation=(0, wy - scale*wy - 1))
        scaled = warp(image, scaling, mode='constant', preserve_range=True)
        scaled_target = warp(target, scaling, mode='constant', preserve_range=True)
        shift = AffineTransform(translation=(0, shift))
        shifted = warp(scaled, shift, mode='edge', preserve_range=True)
        shifted_target = warp(scaled_target, shift, mode='edge', preserve_range=True)
        sample['input'] = shifted.astype(image.dtype)
        sample['target'] = shifted_target.astype(target.dtype)
        return sample

class BaseNormalize():
    def __init__(self, mean, std):
        """
        Initialize the BaseNormalize transform.

        Args:
            mean (array-like): Mean values for normalization.
            std (array-like): Standard deviation values for normalization.
        """
        self.normalize = transforms.Normalize(mean, std)

    def __call__(self, x):
        """
        Apply the normalization to the input.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Normalized tensor.
        """
        return self.normalize(x)

class InputNormalize(BaseNormalize):
    def __call__(self, sample):
        """
        Apply the normalization to the input sample.

        Args:
            sample (dict): A sample containing 'input' and 'target'.

        Returns:
            dict: The transformed sample with normalized input.
        """
        x = sample['input']
        sample['input'] = super().__call__(x)
        return sample

class TargetNormalize(BaseNormalize):
    def __call__(self, sample):
        """
        Apply the normalization to the target sample.

        Args:
            sample (dict): A sample containing 'input' and 'target'.

        Returns:
            dict: The transformed sample with normalized target.
        """
        x = sample['target']
        sample['target'] = super().__call__(x)
        return sample

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

class FlipChannels(object):
    def __init__(self, only_input=False):
        """
        Initialize the FlipChannels transform.

        Args:
            only_input (bool): Whether to flip only the input channels. Defaults to False.
        """
        self.only_input = only_input

    def __call__(self, sample):
        """
        Flip the channels of the input (and optionally the target).

        Args:
            sample (dict): A sample containing 'input' and 'target'.

        Returns:
            dict: The transformed sample with flipped channels.
        """
        input, target = sample['input'], sample['target']
        input = input.transpose((2, 0, 1))
        if not self.only_input:
            target = target.transpose((2, 0, 1))
        return {'input': input, 'target': target}

class Resize():
    def __init__(self, target_size=(256, 256)):
        """
        Initialize the Resize transform.

        Args:
            target_size (tuple): The target size for resizing. Defaults to (256, 256).
        """
        self.target_size = target_size

    def __call__(self, sample):
        """
        Resize the input and target to the target size.

        Args:
            sample (dict): A sample containing 'input' and 'target'.

        Returns:
            dict: The transformed sample with resized input and target.
        """
        sample['input'] = transform.resize(sample['input'], self.target_size, preserve_range=True)
        sample['target'] = transform.resize(sample['target'], self.target_size, preserve_range=True)
        return sample

class MinResize(Resize):
    def __init__(self, min_size=256):
        """
        Initialize the MinResize transform.

        Args:
            min_size (int): The minimum size for resizing. Defaults to 256.
        """
        self.min_size = min_size
        super().__init__()

    def __call__(self, sample):
        """
        Resize the input and target to ensure the minimum size.

        Args:
            sample (dict): A sample containing 'input' and 'target'.

        Returns:
            dict: The transformed sample with resized input and target.
        """
        wx0, wy0, _ = sample['input'].shape
        min_dim = min(wx0, wy0)
        k = 1
        if min_dim < self.min_size:
            k = self.min_size / min_dim
        self.target_size = k * np.array((wx0, wy0))
        return super().__call__(sample)

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

class RandomCrop():
    def __init__(self, target_size=(224, 224), edge=5):
        """
        Initialize the RandomCrop transform.

        Args:
            target_size (tuple): The target size for the crop. Defaults to (224, 224).
            edge (int): The edge padding for the crop. Defaults to 5.
        """
        self.target_size = target_size
        self.edge = edge

    def __call__(self, sample):
        """
        Randomly crop the input and target to the target size.

        Args:
            sample (dict): A sample containing 'input' and 'target'.

        Returns:
            dict: The transformed sample with randomly cropped input and target.
        """
        wx, wy = self.target_size
        wx0, wy0, _ = sample['target'].shape
        try:
            center_x = np.random.randint(self.edge + wx // 2, wx0 - self.edge - wx // 2)
            center_y = np.random.randint(self.edge + wy // 2, wy0 - self.edge - wy // 2)
        except:
            raise ValueError('Error in cropping: invalid sample shape', sample['target'].shape)
        crop_x_0 = center_x - wx // 2
        crop_x_1 = center_x + wx // 2
        crop_y_0 = center_y - wy // 2
        crop_y_1 = center_y + wy // 2
        sample['input'] = sample['input'][crop_x_0:crop_x_1, crop_y_0:crop_y_1]
        sample['target'] = sample['target'][crop_x_0:crop_x_1, crop_y_0:crop_y_1]
        return sample
