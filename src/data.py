import os
from torch.utils.data import Dataset, DataLoader, random_split
from utils import *
import yaml

# Load configuration file
with open(os.path.join("../config/", "global_config.yml"), 'r') as stream:
    data_loaded = yaml.safe_load(stream)
SEISMICROOT = data_loaded['SEISMICROOT']
DERAINROOT = data_loaded['DERAINROOT']
DERAINTRAIN = os.path.join(SEISMICROOT, 'train/Rain13K')
DERAINTEST = os.path.join(SEISMICROOT, 'test/Rain13K')
SEISMICDIR = os.path.join(SEISMICROOT, 'data/')

# Base dataset loader class
class BaseLoader(Dataset):
    def read_input(self, idx):
        pass

    def read_target(self, idx):
        pass

# Loader for first break data
class FirstBreakLoader(BaseLoader):
    def __init__(self, rootdir, transform=None):
        """
        Initialize the FirstBreakLoader.

        Args:
            rootdir (str): Root directory for the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.rootdir = rootdir
        self.inputdir = os.path.join(self.rootdir, 'input/')
        self.targetdir = os.path.join(self.rootdir, 'target/')
        self.inputs = sorted(os.listdir(self.inputdir))
        self.targets = sorted(os.listdir(self.targetdir))
        self.transform = transform
        self.class_names = ['empty', 'wave']

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        assert len(self.targets) == len(self.inputs)
        return len(self.inputs)

    def read_input(self, idx):
        """
        Read an input sample.

        Args:
            idx (int): Index of the sample to read.

        Returns:
            np.ndarray: The input sample.
        """
        image = np.load(os.path.join(self.inputdir, self.inputs[idx]))
        return image[..., None]

    def read_target(self, idx):
        """
        Read a target sample.

        Args:
            idx (int): Index of the sample to read.

        Returns:
            np.ndarray: The target sample.
        """
        return np.load(os.path.join(self.targetdir, self.targets[idx]))

    def __getitem__(self, idx):
        """
        Get a sample.

        Args:
            idx (int): Index of the sample to get.

        Returns:
            dict: A dictionary containing the input and target samples.
        """
        image = self.read_input(idx)
        target = self.read_target(idx)
        sample = {'input': image, 'target': target}
        return self.transform(sample) if self.transform else sample

# Loader for denoise data
class DenoiseLoader(FirstBreakLoader):
    def __init__(self, *pargs, **kwargs):
        """
        Initialize the DenoiseLoader.

        Inherits from FirstBreakLoader.
        """
        super(DenoiseLoader, self).__init__(*pargs, **kwargs)
        self.targetdir = self.inputdir
        self.targets = self.inputs
        self.transform = self.transform

    def read_target(self, idx):
        """
        Read a target sample with an additional dimension.

        Args:
            idx (int): Index of the sample to read.

        Returns:
            np.ndarray: The target sample with an added dimension.
        """
        return super(DenoiseLoader, self).read_target(idx)[..., None]

# Loader for derain data
class DerainLoader(BaseLoader):
    def __init__(self, rootdir, transform=None):
        """
        Initialize the DerainLoader.

        Args:
            rootdir (str): Root directory for the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.rootdir = rootdir
        self.inputdir = os.path.join(self.rootdir, 'input/')
        self.targetdir = os.path.join(self.rootdir, 'target/')
        self.inputs = os.listdir(self.inputdir)
        self.targets = os.listdir(self.targetdir)
        self.transform = transform

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        assert len(self.targets) == len(self.inputs)
        return len(self.inputs)

    def read_input(self, idx):
        """
        Read an input sample.

        Args:
            idx (int): Index of the sample to read.

        Returns:
            np.ndarray: The input sample.
        """
        image = io.imread(os.path.join(self.inputdir, self.inputs[idx]))
        return image

    def read_target(self, idx):
        """
        Read a target sample.

        Args:
            idx (int): Index of the sample to read.

        Returns:
            np.ndarray: The target sample.
        """
        return io.imread(os.path.join(self.targetdir, self.targets[idx]))

    def __getitem__(self, idx):
        """
        Get a sample.

        Args:
            idx (int): Index of the sample to get.

        Returns:
            dict: A dictionary containing the input and target samples.
        """
        image = self.read_input(idx)
        target = self.read_target(idx)
        sample = {'input': image, 'target': target}
        return self.transform(sample) if self.transform else sample

# Function to get derain dataset
def get_derain_dataset(rootdir=None,
                       min_size=256, crop_size=(224, 224), target_size=(224, 224), normalize=False,
                       noise_transforms=[]):
    """
    Get the derain dataset.

    Args:
        rootdir (str, optional): Root directory for the dataset. Defaults to DERAINTRAIN.
        min_size (int): Minimum size for resizing.
        crop_size (tuple): Size for random crop.
        target_size (tuple): Size for resizing after crop.
        normalize (bool): Whether to normalize the images.
        noise_transforms (list): List of additional transforms for noise.

    Returns:
        DerainLoader: The derain dataset loader.
    """
    if rootdir is None:
        rootdir = DERAINTRAIN
    transforms_ = []
    ImageNormalize = [InputNormalize(imagenet_mean, imagenet_std)]
    ImageChangeType = [ChangeType(), Scale()]
    transforms_ += [MinResize(min_size=min_size)]
    transforms_ += ImageChangeType
    transforms_ += [RandomCrop(crop_size)]
    if crop_size != target_size:
        transforms_ += [Resize(target_size)]
    transforms_ += noise_transforms
    transforms_ += [FlipChannels(), ToTensor()]
    if normalize:
        transforms_ += ImageNormalize
    return DerainLoader(rootdir, transform=transforms.Compose(transforms_))

# Function to get first break dataset
def get_first_break_dataset(rootdir=None,
                            target_size=(224, 224),
                            noise_transforms=[]):
    """
    Get the first break dataset.

    Args:
        rootdir (str, optional): Root directory for the dataset. Defaults to SEISMICDIR.
        target_size (tuple): Size for resizing.
        noise_transforms (list): List of additional transforms for noise.

    Returns:
        FirstBreakLoader: The first break dataset loader.
    """
    if rootdir is None:
        rootdir = SEISMICDIR
    transforms_ = []
    transforms_ += noise_transforms
    transforms_ += [ChangeType(problem='segment')]
    transforms_ += [ScaleNormalize('input')]
    transforms_ += [FlipChannels(only_input=True), ToTensor()]
    return FirstBreakLoader(rootdir, transform=transforms.Compose(transforms_))

# Loader for real seismic data
class RealDataLoader(BaseLoader):
    def __init__(self, rootdir, transform=None):
        """
        Initialize the RealDataLoader.

        Args:
            rootdir (str): Root directory for the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.rootdir = rootdir
        self.inputdir = os.path.join(self.rootdir, 'input/')
        self.targetdir = os.path.join(self.rootdir, 'target/')
        self.inputs = os.listdir(self.inputdir)
        self.targets = os.listdir(self.targetdir)
        self.transform = transform
        self.class_names = ['empty', 'wave']

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        assert len(self.targets) == len(self.inputs)
        return len(self.inputs)

    def read_input(self, idx):
        """
        Read an input sample.

        Args:
            idx (int): Index of the sample to read.

        Returns:
            np.ndarray: The input sample.
        """
        image = np.load(os.path.join(self.inputdir, self.inputs[idx]))
        return image[..., None]

    def read_target(self, idx):
        """
        Read a target sample.

        Args:
            idx (int): Index of the sample to read.

        Returns:
            np.ndarray: The target sample.
        """
        image = np.load(os.path.join(self.targetdir, self.targets[idx]))
        return image[..., None]

    def __getitem__(self, idx):
        """
        Get a sample.

        Args:
            idx (int): Index of the sample to get.

        Returns:
            dict: A dictionary containing the input and target samples.
        """
        image = self.read_input(idx)
        target = self.read_target(idx)
        sample = {'input': image, 'target': target}
        return self.transform(sample) if self.transform else sample

# Function to get denoise dataset
def get_denoise_dataset(rootdir=None,
                       noise_transforms=[]):
    """
    Get the denoise dataset.

    Args:
        rootdir (str, optional): Root directory for the dataset. Defaults to SEISMICDIR.
        noise_transforms (list): List of additional transforms for noise.

    Returns:
        DenoiseLoader: The denoise dataset loader.
    """
    if rootdir is None:
        rootdir = SEISMICDIR
    transforms_ = []
    transforms_ += noise_transforms
    transforms_ += [ChangeType()]
    transforms_ += [ScaleNormalize('input'), ScaleNormalize('target')]
    transforms_ += [FlipChannels(), ToTensor()]
    return DenoiseLoader(rootdir, transform=transforms.Compose(transforms_))

# Function to get real dataset
def get_real_dataset(rootdir="../realdata",
                            target_size=(224, 224),
                            noise_transforms=[]):
    """
    Get the real dataset.

    Args:
        rootdir (str, optional): Root directory for the dataset. Defaults to "../realdata".
        target_size (tuple): Size for resizing.
        noise_transforms (list): List of additional transforms for noise.

    Returns:
        RealDataLoader: The real dataset loader.
    """
    transforms_ = []
    transforms_ += noise_transforms
    transforms_ += [Resize(target_size=target_size)]
    transforms_ += [ScaleNormalize('input')]
    transforms_ += [FlipChannels(), ToTensor()]
    return RealDataLoader(rootdir, transform=transforms.Compose(transforms_))

# Function to get dataset based on type
def get_dataset(dtype, *pargs, **kwargs):
    """
    Get dataset based on type.

    Args:
        dtype (str): Type of the dataset ('derain', 'firstbreak', 'denoise', 'real').
        *pargs: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Dataset: The dataset loader.

    Raises:
        ValueError: If the dataset type is unknown.
    """
    if dtype == 'derain':
        dataset = get_derain_dataset(*pargs, **kwargs)
    elif dtype == 'firstbreak':
        dataset = get_first_break_dataset(*pargs, **kwargs)
    elif dtype == 'denoise':
        dataset = get_denoise_dataset(*pargs, **kwargs)
    elif dtype == 'real':
        dataset = get_real_dataset(*pargs, **kwargs)
    else:
        raise ValueError("Unknown Dataset Type")
    return dataset

# Function to split dataset into training and validation sets
def get_train_val_dataset(dataset, valid_split=0.1, **kwargs):
    """
    Split dataset into training and validation sets.

    Args:
        dataset (Dataset): The dataset to split.
        valid_split (float, optional): Fraction of the dataset to be used as validation set. Defaults to 0.1.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        tuple: Training and validation datasets.
    """
    train_size = int((1 - valid_split) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], **kwargs)
    return train_dataset, val_dataset
