import numpy as np
import torch




class ConfusionMatrix:
    """Accumulate a confusion matrix for a classification task."""

    def __init__(self, num_classes):
        self.value = 0
        self.num_classes = num_classes

    @torch.no_grad()
    def update(self, pred, true):  # doesn't allow for "ignore_index"
        """Update the confusion matrix with the given predictions."""
        unique_mapping = true.flatten() * self.num_classes + pred.flatten()
        bins = torch.bincount(unique_mapping, minlength=self.num_classes**2)
        self.value += bins.view(self.num_classes, self.num_classes)

    def reset(self):
        """Reset all accumulated values."""
        self.value = 0

    @property
    def tp(self):
        """Get the true positive samples per-class."""
        return self.value.diag()

    @property
    def fn(self):
        """Get the false negative samples per-class."""
        return self.value.sum(dim=1) - self.value.diag()

    @property
    def fp(self):
        """Get the false positive samples per-class."""
        return self.value.sum(dim=0) - self.value.diag()

    @property
    def tn(self):
        """Get the true negative samples per-class."""
        # return self.total - (self.tp + self.fn + self.fp)
        # this is the same as the above but ~slightly~ more efficient
        tp = self.value.diag()
        actual = self.value.sum(dim=1)  # tp + fn
        predicted = self.value.sum(dim=0)  # tp + fp
        # rest = actual + predicted - tp  # tp + fn + fp
        # return actual.sum() - rest
        return actual.sum() + tp - (actual + predicted)

    @property
    def count(self):  # a.k.a. actual positive class
        """Get the number of samples per-class."""
        # return self.tp + self.fn
        return self.value.sum(dim=1)

    @property
    def frequency(self):
        """Get the per-class frequency."""
        # we avoid dividing by zero using: max(denominator, 1)
        # return self.count / self.total.clamp(min=1)
        count = self.value.sum(dim=1)
        return count / count.sum().clamp(min=1)

    @property
    def total(self):
        """Get the total number of samples."""
        return self.value.sum()