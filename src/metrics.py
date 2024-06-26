import numpy as np
import pandas as pd

class MetricsBase(object):
    def __init__(self, num_classes, names):
        pass

    def pixel_accuracy(self):
        raise NotImplementedError

    def pixel_accuracy_class(self):
        raise NotImplementedError

    def mean_intersection_over_union(self):
        raise NotImplementedError

    def frequency_weighted_intersection_over_union(self):
        raise NotImplementedError

    def _generate_matrix(self):
        raise NotImplementedError

    def get_table(self):
        raise NotImplementedError

    def add_batch(self, gt, pred):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

class ConfusionMatrix(MetricsBase):
    def __init__(self, num_classes, names):
        super(ConfusionMatrix, self).__init__(num_classes, names)
        assert num_classes == len(names)
        self.num_classes = num_classes
        self.names = names
        self.confusion_matrix = np.zeros((self.num_classes,) * 2)
        self.name = 'mIoU'

    def pixel_accuracy(self):
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def pixel_accuracy_class(self):
        acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        acc = np.nanmean(acc)
        return acc

    def mean_intersection_over_union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def get(self):
        return self.mean_intersection_over_union()

    def frequency_weighted_intersection_over_union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pred_image):
        mask = (gt_image >= 0) & (gt_image < self.num_classes)
        label = self.num_classes * gt_image[mask].astype('int') + pred_image[mask]
        count = np.bincount(label, minlength=self.num_classes ** 2)
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        return confusion_matrix

    def get_table(self):
        eps = 1e-4
        total_elem = np.sum(self.confusion_matrix, axis=None)
        tp = np.diag(self.confusion_matrix)
        fp_plus_tp = np.sum(self.confusion_matrix, axis=0)
        fn_plus_tp = np.sum(self.confusion_matrix, axis=1)

        A = (total_elem - (fp_plus_tp + fn_plus_tp - 2 * tp)) / total_elem
        R = tp / (eps + fn_plus_tp)
        P = tp / (eps + fp_plus_tp)
        F1 = 2 * P * R / (eps + P + R)
        IOU = tp / (eps + fp_plus_tp + fn_plus_tp - tp)

        df = pd.DataFrame(data=np.column_stack([IOU, F1, P, R, A]),
                          columns=['IoU', 'F1', 'Prec', 'recall', 'Acc'])

        df = df.round(4)
        df.index = self.names
        total = df.iloc[:, :].mean()
        total_bg = df.iloc[1:, :].mean()
        df.loc['total'] = total
        df.loc['total(-bg)'] = total_bg

        return df

    def add_batch(self, gt_image, pred_image):
        assert gt_image.shape == pred_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pred_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes,) * 2)


class RMSE(object):
    def __init__(self):
        self.sq_errors = []
        self.num_pix = 0
        self.name = 'RMSE'

    def get(self):
        return np.sqrt(
            np.sum(np.array(self.sq_errors)) / self.num_pix
        )

    def add_batch(self, pred, target):
        sqe = (pred - target) ** 2
        self.sq_errors.append(np.sum(sqe))
        self.num_pix += target.size

    def reset(self):
        self.sq_errors = []
        self.num_pix = 0


# Used to keep track of statistics
class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count