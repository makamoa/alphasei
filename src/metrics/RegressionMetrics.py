import numpy as np
import torch 
from pytorch_msssim import ms_ssim, ssim
from math import exp

class RegressionMetricsFull:
    """
    Accumulate predictions and true values to compute various regression metrics.
    Metrics: MSE, MS-SSIM, PSNR, RMSE, and MAE.
    All predictions and true values are accumulated in memory.
    """
    
    def __init__(self, data_range: float, window_size: int = 11, normalize: bool = True):
        self.data_range = data_range # for PSNR and MS-SSIM
        self.window_size = window_size
        self.normalize = normalize
        self.pred = []
        self.true = []
        
        
    @torch.no_grad()
    def update(self, pred: torch.Tensor, true: torch.Tensor):
        """Update the predictions and true values."""
        self.pred.append(pred)
        self.true.append(true)
        
    def reset(self): 
        """Reset all accumulated values."""
        self.pred = []
        self.true = []
        
    def _normalize_ssim(self, value):
        return (value + 1) / 2 if self.normalize else value
    
    @property
    def _preds_and_trues(self):
        """Combine all predictions and true values into single tensors."""
        if not self.predictions:
            raise ValueError("No data has been accumulated yet.")
        preds = torch.cat(self.predictions)
        trues = torch.cat(self.true_values)
        return preds, trues
    
    @property
    def mse(self):
        """Get the mean squared error."""
        pred, true = self._preds_and_trues
        return torch.mean((pred - true) ** 2)
    
    @property
    def rmse(self):
        """Get the root mean squared error."""
        return torch.sqrt(self.mse)
    
    @property
    def mae(self):
        """Get the mean absolute error."""
        pred, true = self._preds_and_trues
        return torch.mean(torch.abs(pred - true))
    
    @property
    def ms_ssim(self):
        """Get the multi-scale structural similarity index."""
        pred, true = self._preds_and_trues
        
        pred = pred.view(-1, 1, self.last_pred.shape[-2], self.last_pred.shape[-1]).clamp(0, self.data_range)
        true = true.view(-1, 1, self.last_true.shape[-2], self.last_true.shape[-1]).clamp(0, self.data_range)
        """
        (b, h, w) -> (b, 1, h, w)
        (h, w) -> (1, 1, h, w)
        (b, d, h, w) -> (b*d, 1, h, w)
        """
        msssim = ms_ssim(pred, true, data_range=self.data_range, size_average=True, win_size=self.window_size) 
        return self._normalize_ssim(msssim)
    
    @property
    def ssim(self):
        """Get the structural similarity index."""
        pred, true = self._preds_and_trues
        
        pred = pred.view(-1, 1, self.last_pred.shape[-2], self.last_pred.shape[-1]).clamp(0, self.data_range)
        true = true.view(-1, 1, self.last_true.shape[-2], self.last_true.shape[-1]).clamp(0, self.data_range)
        """
        (b, h, w) -> (b, 1, h, w)
        (h, w) -> (1, 1, h, w)
        (b, d, h, w) -> (b*d, 1, h, w)
        """
        sim = ssim(pred, true, data_range=self.data_range, size_average=True, win_size=self.window_size)
        return self._normalize_ssim(sim)
    
    @property
    def psnr(self):
        """Get the peak signal-to-noise ratio."""
        if self.mse == 0:
            Warning.warn("MSE is zero, PSNR is infinite.")
            return float('inf')
        return 20 * torch.log10(self.data_range / torch.sqrt(self.mse))
    
    def __repr__(self):
        """A tabular representation of the metrics."""
        metrics = ['MSE', 'PSNR', 'RMSE', 'MAE', 'MS-SSIM', 'SSIM']
        values = [self.mse, self.psnr, self.rmse, self.mae, self.ms_ssim, self.ssim]

        out = 'Regression Metrics:\n'
        out += '-' * 60 + '\n'
        for metric, value in zip(metrics, values):
            out += f'{metric:<10} {value:.4f}\n'
        return out
        
class RegressionMetricsLight:
    """
    Metrics: MSE, RMSE, MAE, PSNR, and MS-SSIM (last batch).
    
    Data (preds, and true) is not accumulated in memory, only last batch.
    """
    
    def __init__ (self, data_range: float, window_size: int = 11, normalize: bool = True):
        self.data_range = data_range
        self.window_size = window_size
        self.normalize = normalize
        self.reset()
        
    def _normalize_ssim(self, value):
        return (value + 1) / 2 if self.normalize else value
        
    def reset(self):
        """Reset all accumulated statistics."""
        self.sum_squared_error = 0.0
        self.sum_absolute_error = 0.0
        
        self.sum_true = 0.0
        self.sum_pred = 0.0
        
        self.sum_true_squared = 0.0
        self.sum_pred_squared = 0.0
        
        self.sum_product = 0.0
        
        self.n_samples = 0
        self.last_batch_size = 0
        self.last_pred = None
        self.last_true = None
        
    @torch.no_grad()
    def update(self, pred: torch.Tensor , true: torch.Tensor):
        """Update the predictions and true values."""
        self.last_batch_size = true.numel()
        self.n_samples += self.last_batch_size
        
        self.last_pred = pred
        self.last_true = true
        
        self.sum_squared_error += torch.sum((pred - true) ** 2).item()
        self.sum_absolute_error += torch.sum(torch.abs(pred - true)).item()
        
        self.sum_true += torch.sum(true).item()
        self.sum_pred += torch.sum(pred).item()
        
        self.sum_true_squared += torch.sum(true ** 2).item()
        self.sum_pred_squared += torch.sum(pred ** 2).item()
        
        self.sum_product += torch.sum(pred * true).item()
        
    @property
    def mse(self):
        """Get the mean squared error."""
        return self.sum_squared_error / self.n_samples
    
    @property
    def rmse(self):
        """Get the root mean squared error."""
        return torch.sqrt(torch.tensor(self.mse))
    
    @property
    def mae(self):
        """Get the mean absolute error."""
        return self.sum_absolute_error / self.n_samples
    
    @property
    def psnr(self):
        """Get the peak signal-to-noise ratio."""
        if self.mse == 0:
            return float('inf')
        return 20 * torch.log10(torch.tensor(self.data_range / torch.sqrt(torch.tensor(self.mse))))        
    
    @property
    def ms_ssim(self):
        """
        Get the multi-scale structural similarity index.
        Shape must be (batch, channel, height, width):

        """
        if self.last_pred is None or self.last_true is None:
            raise ValueError("No data available for MS-SSIM calculation")
        
        pred = self.last_pred.view(-1, 1, self.last_pred.shape[-2], self.last_pred.shape[-1]).clamp(0, self.data_range)
        true = self.last_true.view(-1, 1, self.last_true.shape[-2], self.last_true.shape[-1]).clamp(0, self.data_range)
        """
        (b, h, w) -> (b, 1, h, w)
        (h, w) -> (1, 1, h, w)
        (b, d, h, w) -> (b*d, 1, h, w)
        """
        sim =  ms_ssim(pred, true, data_range=self.data_range, size_average=True, win_size=self.window_size)
        return self._normalize_ssim(sim)
    
    @property
    def ssim(self):
        """
        Get the structural similarity index.
        Shape must be (batch, channel, height, width):
        """
        
        if self.last_pred is None or self.last_true is None:
            raise ValueError("No data available for SSIM calculation")
        
        pred = self.last_pred.view(-1, 1, self.last_pred.shape[-2], self.last_pred.shape[-1]).clamp(0, self.data_range)
        true = self.last_true.view(-1, 1, self.last_true.shape[-2], self.last_true.shape[-1]).clamp(0, self.data_range)
        
        sim = ssim(pred, true, data_range=self.data_range, size_average=True, win_size=self.window_size)
        return self._normalize_ssim(sim)
    
    
    def __repr__(self):
        """A tabular representation of the metrics."""
        metrics = ['MSE', 'RMSE', 'MAE', 'PSNR']
        values = [self.mse, self.rmse, self.mae, self.psnr]
        
        out = 'Regression Metrics:\n'
        out += '-' * 60 + '\n'
        for metric, value in zip(metrics, values):
            out += f'{metric:<10} {value:.4f}\n'
        
        if self.ms_ssim is not None:
            out += f'{'MS-SSIM':<10} {self.ms_ssim.item():.4f} (last batch)\n'
        else:
            out += f'{'MS-SSIM':<10} Not available\n'
        
        if self.ssim is not None:
            out += f'{'SSIM':<10} {self.ssim.item():.4f} (last batch)\n'
        else:
            out += f'{'SSIM':<10} Not available\n'
        
        return out
        