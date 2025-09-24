import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

class MSEWithSpectralLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(MSEWithSpectralLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        # Time domain loss (MSE)
        mse = self.mse_loss(y_pred, y_true)
        
        # Frequency domain loss (Spectral Loss)
        y_true_fft = torch.fft.rfft(y_true, dim=-1)
        y_pred_fft = torch.fft.rfft(y_pred, dim=-1)
        
        spectral_loss = torch.mean(torch.abs(y_true_fft - y_pred_fft)**2)
        
        # Combine the losses
        total_loss = self.alpha * mse + self.beta * spectral_loss
        return total_loss


class L1WithSpectralLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(L1WithSpectralLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1_loss = nn.L1Loss()

    def forward(self, y_pred, y_true):
        l1_loss = self.l1_loss(y_pred, y_true)
        y_true_fft = torch.fft.rfft(y_true, dim=-1)
        y_pred_fft = torch.fft.rfft(y_pred, dim=-1)
        spectral_loss = torch.mean(torch.abs(y_true_fft - y_pred_fft) ** 2)
        total_loss = self.alpha * l1_loss + self.beta * spectral_loss
        return total_loss
    

def random_time_mask(x: torch.Tensor, mask_ratio: float = 0.5,
                     mode: str = "zero") -> torch.Tensor:
    assert x.dim() in (2, 3)
    if x.dim() == 2:
        x = x.unsqueeze(0)
    B, C, T = x.shape
    k = int(T * mask_ratio)
    idx = torch.randperm(T)[:k]
    mask = torch.zeros(B, 1, T, dtype=torch.bool, device=x.device)
    mask[:, :, idx] = True

    x_masked = x.clone()
    if mode == "zero":
        x_masked = x_masked.masked_fill(mask, 0.0)
    elif mode == "noise":
        noise = torch.randn_like(x_masked) * x_masked.std(dim=-1, keepdim=True).clamp_min(1e-6)
        x_masked = torch.where(mask, noise, x_masked)
    elif mode == "mean":
        mean = x_masked.mean(dim=-1, keepdim=True)
        x_masked = torch.where(mask, mean, x_masked)
    else:
        raise ValueError("mode must be zero|noise|mean")

    return x_masked if x.dim() == 3 else x_masked.squeeze(0)