import torch
import torch.fft as fft
import torch.nn as nn
from torch.nn import functional as F
import math


class FourierLossETH(nn.Module):  # nn.Module
    def __init__(self, reduction='mean', **kwargs):
        super(FourierLossETH, self).__init__()
        self.reduction = reduction

    def forward(self, sr, hr):
        sr_w, hr_w = self.addWindows(sr, hr)
        sr_amplitudes, sr_phases = self.comFourier(sr_w)
        hr_amplitudes, hr_phases = self.comFourier(hr_w)
        amp_loss = self.get_l1loss(sr_amplitudes, hr_amplitudes)
        phase_loss = self.get_angleloss(sr_phases, hr_phases)
        
        return phase_loss, amp_loss

    def comFourier(self, image):
        Fs = fft.fftshift(fft.fftn(image, dim=(2, 3)), dim=(2, 3))
        amplitudes = Fs.abs()
        phases = torch.angle(Fs)
        return amplitudes, phases

    def addWindows(self, sr, hr):
        b, c, h, w = sr.size()
        win1 = torch.hann_window(h).reshape(h, 1)
        win2 = torch.hann_window(w).reshape(1, w)
        win = torch.mm(win1, win2)
        sr, hr = sr * win, hr * win
        return sr, hr

    def get_angleloss(self, pred, target):
        """
        pred_phase, true_phase: complex phases in (-π, π] with shape [B, C, H, W]

        Computes     2/N * Σ_{u=0}^{H/2-1} Σ_v  | d(Δφ) |
        where        d(Δφ) = min(|Δφ|, 2π-|Δφ|)  ∈ [0, π].
        """
        # 1. signed difference wrapped to (-π, π]
        diff = (pred - target + math.pi) % (2*math.pi) - math.pi   # [-π, π)

        # 2. shortest distance on the circle, 0 … π
        d = torch.abs(diff)
        d = torch.minimum(d, 2*math.pi - d)

        # 3. keep only the positive-frequency half along the first spatial axis
        d = d[..., : d.shape[-2] // 2, :]

        # 4. L1 reduction and factor 2
        loss = d.mean()

        return loss

    def get_l1loss(self, pred, target):
        # absolute difference
        diff = torch.abs(pred - target)                # [B,C,H,W]

        # positive-frequency half-plane (rows 0 … H/2-1)
        diff = diff[..., : diff.shape[-2] // 2, :]           # [B,C,H/2,W]

        # L1 over that half and factor 2
        loss = diff.mean()

        return loss
    

class FourierLossDelft(nn.Module):
    def __init__(self, reduction='mean', **kwargs):
        super(FourierLossDelft, self).__init__()
        self.reduction = reduction

    def forward(self, sr, hr):
        sr_w, hr_w = self.addWindows(sr, hr)
        sr_F = self.comFourier(sr_w)
        hr_F = self.comFourier(hr_w)
        amp_loss = self.get_log_amplitude_loss(sr_F, hr_F)
        phase_loss = self.get_phase_cosine_loss(sr_F, hr_F)
        return phase_loss, amp_loss

    def comFourier(self, image):
        # returns complex-valued FFT output
        return fft.fftshift(fft.fftn(image, dim=(2, 3)), dim=(2, 3))

    def addWindows(self, sr, hr):
        b, c, h, w = sr.size()
        win1 = torch.hann_window(h, device=sr.device).reshape(h, 1)
        win2 = torch.hann_window(w, device=sr.device).reshape(1, w)
        win = torch.mm(win1, win2)  # [H, W]
        sr, hr = sr * win, hr * win
        return sr, hr

    def get_phase_cosine_loss(self, predF, targetF):
        """
        Computes cosine similarity between unit phasors:
        L = 1 - Re( F / |F| ⋅ conj(F̂ / |F̂|) )
        """

        # normalize to unit phasors
        pred_norm = predF / (predF.abs())
        target_norm = targetF / (targetF.abs())

        # cosine similarity: Re[ U ⋅ conj(Û) ]
        dot = (pred_norm * target_norm.conj()).real

        # restrict to positive frequencies
        dot = dot[..., : dot.shape[-2] // 2, :]

        # cosine distance
        loss = (1.0 - dot).mean()

        return loss

    def get_log_amplitude_loss(self, predF, targetF):
        """
        L2 loss between log-amplitudes
        """
        pred_log = torch.log(predF.abs())
        target_log = torch.log(targetF.abs())
        diff = (pred_log - target_log) ** 2
        diff = diff[..., : diff.shape[-2] // 2, :]
        loss = diff.mean()
        return loss
    
    
class FourierLossHK(nn.Module):
    def __init__(self, reduction='mean', **kwargs):
        super(FourierLossHK, self).__init__()
        self.reduction = reduction

    def forward(self, sr, hr):
        sr_w, hr_w = self.addWindows(sr, hr)
        sr_F = self.comFourier(sr_w)
        hr_F = self.comFourier(hr_w)
        amp_loss = self.get_amplitude_loss(sr_F, hr_F)
        corr_loss = self.get_fourier_correlation_loss(sr_F, hr_F)
        return corr_loss, amp_loss

    def comFourier(self, image):
        return fft.fftshift(fft.fftn(image, dim=(2, 3)), dim=(2, 3))

    def addWindows(self, sr, hr):
        b, c, h, w = sr.size()
        win1 = torch.hann_window(h, device=sr.device).reshape(h, 1)
        win2 = torch.hann_window(w, device=sr.device).reshape(1, w)
        win = torch.mm(win1, win2)
        sr, hr = sr * win, hr * win
        return sr, hr

    def get_fourier_correlation_loss(self, predF, targetF):
        """
        Computes 1 - cosine similarity between complex Fourier coefficients:
        L = 1 - Re( <F, F̂> ) / ( ||F|| * ||F̂|| )
        """
        # Flatten the spatial dimensions
        predF = predF[..., : predF.shape[-2] // 2, :]  # keep positive frequencies
        targetF = targetF[..., : targetF.shape[-2] // 2, :]

        dot = (predF * targetF.conj()).real.sum(dim=(-2, -1))  # [B, C]
        norm_pred = predF.abs().pow(2).sum(dim=(-2, -1)).sqrt() 
        norm_target = targetF.abs().pow(2).sum(dim=(-2, -1)).sqrt() 
        cosine = dot / (norm_pred * norm_target)

        loss = (1.0 - cosine).mean()
        return loss

    def get_amplitude_loss(self, predF, targetF):
        pred = predF.abs()
        target = targetF.abs() 
        diff = (pred - target) ** 2
        diff = diff[..., : diff.shape[-2] // 2, :]
        loss = diff.mean()
        return loss
    
    
class FourierLossMSEAmpHF(nn.Module):
    def __init__(self, reduction='mean', **kwargs):
        super(FourierLossMSEAmpHF, self).__init__()
        self.reduction = reduction

    def forward(self, sr, hr):
        sr_w, hr_w = self.addWindows(sr, hr)
        sr_F = self.comFourier(sr_w)
        hr_F = self.comFourier(hr_w)

        # Build frequency weights dynamically from shape
        freq_weights = self.make_high_freq_weights(sr_F.shape, sr.device)

        pix_loss = self.get_pixel_mse_loss(sr, hr)
        amp_loss = self.get_log_amplitude_loss(sr_F, hr_F, freq_weights)
        return pix_loss, amp_loss

    def comFourier(self, image):
        return fft.fftshift(fft.fftn(image, dim=(2, 3)), dim=(2, 3))

    def addWindows(self, sr, hr):
        b, c, h, w = sr.size()
        win1 = torch.hann_window(h, device=sr.device).reshape(h, 1)
        win2 = torch.hann_window(w, device=sr.device).reshape(1, w)
        win = torch.mm(win1, win2)
        sr, hr = sr * win, hr * win
        return sr, hr

    def get_pixel_mse_loss(self, pred, target):
        return torch.mean((pred - target) ** 2)

    def get_log_amplitude_loss(self, predF, targetF, freq_weights):
        # pred_log = torch.log(predF.abs())
        # target_log = torch.log(targetF.abs())
        pred_phase = predF.abs()
        target_phase = targetF.abs()
        diff = (pred_phase - target_phase) ** 2

        # Positive-frequency half-plane
        diff = diff[..., : diff.shape[-2] // 2, :]
        freq_weights = freq_weights[..., : diff.shape[-2], :]

        weighted = diff * freq_weights
        return torch.mean(weighted)

    def make_high_freq_weights(self, shape, device):
        # shape: [B, C, H, W] of FFT image
        _, _, H, W = shape
        u = torch.arange(H, device=device).reshape(H, 1).float()
        v = torch.arange(W, device=device).reshape(1, W).float()
        center_u = (H - 1) / 2
        center_v = (W - 1) / 2

        r = torch.sqrt((u - center_u)**2 + (v - center_v)**2)
        r = r / r.max()  # normalize to [0, 1]

        # Emphasize high frequencies (square to exaggerate)
        w = r**2
        return w.unsqueeze(0).unsqueeze(0)  # shape [1,1,H,W]


if __name__ == '__main__':
    sr_tensor = torch.rand([4, 5, 368, 368])
    hr_tensor = torch.rand([4, 5, 368, 368])
    F_ETH = FourierLossETH()
    F_Delft = FourierLossDelft()
    F_HK = FourierLossHK()
    F_MSEAmpHF = FourierLossMSEAmpHF()
    phase_loss_ETH, amp_loss_ETH = F_ETH(sr_tensor, hr_tensor)
    phase_loss_Delft, amp_loss_Delft = F_Delft(sr_tensor, hr_tensor)
    corr_loss_HK, amp_loss_HK = F_HK(sr_tensor, hr_tensor)
    pix_loss_, amp_loss_ = F_MSEAmpHF(sr_tensor, hr_tensor)
    
    
    print(f'ETH Phase Loss: {phase_loss_ETH.item()}, Amplitude Loss: {amp_loss_ETH.item()}')
    print(f'Delft Phase Loss: {phase_loss_Delft.item()}, Amplitude Loss: {amp_loss_Delft.item()}')
    print(f'HK Correlation Loss: {corr_loss_HK.item()}, Amplitude Loss: {amp_loss_HK.item()}')
    print(f'MSE Pixel Loss: {pix_loss_.item()}, Amplitude Loss: {amp_loss_.item()}')
    a=1
    
    
