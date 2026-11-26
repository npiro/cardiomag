"""Built-in models: sensors are input channels, time is the convolution axis."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model


def _block(cin, cout, k=7, pool=2, drop=0.1):
    return nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout), nn.GELU(),
                         nn.Dropout(drop), nn.MaxPool1d(pool) if pool > 1 else nn.Identity())


@register_model("cnn1d")
class CNN1D(nn.Module):
    """Small temporal CNN classifier."""
    task = "classify"

    def __init__(self, n_sensors, n_times, n_outputs, width=64, depth=3, dropout=0.1):
        super().__init__()
        layers, c = [], n_sensors
        for i in range(depth):
            layers.append(_block(c, width * (2 ** min(i, 2)), drop=dropout))
            c = width * (2 ** min(i, 2))
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * c, n_outputs))

    def forward(self, x):
        h = self.features(x)
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=1)
        return self.head(h)


class _Res(nn.Module):
    def __init__(self, c, k=7, dilation=1, drop=0.1):
        super().__init__()
        p = dilation * (k // 2)
        self.c1 = nn.Conv1d(c, c, k, padding=p, dilation=dilation)
        self.b1 = nn.BatchNorm1d(c)
        self.c2 = nn.Conv1d(c, c, k, padding=p, dilation=dilation)
        self.b2 = nn.BatchNorm1d(c)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        h = F.gelu(self.b1(self.c1(x)))
        h = self.b2(self.c2(self.drop(h)))
        return F.gelu(x + h)


@register_model("resnet1d")
class ResNet1D(nn.Module):
    """Dilated residual temporal CNN with a learned sensor-mixing stem."""
    task = "classify"

    def __init__(self, n_sensors, n_times, n_outputs, width=96, blocks=4, dropout=0.15):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(n_sensors, width, 1), nn.BatchNorm1d(width), nn.GELU())
        self.blocks = nn.Sequential(*[_Res(width, dilation=2 ** (i % 3), drop=dropout) for i in range(blocks)])
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * width, n_outputs))

    def forward(self, x):
        h = self.blocks(self.stem(x))
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], dim=1))


@register_model("conv_ae")
class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder with a dense bottleneck; trained on healthy data only, anomaly score =
    reconstruction error. The dense bottleneck (not translation-equivariant) forces the model to learn
    the healthy spatio-temporal pattern instead of reconstructing any waveform."""
    task = "reconstruct"

    def __init__(self, n_sensors, n_times, n_outputs=0, width=64, latent=16, code=32):
        super().__init__()
        self.T4 = (n_times + (-n_times) % 4) // 4
        self.latent = latent
        self.enc = nn.Sequential(
            nn.Conv1d(n_sensors, width, 7, padding=3), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(width, width, 5, padding=2), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(width, latent, 3, padding=1))
        self.bottleneck = nn.Sequential(nn.Flatten(), nn.Linear(latent * self.T4, code), nn.GELU(),
                                        nn.Linear(code, latent * self.T4))
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.Conv1d(latent, width, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2), nn.Conv1d(width, width, 5, padding=2), nn.GELU(),
            nn.Conv1d(width, n_sensors, 7, padding=3))

    def forward(self, x):
        T = x.shape[-1]
        pad = (-T) % 4
        if pad:
            x = F.pad(x, (0, pad), mode="replicate")
        z = self.bottleneck(self.enc(x)).view(x.shape[0], self.latent, self.T4)
        return self.dec(z)[..., :T]

    @staticmethod
    def score(x, recon):
        return ((x - recon) ** 2).mean(dim=(1, 2))
