"""Minimal KAIR-compatible inference networks (MIT; see LICENSE_KAIR.txt).

Inference adapters derived from the architecture definitions in cszn/KAIR.
This module intentionally does not import KAIR/BasicSR's training environment.
"""
import torch
from torch import nn
from torch.nn import functional as F


def _convs(inputs, outputs, width, depth):
    layers = [nn.Conv2d(inputs, width, 3, padding=1), nn.ReLU(inplace=True)]
    for _ in range(depth - 2):
        layers.extend([nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True)])
    layers.append(nn.Conv2d(width, outputs, 3, padding=1))
    return nn.Sequential(*layers)


class FFDNet(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        width, depth = (64, 15) if channels == 1 else (96, 12)
        self.model = _convs(channels * 4 + 1, channels * 4, width, depth)

    def forward(self, image, sigma):
        h, w = image.shape[-2:]
        image = F.pad(image, (0, w % 2, 0, h % 2), mode="replicate")
        subimages = F.pixel_unshuffle(image, 2)
        noise_map = sigma.expand(image.shape[0], 1, *subimages.shape[-2:])
        restored = self.model(torch.cat([subimages, noise_map], dim=1))
        return F.pixel_shuffle(restored, 2)[..., :h, :w]


class DnCNN(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        # KAIR's downloadable blind checkpoints have batch normalization merged.
        self.model = _convs(channels, channels, 64, 20)

    def forward(self, image):
        return image - self.model(image)
