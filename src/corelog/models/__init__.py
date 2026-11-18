"""CNN architectures: the paper's baseline (Fig. 8) and transfer-learning VGG16."""
from __future__ import annotations

import math

import torch
from torch import nn


class BaselineCNN(nn.Module):
    """Paper Fig. 8: three blocks of 3x3 conv (32, 32, 64) + ReLU -> BatchNorm -> MaxPool,
    then dropout and a single dense softmax layer (head='paper').
    head='notebook' adds the authors' code's two Dense(128) blocks (for ablation)."""

    def __init__(self, n_classes: int, input_size=(60, 300), head: str = "paper", dropout: float = 0.5):
        super().__init__()
        layers, c_in = [], 3
        for c_out in (32, 32, 64):
            layers += [nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(inplace=True),
                       nn.BatchNorm2d(c_out), nn.MaxPool2d(2, ceil_mode=True)]  # Keras padding='same'
            c_in = c_out
        self.features = nn.Sequential(*layers)
        h, w = input_size
        for _ in range(3):
            h, w = math.ceil(h / 2), math.ceil(w / 2)
        flat = 64 * h * w
        if head == "paper":
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(flat, n_classes))
        elif head == "notebook":
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat, 128), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.BatchNorm1d(128),
                nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.BatchNorm1d(128),
                nn.Linear(128, n_classes))
        else:
            raise ValueError(f"unknown head {head!r}")

    def forward(self, x):  # x: B,3,H,W in [0,1]
        return self.head(self.features(x))  # logits


class VGG16Transfer(nn.Module):
    """ImageNet VGG16; convolutional base frozen (paper: only FC layers trained).
    ImageNet normalisation is applied inside the model, so inputs stay in [0,1]."""

    def __init__(self, n_classes: int, weights: str = "imagenet", freeze_features: bool = True):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16
        w = VGG16_Weights.IMAGENET1K_V1 if weights == "imagenet" else None
        self.net = vgg16(weights=w)
        self.net.classifier[6] = nn.Linear(4096, n_classes)
        if freeze_features:
            for p in self.net.features.parameters():
                p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def build_model(cfg: dict, n_classes: int) -> nn.Module:
    m = cfg["model"]
    if m["name"] == "baseline":
        return BaselineCNN(n_classes, tuple(m["input_size"]), m.get("head", "paper"), m.get("dropout", 0.5))
    if m["name"] == "vgg16":
        return VGG16Transfer(n_classes, m.get("vgg_weights", "imagenet"), m.get("vgg_freeze_features", True))
    raise ValueError(f"unknown model.name {m['name']!r}")
