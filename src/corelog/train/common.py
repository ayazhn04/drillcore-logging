"""Shared training utilities: loaders, batch-size schedule, early stopping, checkpoints."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config import output_root
from ..data.dataset import TileDataset
from ..models import build_model


def batch_size_at(epoch: int, t: dict) -> int:
    """Initial batch size, multiplied by `batch_growth_factor` every `batch_growth_every` epochs."""
    every = t.get("batch_growth_every") or 0
    bs = t["batch_size"] * (t.get("batch_growth_factor", 1) ** (epoch // every) if every else 1)
    return int(min(bs, t.get("max_batch_size", bs)))


def loader(df: pd.DataFrame, cfg: dict, batch_size: int, shuffle: bool, seed: int = 0, drop_last=False):
    ds = TileDataset(df, output_root(cfg), tuple(cfg["model"]["input_size"]))
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g, drop_last=drop_last,
                      num_workers=cfg["train"].get("num_workers", 0),
                      persistent_workers=False, pin_memory=torch.cuda.is_available())


def class_weights(y: np.ndarray, n: int, mode: str) -> torch.Tensor | None:
    if mode != "balanced":
        return None
    counts = np.bincount(y, minlength=n).astype(float)
    w = np.where(counts > 0, len(y) / (n * np.maximum(counts, 1)), 0.0)
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def evaluate_loss(model: nn.Module, dl, device, loss_fn) -> tuple[float, float]:
    model.eval()
    tot, correct, n = 0.0, 0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        out = model(x)
        tot += float(loss_fn(out, y)) * len(y)
        correct += int((out.argmax(1) == y).sum())
        n += len(y)
    return (tot / n, correct / n) if n else (math.nan, math.nan)


@torch.no_grad()
def predict_proba(model: nn.Module, dl, device) -> np.ndarray:
    model.eval()
    return np.concatenate([torch.softmax(model(x.to(device)), 1).cpu().numpy() for x, _ in dl]) \
        if len(dl.dataset) else np.zeros((0, 0))


class EarlyStopping:
    def __init__(self, patience: int):
        self.patience, self.best, self.wait, self.best_state, self.best_epoch = patience, math.inf, 0, None, -1

    def step(self, value: float, model: nn.Module, epoch: int) -> bool:
        """Returns True when training should stop."""
        if math.isnan(value):
            return False
        if value < self.best - 1e-6:
            self.best, self.wait, self.best_epoch = value, 0, epoch
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        self.wait += 1
        return self.wait >= self.patience


def save_checkpoint(path: Path, model: nn.Module, cfg: dict, classes: list[str], extra: dict | None = None):
    torch.save({"state_dict": model.state_dict(), "classes": classes, "model_cfg": cfg["model"],
                "extra": extra or {}}, path)


def load_checkpoint(path: Path, device="cpu") -> tuple[nn.Module, list[str], dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = {"model": dict(ck["model_cfg"])}
    if cfg["model"]["name"] == "vgg16":
        cfg["model"]["vgg_weights"] = "none"  # weights come from the checkpoint; no download
    model = build_model(cfg, len(ck["classes"]))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck["classes"], ck
