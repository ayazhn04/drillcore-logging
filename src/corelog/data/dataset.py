"""PyTorch dataset over tiles + labels + splits, and the augmentation used in training."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .. import splits
from ..config import classes as cfg_classes, output_root
from ..labels import load_labels
from ..tiling import read_tiles


class TileDataset(Dataset):
    """Returns (float tensor C,H,W in [0,1], class index or -1 for unlabeled)."""

    def __init__(self, df: pd.DataFrame, root: Path, input_size: tuple[int, int] | None = None):
        self.paths = [Path(root) / p for p in df["tile_path"]]
        self.y = df["y"].to_numpy() if "y" in df else np.full(len(df), -1)
        self.input_size = tuple(input_size) if input_size else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB")
            if self.input_size and (im.height, im.width) != self.input_size:
                im = im.resize((self.input_size[1], self.input_size[0]), Image.Resampling.BILINEAR)
            x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return x, int(self.y[i])


def prepare(cfg: dict) -> dict[str, pd.DataFrame]:
    """Join tiles, labels and split on slice_id. Keys: train, val, test (labeled only),
    unlabeled_train (unlabeled tiles from TRAIN holes only - never val/test holes), all."""
    tiles = read_tiles(cfg)
    labels = load_labels(cfg)
    df = tiles.merge(labels, on="slice_id", how="left")
    unl = cfg["labels"].get("unlabeled", "unlabeled")
    df["label"] = df["label"].fillna(unl)
    cls = cfg_classes(cfg)
    unknown = set(df["label"]) - set(cls) - {unl}
    if unknown:
        raise ValueError(f"labels not in labels.classes: {sorted(unknown)}")
    df["y"] = df["label"].map({c: i for i, c in enumerate(cls)}).fillna(-1).astype(int)
    df["split"] = splits.assign(df, cfg["split"], cfg["project"]["seed"]).to_numpy()
    out = {p: df[(df["split"] == p) & (df["y"] >= 0)].reset_index(drop=True) for p in ("train", "val", "test")}
    out["unlabeled_train"] = df[(df["split"] == "train") & (df["y"] < 0)].reset_index(drop=True)
    out["all"] = df
    return out


def summary(parts: dict[str, pd.DataFrame], classes: list[str]) -> pd.DataFrame:
    rows = []
    for p in ("train", "val", "test", "unlabeled_train"):
        d = parts[p]
        r = {"split": p, "n": len(d), "holes": ",".join(sorted(d["hole_id"].unique()))}
        if p != "unlabeled_train":
            r.update({c: int((d["y"] == i).sum()) for i, c in enumerate(classes)})
        rows.append(r)
    return pd.DataFrame(rows)


def augment(x: torch.Tensor, a: dict, gen: torch.Generator | None = None) -> torch.Tensor:
    """Batch-wise random perturbations on [0,1] images (paper §3.3.2). Independent per call,
    so calling it twice gives the two independent noises of Mean Teacher (eta, eta')."""
    B = x.shape[0]
    dev = x.device

    def u(lo, hi, shape):
        return torch.rand(shape, generator=gen, device="cpu").to(dev) * (hi - lo) + lo

    if a.get("hflip"):
        m = u(0, 1, (B, 1, 1, 1)) < 0.5
        x = torch.where(m, x.flip(-1), x)
    if a.get("vflip"):
        m = u(0, 1, (B, 1, 1, 1)) < 0.5
        x = torch.where(m, x.flip(-2), x)
    if a.get("brightness"):
        b = a["brightness"]
        x = x * u(1 - b, 1 + b, (B, 1, 1, 1))
    if a.get("contrast"):
        c = a["contrast"]
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) * u(1 - c, 1 + c, (B, 1, 1, 1)) + mean
    if a.get("gaussian_noise_std"):
        x = x + torch.randn(x.shape, generator=gen, device="cpu").to(dev) * a["gaussian_noise_std"]
    return x.clamp(0, 1)
