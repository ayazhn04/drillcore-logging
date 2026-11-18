"""Stage 1: baseline CNN / VGG16 architectures and supervised training on synthetic tiles."""
import pandas as pd
import pytest
import torch

from corelog.data.dataset import augment, prepare
from corelog.models import BaselineCNN, VGG16Transfer
from corelog.train.common import batch_size_at, load_checkpoint
from corelog.train.supervised import train_supervised


def test_baseline_shapes_and_paper_head():
    x = torch.rand(2, 3, 60, 300)
    m = BaselineCNN(4, (60, 300), head="paper")
    assert m(x).shape == (2, 4)
    convs = [l for l in m.features if isinstance(l, torch.nn.Conv2d)]
    assert [c.out_channels for c in convs] == [32, 32, 64] and all(c.kernel_size == (3, 3) for c in convs)
    linears = [l for l in m.head if isinstance(l, torch.nn.Linear)]
    assert len(linears) == 1                         # paper Fig. 8: single dense softmax layer
    assert linears[0].in_features == 64 * 8 * 38     # 60x300 -> 8x38 after three 'same' pools
    nb = BaselineCNN(4, (60, 300), head="notebook").eval()
    assert nb(x).shape == (2, 4)


def test_vgg16_frozen_base_without_download():
    m = VGG16Transfer(5, weights="none", freeze_features=True)
    assert m(torch.rand(1, 3, 60, 300)).shape == (1, 5)
    assert not any(p.requires_grad for p in m.net.features.parameters())
    assert all(p.requires_grad for p in m.net.classifier.parameters())


def test_batch_size_schedule():
    t = {"batch_size": 16, "batch_growth_factor": 2, "batch_growth_every": 20, "max_batch_size": 64}
    assert [batch_size_at(e, t) for e in (0, 19, 20, 40, 60, 100)] == [16, 16, 32, 64, 64, 64]


def test_augment_keeps_range_and_is_random():
    x = torch.rand(4, 3, 60, 300)
    a = {"gaussian_noise_std": 0.05, "hflip": True, "vflip": True, "brightness": 0.2, "contrast": 0.2}
    y1, y2 = augment(x, a), augment(x, a)
    assert y1.min() >= 0 and y1.max() <= 1 and not torch.allclose(y1, y2)


def test_prepare_uses_hole_splits_and_keeps_unlabeled_separate(tiled):
    cfg, _, _ = tiled
    p = prepare(cfg)
    assert set(p["train"]["hole_id"]) == {"SYN_A", "SYN_B"} and set(p["test"]["hole_id"]) == {"SYN_D"}
    assert set(p["unlabeled_train"]["hole_id"]) <= {"SYN_A", "SYN_B"}  # never from val/test holes
    assert (p["train"]["y"] >= 0).all()


def test_supervised_training_runs_and_learns(tiled):
    cfg, _, _ = tiled
    import copy
    cfg = copy.deepcopy(cfg)
    cfg["train"].update({"epochs": 8, "max_steps_per_epoch": None, "patience": 8, "lr": 1e-3})
    run = train_supervised(cfg, run_name="pytest_supervised")
    hist = pd.read_csv(run / "history.csv")
    assert {"model.pt", "config.yaml", "environment.json", "history.csv", "data_summary.csv"} <= {p.name for p in run.iterdir()}
    assert hist["train_loss"].min() < 0.5 * hist["train_loss"].iloc[0]
    assert hist["val_acc"].max() > 0.5                # 4 classes, chance = 0.25
    best = hist.loc[hist["val_loss"].idxmin()]
    import json
    assert json.loads((run / "train_summary.json").read_text())["best_epoch"] == int(best["epoch"])
    model, classes, ck = load_checkpoint(run / "model.pt")
    assert classes == cfg["labels"]["classes"] and model(torch.rand(1, 3, 60, 300)).shape == (1, 4)
