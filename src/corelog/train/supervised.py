"""Fully supervised training (baseline CNN or VGG16) with hole-level validation."""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from ..config import classes as cfg_classes
from ..data.dataset import augment, prepare, summary
from ..models import build_model
from ..runinfo import new_run_dir, pick_device, set_seed
from .common import (EarlyStopping, batch_size_at, class_weights, evaluate_loss, loader,
                     save_checkpoint)


def train_supervised(cfg: dict, run_name: str | None = None) -> Path:
    set_seed(cfg["project"]["seed"])
    t = cfg["train"]
    cls = cfg_classes(cfg)
    parts = prepare(cfg)
    if parts["train"].empty:
        raise ValueError("no labeled training tiles: check labels and split.train_holes")
    if parts["val"].empty:
        warnings.warn("no labeled validation tiles: early stopping disabled, last epoch is saved")
    run = new_run_dir(cfg, "supervised", run_name)
    summary(parts, cls).to_csv(run / "data_summary.csv", index=False)
    dev = pick_device(t.get("device", "auto"))
    model = build_model(cfg, len(cls)).to(dev)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=t["lr"],
                           weight_decay=t.get("weight_decay", 0.0))
    w = class_weights(parts["train"]["y"].to_numpy(), len(cls), t.get("class_weighting", "none"))
    loss_fn = nn.CrossEntropyLoss(weight=w.to(dev) if w is not None else None)
    eval_loss = nn.CrossEntropyLoss()
    val_dl = loader(parts["val"], cfg, 256, shuffle=False)
    stopper = EarlyStopping(t["patience"])
    gen = torch.Generator().manual_seed(cfg["project"]["seed"])
    hist = []
    for epoch in range(t["epochs"]):
        t0, bs = time.time(), batch_size_at(epoch, t)
        model.train()
        dl = loader(parts["train"], cfg, bs, shuffle=True, seed=cfg["project"]["seed"] + epoch)
        tot, n = 0.0, 0
        for step, (x, y) in enumerate(dl):
            if t.get("max_steps_per_epoch") and step >= t["max_steps_per_epoch"]:
                break
            x, y = augment(x.to(dev), t.get("augment", {}), gen), y.to(dev)
            loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot, n = tot + loss.item() * len(y), n + len(y)
        vl, va = evaluate_loss(model, val_dl, dev, eval_loss)
        hist.append({"epoch": epoch + 1, "batch_size": bs, "train_loss": tot / max(n, 1),
                     "val_loss": vl, "val_acc": va, "seconds": round(time.time() - t0, 2)})
        print(f"epoch {epoch + 1:4d} bs={bs:4d} train_loss={tot / max(n, 1):.4f} val_loss={vl:.4f} val_acc={va:.4f}")
        if not parts["val"].empty and stopper.step(vl, model, epoch + 1):
            print(f"early stopping at epoch {epoch + 1}; best epoch {stopper.best_epoch}")
            break
    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)
    pd.DataFrame(hist).to_csv(run / "history.csv", index=False)
    save_checkpoint(run / "model.pt", model, cfg, cls, {"best_epoch": stopper.best_epoch, "kind": "supervised"})
    (run / "train_summary.json").write_text(json.dumps(
        {"best_epoch": stopper.best_epoch, "best_val_loss": stopper.best, "epochs_run": len(hist),
         "device": str(dev)}, indent=2, default=str))
    return run
