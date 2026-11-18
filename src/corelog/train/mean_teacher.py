"""Mean Teacher (Tarvainen & Valpola 2017; paper §3.4), implemented so the consistency
term actually trains the student (the code behind the paper's Table 3 computed it
outside the gradient tape -> zero gradient; audit §5).

Per step:
  x = [labeled batch ; unlabeled batch]          (unlabeled ONLY from training holes)
  student sees augment(x), teacher sees an independent augment(x)   (eta, eta')
  L = CE(student(x_l), y_l) + lambda(epoch) * MSE(softmax(student(x)), softmax(teacher(x)))
  lambda = consistency_max * exp(-5 (1 - epoch/rampup)^2)   (paper: -> 10 over 20 epochs)
  teacher <- alpha * teacher + (1 - alpha) * student, alpha = 0.99 during ramp-up, then 0.999
"""
from __future__ import annotations

import copy
import itertools
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn
from torch import nn

from ..config import classes as cfg_classes
from ..data.dataset import augment, prepare, summary
from ..models import build_model
from ..runinfo import new_run_dir, pick_device, set_seed
from .common import EarlyStopping, batch_size_at, evaluate_loss, loader, save_checkpoint


def sigmoid_rampup(epoch: float, length: float) -> float:
    if length <= 0:
        return 1.0
    p = 1.0 - min(max(epoch, 0.0), length) / length
    return float(math.exp(-5.0 * p * p))


def mt_loss(student: nn.Module, teacher: nn.Module, xl, yl, xu, lam: float, aug: dict, gen=None):
    """Return (total, ce, consistency). Gradients flow into the student only."""
    x = torch.cat([xl, xu], 0) if xu is not None and len(xu) else xl
    s_logits = student(augment(x, aug, gen))
    with torch.no_grad():
        t_prob = torch.softmax(teacher(augment(x, aug, gen)), 1)
    ce = Fn.cross_entropy(s_logits[: len(xl)], yl)
    cons = Fn.mse_loss(torch.softmax(s_logits, 1), t_prob)  # mean over (N_l + N_u) samples and classes
    return ce + lam * cons, ce, cons


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, alpha: float) -> None:
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.mul_(alpha).add_(s.detach(), alpha=1 - alpha)
    for t, s in zip(teacher.buffers(), student.buffers()):
        if t.dtype.is_floating_point:
            t.mul_(alpha).add_(s, alpha=1 - alpha)
        else:
            t.copy_(s)


def _cycle(dl):
    while True:
        yield from dl


def train_mean_teacher(cfg: dict, run_name: str | None = None) -> Path:
    set_seed(cfg["project"]["seed"])
    t, mt = cfg["train"], cfg["mean_teacher"]
    cls = cfg_classes(cfg)
    parts = prepare(cfg)
    if parts["train"].empty:
        raise ValueError("no labeled training tiles")
    if parts["unlabeled_train"].empty:
        warnings.warn("no unlabeled tiles in training holes: consistency uses labeled tiles only")
    run = new_run_dir(cfg, "meanteacher", run_name)
    summary(parts, cls).to_csv(run / "data_summary.csv", index=False)
    dev = pick_device(t.get("device", "auto"))
    student = build_model(cfg, len(cls)).to(dev)
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=t["lr"],
                           weight_decay=t.get("weight_decay", 0.0))
    gen = torch.Generator().manual_seed(cfg["project"]["seed"])
    val_dl = loader(parts["val"], cfg, 256, shuffle=False)
    ce_eval = nn.CrossEntropyLoss()
    which = mt.get("evaluate", "teacher")
    stopper, hist, step_global = EarlyStopping(t["patience"]), [], 0
    for epoch in range(t["epochs"]):
        t0, bs = time.time(), batch_size_at(epoch, t)
        ubs = max(1, int(mt["unlabeled_batch_size"] * bs / t["batch_size"]))
        ldl = loader(parts["train"], cfg, bs, shuffle=True, seed=cfg["project"]["seed"] + epoch, drop_last=len(parts["train"]) > bs)
        uit = _cycle(loader(parts["unlabeled_train"], cfg, ubs, shuffle=True, seed=10_000 + epoch)) \
            if len(parts["unlabeled_train"]) else itertools.repeat((None, None))
        lam = mt["consistency_max"] * sigmoid_rampup(epoch, mt["rampup_epochs"])
        alpha = mt["ema_decay_rampup"] if epoch < mt["rampup_epochs"] else mt["ema_decay"]
        student.train(); teacher.train()
        sums, n = np.zeros(3), 0
        for step, (xl, yl) in enumerate(ldl):
            if t.get("max_steps_per_epoch") and step >= t["max_steps_per_epoch"]:
                break
            xu, _ = next(uit)
            xu = xu.to(dev) if xu is not None else None
            loss, ce, cons = mt_loss(student, teacher, xl.to(dev), yl.to(dev), xu, lam, t.get("augment", {}), gen)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ema_update(teacher, student, alpha)
            step_global += 1
            sums += [loss.item(), ce.item(), cons.item()]
            n += 1
        sl, sa = evaluate_loss(student, val_dl, dev, ce_eval)
        tl, ta = evaluate_loss(teacher, val_dl, dev, ce_eval)
        n = max(n, 1)
        hist.append({"epoch": epoch + 1, "batch_size": bs, "lambda": lam, "ema_alpha": alpha,
                     "loss": sums[0] / n, "ce": sums[1] / n, "consistency": sums[2] / n,
                     "val_loss_student": sl, "val_acc_student": sa, "val_loss_teacher": tl, "val_acc_teacher": ta,
                     "seconds": round(time.time() - t0, 2)})
        print(f"epoch {epoch + 1:4d} lam={lam:.3f} ce={sums[1] / n:.4f} cons={sums[2] / n:.5f} "
              f"val_loss S={sl:.4f} T={tl:.4f} val_acc S={sa:.4f} T={ta:.4f}")
        chosen = teacher if which == "teacher" else student
        if not parts["val"].empty and stopper.step(tl if which == "teacher" else sl, chosen, epoch + 1):
            print(f"early stopping at epoch {epoch + 1}; best epoch {stopper.best_epoch}")
            break
    final = teacher if which == "teacher" else student
    if stopper.best_state is not None:
        final.load_state_dict(stopper.best_state)
    pd.DataFrame(hist).to_csv(run / "history.csv", index=False)
    save_checkpoint(run / "model.pt", final, cfg, cls, {"best_epoch": stopper.best_epoch, "kind": "mean_teacher",
                                                        "evaluated_model": which})
    (run / "train_summary.json").write_text(json.dumps(
        {"best_epoch": stopper.best_epoch, "best_val_loss": stopper.best, "epochs_run": len(hist),
         "steps": step_global, "evaluated_model": which, "device": str(dev),
         "n_unlabeled_train": len(parts["unlabeled_train"])}, indent=2, default=str))
    return run
