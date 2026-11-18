"""Stage 3: Mean Teacher. Central regression test: the consistency term must produce a
non-zero gradient in the student (it was zero in the code behind the paper's Table 3)."""
import copy
import json

import pandas as pd
import pytest
import torch

from corelog.models import BaselineCNN
from corelog.train.mean_teacher import ema_update, mt_loss, sigmoid_rampup, train_mean_teacher

AUG = {"gaussian_noise_std": 0.05, "hflip": True, "vflip": True, "brightness": 0.2, "contrast": 0.2}


def _pair():
    torch.manual_seed(0)
    s = BaselineCNN(4, (60, 300))
    t = copy.deepcopy(s)
    for p in t.parameters():
        p.requires_grad_(False)
    return s, t


def test_consistency_term_has_gradient_in_student_only():
    s, t = _pair()
    xl, yl, xu = torch.rand(4, 3, 60, 300), torch.tensor([0, 1, 2, 3]), torch.rand(6, 3, 60, 300)
    s.train(); t.train()
    _, _, cons = mt_loss(s, t, xl, yl, xu, lam=10.0, aug=AUG)
    assert cons.item() > 0                                   # independent noise -> student != teacher outputs
    cons.backward()
    g = sum(p.grad.abs().sum().item() for p in s.parameters() if p.grad is not None)
    assert g > 0, "consistency loss must back-propagate into the student"
    assert all(p.grad is None for p in t.parameters())       # teacher is never trained by gradients


def test_lambda_changes_the_update():
    s, t = _pair()
    xl, yl, xu = torch.rand(4, 3, 60, 300), torch.tensor([0, 1, 2, 3]), torch.rand(6, 3, 60, 300)
    grads = []
    for lam in (0.0, 10.0):
        s.zero_grad()
        gen = torch.Generator().manual_seed(1)               # identical noise in both runs
        s.eval(); t.eval()                                   # no dropout -> deterministic
        loss, _, _ = mt_loss(s, t, xl, yl, xu, lam, AUG, gen)
        loss.backward()
        grads.append(torch.cat([p.grad.flatten() for p in s.parameters()]).clone())
    assert not torch.allclose(grads[0], grads[1])


def test_ema_update_formula_and_buffers():
    s, t = _pair()
    with torch.no_grad():
        for p in s.parameters():
            p.add_(1.0)
    before = [p.clone() for p in t.parameters()]
    ema_update(t, s, 0.99)
    for b, tp, sp in zip(before, t.parameters(), s.parameters()):
        assert torch.allclose(tp, 0.99 * b + 0.01 * sp, atol=1e-6)


def test_rampup_schedule_matches_paper():
    assert sigmoid_rampup(0, 20) == pytest.approx(0.006738, rel=1e-3)
    assert sigmoid_rampup(20, 20) == 1.0 and sigmoid_rampup(50, 20) == 1.0
    assert 0 < sigmoid_rampup(10, 20) < 1


def test_mean_teacher_training_run(tiled):
    cfg, _, _ = tiled
    cfg = copy.deepcopy(cfg)
    cfg["mean_teacher"]["enabled"] = True
    cfg["train"].update({"epochs": 3, "lr": 1e-3})
    run = train_mean_teacher(cfg, run_name="pytest_meanteacher")
    h = pd.read_csv(run / "history.csv")
    assert (h["consistency"] > 0).all() and h["lambda"].iloc[-1] > h["lambda"].iloc[0]
    assert {"val_acc_teacher", "val_acc_student"} <= set(h.columns)
    s = json.loads((run / "train_summary.json").read_text())
    assert s["evaluated_model"] == "teacher" and s["n_unlabeled_train"] > 0
