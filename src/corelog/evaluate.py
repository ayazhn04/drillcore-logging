"""Model evaluation on held-out HOLES (audit §7): accuracy, macro-F1, per-class metrics,
confusion matrix, per-hole results, summary without trivial classes (non_core), and
bootstrap confidence intervals.

Bootstrap note: tiles from one hole are spatially autocorrelated, so tile-level CIs are
too narrow. When >= 3 test holes exist, holes are resampled instead.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .config import deep_merge, load_config
from .data.dataset import prepare
from .runinfo import pick_device, provenance
from .train.common import load_checkpoint, loader, predict_proba


def run_config(run_dir: Path, current: dict | None = None) -> dict:
    """Config the run was trained with; only `paths` may be taken from the current config
    (e.g. the dataset moved), so classes and splits stay exactly those of training."""
    cfg = load_config([run_dir / "config.yaml"], base=None)
    if current is not None:
        cfg = deep_merge(cfg, {"paths": current.get("paths", {})})
    return cfg


def metrics(y: np.ndarray, p: np.ndarray, classes: list[str]) -> dict:
    labels = list(range(len(classes)))
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, p)) if len(y) else float("nan"),
            "macro_f1": float(f1_score(y, p, labels=labels, average="macro", zero_division=0)),
            "per_class": {c: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]),
                              "support": int(sup[i])} for i, c in enumerate(classes)}}


def bootstrap(df: pd.DataFrame, classes: list[str], n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    holes = df["hole_id"].unique()
    by_hole = len(holes) >= 3
    acc, mf1 = [], []
    for _ in range(n):
        if by_hole:
            pick = rng.choice(holes, len(holes), replace=True)
            s = pd.concat([df[df["hole_id"] == h] for h in pick])
        else:
            s = df.iloc[rng.integers(0, len(df), len(df))]
        acc.append(accuracy_score(s["y"], s["pred"]))
        mf1.append(f1_score(s["y"], s["pred"], labels=list(range(len(classes))), average="macro", zero_division=0))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"unit": "hole" if by_hole else "tile (optimistic: autocorrelated)", "n": n,
            "accuracy_95ci": q(acc), "macro_f1_95ci": q(mf1)}


def evaluate_run(run_dir: str | Path, current_cfg: dict | None = None, split: str = "test") -> dict:
    run_dir = Path(run_dir)
    cfg = run_config(run_dir, current_cfg)
    dev = pick_device(cfg["train"].get("device", "auto") if current_cfg is None else current_cfg["train"].get("device", "auto"))
    model, classes, _ = load_checkpoint(run_dir / "model.pt", dev)
    df = prepare(cfg)[split]
    if df.empty:
        raise ValueError(f"no labeled tiles in split {split!r}")
    prob = predict_proba(model, loader(df, cfg, 256, shuffle=False), dev)
    df = df.assign(pred=prob.argmax(1))
    out = run_dir / f"eval_{split}"
    out.mkdir(exist_ok=True)
    pred = df[["slice_id", "hole_id", "depth_from_m", "depth_to_m", "label"]].copy()
    pred["predicted"] = [classes[i] for i in df["pred"]]
    for i, c in enumerate(classes):
        pred[f"p_{c}"] = prob[:, i].round(5)
    pred.to_csv(out / "predictions.csv", index=False)
    cm = confusion_matrix(df["y"], df["pred"], labels=list(range(len(classes))))
    pd.DataFrame(cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes]).to_csv(out / "confusion.csv")
    res = {**provenance(cfg), "split": split, "holes": sorted(df["hole_id"].unique()), "overall": metrics(df["y"], df["pred"], classes)}
    excl = [classes.index(c) for c in cfg.get("evaluation", {}).get("exclude_classes_for_summary", []) if c in classes]
    if excl:
        keep = ~df["y"].isin(excl)
        res[f"excluding_{'_'.join(classes[i] for i in excl)}"] = metrics(df.loc[keep, "y"], df.loc[keep, "pred"], classes)
    res["per_hole"] = {h: {"n": len(g), "accuracy": float(accuracy_score(g["y"], g["pred"]))} for h, g in df.groupby("hole_id")}
    nb = cfg.get("evaluation", {}).get("bootstrap", 0)
    if nb:
        res["bootstrap"] = bootstrap(df, classes, nb, cfg["project"]["seed"])
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    return res
