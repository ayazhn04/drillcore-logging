"""Inference on tiles and depth-indexed geological log generation (paper Fig. 16, extended).

predict_tiles -> per-tile class probabilities (any hole, labelled or not)
to_log        -> depth intervals: optional moving-average smoothing of probabilities,
                 non-core policy, merging of consecutive tiles with the same class,
                 absorption of intervals shorter than min_interval_m
compare_with_log -> depth-weighted agreement with the geologists' journal
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import data_path, output_root
from .evaluate import run_config
from .manifest import read_logs
from .runinfo import pick_device, provenance
from .tiling import read_tiles
from .train.common import load_checkpoint, loader, predict_proba

MAX_GAP_M = 0.5  # a gap larger than this (missing photo, core loss) starts a new interval


def predict_tiles(run_dir: Path, cfg: dict, holes: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    dev = pick_device(cfg["train"].get("device", "auto"))
    model, classes, _ = load_checkpoint(run_dir / "model.pt", dev)
    rcfg = run_config(run_dir, cfg)
    tiles = read_tiles(cfg)
    if holes:
        tiles = tiles[tiles["hole_id"].isin(holes)]
    prob = predict_proba(model, loader(tiles.reset_index(drop=True), rcfg, 256, shuffle=False), dev)
    df = tiles[["slice_id", "hole_id", "depth_from_m", "depth_to_m"]].reset_index(drop=True)
    for i, c in enumerate(classes):
        df[f"p_{c}"] = prob[:, i]
    return df, classes


def to_log(pred: pd.DataFrame, classes: list[str], window: int = 1, non_core_policy: str = "second_best",
           min_interval_m: float = 0.0, non_core: str = "non_core") -> pd.DataFrame:
    out = []
    pcols = [f"p_{c}" for c in classes]
    for hole, g in pred.sort_values(["hole_id", "depth_from_m"]).groupby("hole_id"):
        P = g[pcols].to_numpy(float)
        if window > 1:
            P = pd.DataFrame(P).rolling(window, center=True, min_periods=1).mean().to_numpy()
        order = np.argsort(-P, axis=1)
        cls = order[:, 0].copy()
        keep = np.ones(len(g), bool)
        if non_core in classes:
            nc = classes.index(non_core)
            if non_core_policy == "second_best":
                cls = np.where(cls == nc, order[:, 1], cls)
            elif non_core_policy == "drop":
                keep = cls != nc
        d0, d1 = g["depth_from_m"].to_numpy(), g["depth_to_m"].to_numpy()
        prob = P[np.arange(len(g)), cls]
        ints = []
        for i in np.where(keep)[0]:
            if ints and ints[-1]["cls"] == cls[i] and d0[i] - ints[-1]["to"] <= MAX_GAP_M:
                iv = ints[-1]
                iv["to"] = max(iv["to"], d1[i]); iv["p"].append(prob[i]); iv["n"] += 1
            else:
                ints.append({"cls": cls[i], "from": d0[i], "to": d1[i], "p": [prob[i]], "n": 1})
        ints = _absorb_short(ints, min_interval_m)
        out += [{"hole_id": hole, "depth_from_m": round(iv["from"], 3), "depth_to_m": round(iv["to"], 3),
                 "predicted_class": classes[iv["cls"]], "mean_probability": round(float(np.mean(iv["p"])), 4),
                 "n_tiles": iv["n"]} for iv in ints]
    return pd.DataFrame(out)


def _remerge(ints: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for iv in ints:
        if merged and merged[-1]["cls"] == iv["cls"] and iv["from"] - merged[-1]["to"] <= MAX_GAP_M:
            m = merged[-1]
            m["to"] = max(m["to"], iv["to"]); m["p"] = m["p"] + iv["p"]; m["n"] += iv["n"]
        else:
            merged.append(dict(iv))
    return merged


def _absorb_short(ints: list[dict], min_len: float) -> list[dict]:
    """Relabel the shortest interval below min_len with its longer adjacent neighbour's class,
    re-merge, repeat. Isolated short intervals (no adjacent neighbour) are kept."""
    if min_len <= 0:
        return ints
    while True:
        lens = [iv["to"] - iv["from"] for iv in ints]
        cand = [i for i, L in enumerate(lens) if L < min_len and not ints[i].get("_keep")]
        if not cand:
            return [{k: v for k, v in iv.items() if k != "_keep"} for iv in ints]
        i = min(cand, key=lambda k: lens[k])
        nb = [j for j in (i - 1, i + 1) if 0 <= j < len(ints)
              and ints[max(i, j)]["from"] - ints[min(i, j)]["to"] <= MAX_GAP_M]
        if not nb:
            ints[i]["_keep"] = True
            continue
        ints[i]["cls"] = ints[max(nb, key=lambda k: lens[k])]["cls"]
        ints = _remerge(ints)


def compare_with_log(pred_log: pd.DataFrame, logs: pd.DataFrame, class_map: dict, step: float = 0.01) -> dict:
    """Depth-weighted agreement between predicted and journal intervals (1-cm sampling)."""
    cmap = {str(k): v for k, v in class_map.items()}
    res = {}
    for hole, p in pred_log.groupby("hole_id"):
        lg = logs[logs["hole_id"] == hole].assign(cls=lambda d: d["lithology_code"].astype(str).map(cmap)).dropna(subset=["cls"])
        if lg.empty:
            continue
        z = np.arange(p["depth_from_m"].min(), p["depth_to_m"].max(), step) + step / 2
        def at(df, col):
            idx = np.searchsorted(df["depth_from_m"].to_numpy(), z, side="right") - 1
            ok = (idx >= 0) & (z < df["depth_to_m"].to_numpy()[np.clip(idx, 0, None)])
            return np.where(ok, df[col].to_numpy()[np.clip(idx, 0, None)], None)
        a = at(p.sort_values("depth_from_m"), "predicted_class")
        b = at(lg.sort_values("depth_from_m"), "cls")
        both = (a != None) & (b != None)  # noqa: E711
        if not both.any():
            continue
        res[hole] = {"compared_m": round(float(both.sum() * step), 2),
                     "depth_weighted_agreement": round(float((a[both] == b[both]).mean()), 4),
                     "n_pred_intervals": int(len(p)), "n_log_intervals": int(len(lg))}
    return res


def predict_log(run_dir: str | Path, cfg: dict, holes: list[str] | None = None, compare: bool = True) -> dict:
    run_dir = Path(run_dir)
    inf = cfg.get("inference", {})
    pred, classes = predict_tiles(run_dir, cfg, holes)
    out = output_root(cfg) / "predicted_logs" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out / "tile_predictions.csv", index=False)
    log = to_log(pred, classes, inf.get("smoothing_window_tiles", 1), inf.get("non_core_policy", "second_best"),
                 inf.get("min_interval_m", 0.0))
    log.to_csv(out / "predicted_log.csv", index=False)
    res = {**provenance(cfg), "output_dir": str(out), "holes": sorted(log["hole_id"].unique()), "n_intervals": len(log)}
    if compare:
        try:
            logs = read_logs(data_path(cfg, "log_intervals"))
            res["comparison_with_journal"] = compare_with_log(log, logs, cfg["labels"].get("class_map", {}))
        except FileNotFoundError:
            res["comparison_with_journal"] = "no log_intervals.csv"
    (out / "summary.json").write_text(json.dumps(res, indent=2, default=str))
    return res
