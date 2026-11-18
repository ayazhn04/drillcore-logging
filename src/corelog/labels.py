"""Tile labels from the geological journals (depth overlap) and label-source merging.

A tile receives a log label when one mapped class covers >= `min_purity` of its depth
span; otherwise it is `unlabeled` (contacts, gaps, unmapped codes) and can only be used
as unlabeled data for Mean Teacher. Label files are keyed by slice_id.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import classes as cfg_classes, data_path, output_root
from .manifest import read_logs

LOG_LABELS_CSV = "labels/labels_log.csv"
SOM_LABELS_CSV = "labels/labels_som.csv"


def log_labels(tiles: pd.DataFrame, logs: pd.DataFrame, class_map: dict, classes: list[str],
               min_purity: float = 0.8, unlabeled: str = "unlabeled") -> pd.DataFrame:
    cmap = {str(k): v for k, v in class_map.items()}
    bad = set(cmap.values()) - set(classes)
    if bad:
        raise ValueError(f"class_map targets not in labels.classes: {sorted(bad)}")
    logs = logs.assign(cls=logs["lithology_code"].astype(str).map(cmap)).dropna(subset=["cls"])
    out = []
    for hole, tg in tiles.groupby("hole_id"):
        lg = logs[logs["hole_id"] == hole]
        a0, a1 = tg["depth_from_m"].to_numpy()[:, None], tg["depth_to_m"].to_numpy()[:, None]
        length = (a1 - a0)[:, 0]
        cover = np.zeros((len(tg), len(classes)))
        if len(lg):
            b0, b1 = lg["depth_from_m"].to_numpy()[None], lg["depth_to_m"].to_numpy()[None]
            ov = np.clip(np.minimum(a1, b1) - np.maximum(a0, b0), 0, None)  # (tiles, intervals)
            ci = np.array([classes.index(c) for c in lg["cls"]])
            for j in range(len(classes)):
                cover[:, j] = ov[:, ci == j].sum(1)
        frac = cover / length[:, None]
        best = frac.argmax(1)
        purity = frac.max(1)
        lab = np.where(purity >= min_purity, np.array(classes)[best], unlabeled)
        out.append(pd.DataFrame({"slice_id": tg["slice_id"].to_numpy(), "label": lab,
                                 "purity": purity.round(4), "log_coverage": frac.sum(1).round(4)}))
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["slice_id", "label", "purity", "log_coverage"])
    df["source"] = "log"
    return df


def make_log_labels(cfg: dict, tiles: pd.DataFrame) -> pd.DataFrame:
    lab = cfg["labels"]
    df = log_labels(tiles, read_logs(data_path(cfg, "log_intervals")), lab.get("class_map", {}),
                    cfg_classes(cfg), lab.get("min_purity", 0.8), lab.get("unlabeled", "unlabeled"))
    p = output_root(cfg) / LOG_LABELS_CSV
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return df


def load_labels(cfg: dict) -> pd.DataFrame:
    """slice_id -> label according to split.label_source (log | som | log_then_som)."""
    src = cfg.get("split", {}).get("label_source", "log")
    unl = cfg["labels"].get("unlabeled", "unlabeled")
    root = output_root(cfg)

    def rd(rel):
        p = root / rel
        if not p.exists():
            raise FileNotFoundError(f"{p} not found: run the step that creates it first")
        return pd.read_csv(p)[["slice_id", "label"]]

    if src == "log":
        return rd(LOG_LABELS_CSV)
    if src == "som":
        return rd(SOM_LABELS_CSV)
    if src == "log_then_som":
        a, b = rd(LOG_LABELS_CSV), rd(SOM_LABELS_CSV).rename(columns={"label": "som"})
        m = a.merge(b, on="slice_id", how="outer")
        m["label"] = m["label"].where(m["label"].notna() & (m["label"] != unl), m["som"]).fillna(unl)
        return m[["slice_id", "label"]]
    raise ValueError(f"unknown split.label_source {src!r}")
