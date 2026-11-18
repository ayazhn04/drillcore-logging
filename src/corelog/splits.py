"""Train/validation/test assignment by DRILL HOLE (no spatial leakage; audit §7, T4e).

Modes
  holes               explicit split.train_holes / val_holes / test_holes
  leave_one_hole_out  holes sorted; test = holes[fold], val = holes[fold+1], train = rest
  random              slice-level random 70/10/20 - ONLY to quantify the optimism of the
                      paper's protocol; never use it to report performance.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


class SplitError(ValueError):
    pass


def hole_split(holes: list[str], cfg_split: dict) -> dict[str, str]:
    mode = cfg_split.get("mode", "holes")
    holes = sorted(set(map(str, holes)))
    if mode == "holes":
        m = {}
        for part in ("train", "val", "test"):
            for h in map(str, cfg_split.get(f"{part}_holes", []) or []):
                if h in m:
                    raise SplitError(f"hole {h} assigned to both {m[h]} and {part}")
                m[h] = part
        unknown = set(m) - set(holes)
        if unknown:
            warnings.warn(f"split lists holes with no tiles: {sorted(unknown)}")
        if not any(v == "train" for v in m.values()):
            raise SplitError("no training holes configured (split.train_holes)")
        return {h: m.get(h, "none") for h in holes}
    if mode == "leave_one_hole_out":
        if len(holes) < 3:
            raise SplitError("leave_one_hole_out needs >= 3 holes")
        k = int(cfg_split.get("fold", 0)) % len(holes)
        test, val = holes[k], holes[(k + 1) % len(holes)]
        return {h: "test" if h == test else "val" if h == val else "train" for h in holes}
    raise SplitError(f"unknown split.mode {mode!r}")


def assign(tiles: pd.DataFrame, cfg_split: dict, seed: int = 0) -> pd.Series:
    """Series aligned with tiles: train | val | test | none."""
    if cfg_split.get("mode") == "random":
        warnings.warn("split.mode=random: neighbouring tiles fall in different splits; results are optimistic")
        r = np.random.default_rng(seed).random(len(tiles))
        return pd.Series(np.where(r < 0.7, "train", np.where(r < 0.8, "val", "test")), index=tiles.index)
    m = hole_split(tiles["hole_id"].astype(str).unique().tolist(), cfg_split)
    s = tiles["hole_id"].astype(str).map(m)
    check_disjoint(tiles, s)
    return s


def check_disjoint(tiles: pd.DataFrame, split: pd.Series) -> None:
    sets = {p: set(tiles.loc[split == p, "hole_id"]) for p in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if sets[a] & sets[b]:
            raise SplitError(f"holes shared between {a} and {b}: {sorted(sets[a] & sets[b])}")
