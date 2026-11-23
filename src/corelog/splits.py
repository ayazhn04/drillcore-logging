"""Train/validation/test assignment by DRILL HOLE (no spatial leakage; audit §7, T4e).

Modes
  holes               explicit split.train_holes / val_holes / test_holes
  leave_one_hole_out  holes sorted; test = holes[fold], val = holes[fold+1], train = rest
  random              slice-level random 70/10/20 - ONLY to quantify the optimism of the
                      paper's protocol; never use it to report performance.

Few holes (e.g. two): set `split.val_from_train` = {fraction, buffer_m, position} and leave
val_holes empty. In every TRAINING hole, one contiguous depth block covering `fraction`
of the hole's tiled depth range becomes validation; tiles within `buffer_m` of that block
are dropped from both (split 'none'), so no training tile touches a validation tile.
`position` (0 = shallowest block, 1 = deepest) places the block. The TEST holes stay
whole and unseen. Validation is then within-hole: use it for early stopping and model
selection only, never as the reported result. leave_one_hole_out also accepts 2 holes
when val_from_train is set (test = holes[fold], train = the other).
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
        if _val_block(cfg_split) and any(v == "val" for v in m.values()):
            raise SplitError("use either split.val_holes or split.val_from_train, not both")
        return {h: m.get(h, "none") for h in holes}
    if mode == "leave_one_hole_out":
        k = int(cfg_split.get("fold", 0)) % max(len(holes), 1)
        if _val_block(cfg_split):
            if len(holes) < 2:
                raise SplitError("leave_one_hole_out needs >= 2 holes")
            return {h: "test" if h == holes[k] else "train" for h in holes}
        if len(holes) < 3:
            raise SplitError("leave_one_hole_out needs >= 3 holes (or set split.val_from_train)")
        test, val = holes[k], holes[(k + 1) % len(holes)]
        return {h: "test" if h == test else "val" if h == val else "train" for h in holes}
    raise SplitError(f"unknown split.mode {mode!r}")


def _val_block(cfg_split: dict) -> dict | None:
    v = cfg_split.get("val_from_train")
    if not v or not float(v.get("fraction", 0) or 0):
        return None
    f = float(v["fraction"])
    if not 0 < f < 1:
        raise SplitError("split.val_from_train.fraction must be in (0, 1)")
    return {"fraction": f, "buffer_m": float(v.get("buffer_m", 0.0) or 0.0),
            "position": min(max(float(v.get("position", 1.0)), 0.0), 1.0)}


def _carve_val(tiles: pd.DataFrame, s: pd.Series, vb: dict) -> pd.Series:
    s = s.copy()
    for hole in sorted(set(tiles.loc[s == "train", "hole_id"].astype(str))):
        idx = tiles.index[(tiles["hole_id"].astype(str) == hole) & (s == "train")]
        d0, d1 = tiles.loc[idx, "depth_from_m"], tiles.loc[idx, "depth_to_m"]
        lo, hi = float(d0.min()), float(d1.max())
        length = vb["fraction"] * (hi - lo)
        b0 = lo + vb["position"] * (hi - lo - length)
        b1 = b0 + length
        val = (d0 >= b0) & (d1 <= b1)
        near = ~val & (d1 > b0 - vb["buffer_m"]) & (d0 < b1 + vb["buffer_m"])
        s.loc[idx[val.to_numpy()]] = "val"
        s.loc[idx[near.to_numpy()]] = "none"
    return s


def assign(tiles: pd.DataFrame, cfg_split: dict, seed: int = 0) -> pd.Series:
    """Series aligned with tiles: train | val | test | none."""
    if cfg_split.get("mode") == "random":
        warnings.warn("split.mode=random: neighbouring tiles fall in different splits; results are optimistic")
        r = np.random.default_rng(seed).random(len(tiles))
        return pd.Series(np.where(r < 0.7, "train", np.where(r < 0.8, "val", "test")), index=tiles.index)
    m = hole_split(tiles["hole_id"].astype(str).unique().tolist(), cfg_split)
    s = tiles["hole_id"].astype(str).map(m)
    vb = _val_block(cfg_split)
    if vb:
        s = _carve_val(tiles, s, vb)
    check_disjoint(tiles, s, vb["buffer_m"] if vb else None)
    return s


def check_disjoint(tiles: pd.DataFrame, split: pd.Series, within_hole_buffer_m: float | None = None) -> None:
    """Test holes never share a hole with train/val. Train and val may share a hole only in
    val_from_train mode, and then no train tile may come closer than the buffer to a val tile."""
    sets = {p: set(tiles.loc[split == p, "hole_id"].astype(str)) for p in ("train", "val", "test")}
    pairs = (("train", "test"), ("val", "test")) + ((("train", "val"),) if within_hole_buffer_m is None else ())
    for a, b in pairs:
        if sets[a] & sets[b]:
            raise SplitError(f"holes shared between {a} and {b}: {sorted(sets[a] & sets[b])}")
    if within_hole_buffer_m is not None:
        for hole in sets["train"] & sets["val"]:
            h = tiles["hole_id"].astype(str) == hole
            tr, va = tiles[h & (split == "train")], tiles[h & (split == "val")]
            gap = np.maximum(va["depth_from_m"].to_numpy()[:, None] - tr["depth_to_m"].to_numpy()[None],
                             tr["depth_from_m"].to_numpy()[None] - va["depth_to_m"].to_numpy()[:, None])
            if gap.size and gap.min() < within_hole_buffer_m - 1e-6:
                raise SplitError(f"hole {hole}: a train tile is {gap.min():.3f} m from a val tile "
                                 f"(buffer {within_hole_buffer_m} m)")
