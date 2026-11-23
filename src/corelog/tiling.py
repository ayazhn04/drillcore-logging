"""Cut core rows into fixed-size, depth-registered tiles (paper §3.1.3).

Each row from rows.csv is cropped (optional pixel box), optionally trimmed at both
ends, flipped if depth runs right-to-left, resized to `tile_height` keeping the aspect
ratio, and cut into `tile_width` windows. Depth is interpolated linearly along the row
(same approach as the paper's Fig. 16). A trailing partial window is kept by shifting
it back to end exactly at the row end (no padding), if it covers >= min_last_fraction.
Every tile gets a stable `slice_id`; all later steps join on slice_id, never on array
position (audit DL-2 / T4c).

Rock masks (optional; `mask_path` column or `tiling.mask_template`): the mask goes
through exactly the same crop / flip / trim / resize as the image. Each tile records
its `rock_fraction`; tiles below `tiling.min_rock_fraction` are not written (they are
listed in tiles/dropped_tiles.csv) and the mask tile is saved next to the image tile so
that descriptors can ignore the background (features.use_mask).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import images
from .config import data_path, data_root, output_root
from .manifest import BBOX, mask_path_for, read_rows

TILES_CSV = "tiles/tiles.csv"
DROPPED_CSV = "tiles/dropped_tiles.csv"
ROWS_TILING_CSV = "tiles/rows_tiling.csv"  # one line per manifest row: geometry, tiles made/kept, rock coverage
TILING_RUN_JSON = "tiles/tiling_run.json"  # tile geometry of this run, written even when no tile is produced


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s))


def slice_id(hole: str, photo: str, row: str, k: int) -> str:
    return f"{_safe(hole)}__{_safe(photo)}__r{_safe(row)}__t{k:03d}"


def tile_row_masked(img: np.ndarray, mask: np.ndarray | None, depth_from: float, depth_to: float,
                    tile_h: int, tile_w: int, stride: int, min_last_fraction: float = 0.5,
                    trim_fraction: float = 0.0, direction: str = "ltr", info: dict | None = None):
    """Like tile_row, plus the mask (bool H, W) transformed identically.
    Returns [(tile, mask_tile | None, depth_from_m, depth_to_m), ...]. If `info` is a dict it
    receives the resized row width (`scaled_w`), window starts and per-column rock pixels."""
    if mask is not None and mask.shape[:2] != img.shape[:2]:
        raise ValueError(f"mask shape {mask.shape[:2]} differs from image shape {img.shape[:2]}")
    if direction == "rtl":
        img = img[:, ::-1]
        mask = None if mask is None else mask[:, ::-1]
    W = img.shape[1]
    if trim_fraction > 0:
        t = int(round(W * trim_fraction))
        img = img[:, t:W - t]
        mask = None if mask is None else mask[:, t:W - t]
        span = depth_to - depth_from
        depth_from, depth_to = depth_from + span * trim_fraction, depth_to - span * trim_fraction
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * tile_h / h)))
    row = np.asarray(Image.fromarray(np.ascontiguousarray(img)).resize((new_w, tile_h), Image.Resampling.LANCZOS))
    mrow = None
    if mask is not None:  # area-weighted downsampling, then >= 50 % rock -> rock
        m8 = Image.fromarray(np.ascontiguousarray(mask).astype(np.uint8) * 255)
        mrow = np.asarray(m8.resize((new_w, tile_h), Image.Resampling.BOX)) >= 128
    m_per_px = (depth_to - depth_from) / new_w
    starts = list(range(0, new_w - tile_w + 1, stride))
    end_covered = (starts[-1] + tile_w) if starts else 0
    if new_w - end_covered >= min_last_fraction * tile_w and new_w >= tile_w:
        starts.append(new_w - tile_w)
    if info is not None:
        info.update(scaled_w=new_w, starts=starts, rock_cols=None if mrow is None else mrow.sum(0))
    return [(row[:, s:s + tile_w], None if mrow is None else mrow[:, s:s + tile_w],
             depth_from + s * m_per_px, depth_from + (s + tile_w) * m_per_px) for s in starts]


def tile_row(img: np.ndarray, depth_from: float, depth_to: float, tile_h: int, tile_w: int,
             stride: int, min_last_fraction: float = 0.5, trim_fraction: float = 0.0,
             direction: str = "ltr") -> list[tuple[np.ndarray, float, float]]:
    """Return [(tile, depth_from_m, depth_to_m), ...] for one row image (H, W, 3)."""
    return [(t, d0, d1) for t, _, d0, d1 in tile_row_masked(img, None, depth_from, depth_to, tile_h, tile_w,
                                                           stride, min_last_fraction, trim_fraction, direction)]


def load_mask(path: Path) -> np.ndarray:
    """Rock mask as bool (H, W): any pixel >= 128 in grey level is rock."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L")) >= 128


def make_tiles(cfg: dict, write_images: bool = True) -> pd.DataFrame:
    t = cfg["tiling"]
    rows = read_rows(data_path(cfg, "rows_manifest"))
    if "exclude" in rows:
        rows = rows[rows["exclude"] == 0]
    out_root = output_root(cfg)
    template, min_rock = t.get("mask_template"), float(t.get("min_rock_fraction", 0.0) or 0.0)
    recs, dropped, row_stats, cache_path, cache_img = [], [], [], None, None
    tw = t["tile_width"]
    for _, r in rows.sort_values(["image_path"]).iterrows():
        p = data_root(cfg) / r["image_path"]
        if p != cache_path:
            cache_img, cache_path = images.load_rgb(p), p
        img = cache_img
        mp = mask_path_for(r, template)
        mask = load_mask(data_root(cfg) / mp) if mp else None
        if all(c in r and not pd.isna(r[c]) for c in BBOX):
            x0, y0, x1, y1 = (int(r[c]) for c in BBOX)
            img = img[y0:y1, x0:x1]
            mask = None if mask is None else mask[y0:y1, x0:x1]
        info: dict = {}
        pieces = tile_row_masked(img, mask, float(r["depth_from_m"]), float(r["depth_to_m"]),
                                 t["tile_height"], tw, t["stride"],
                                 t.get("min_last_fraction", 0.5), t.get("trim_fraction", 0.0),
                                 r.get("direction", "ltr") if isinstance(r.get("direction", "ltr"), str) else "ltr",
                                 info=info)
        kept_cols = np.zeros(info["scaled_w"], bool)
        n_kept = 0
        for k, (tile, mtile, d0, d1) in enumerate(pieces):
            sid = slice_id(r["hole_id"], r["photo_id"], r["row_id"], k)
            rock = float(mtile.mean()) if mtile is not None else float("nan")
            rec = {"slice_id": sid, "hole_id": r["hole_id"], "photo_id": r["photo_id"],
                   "row_id": r["row_id"], "tile_index": k, "depth_from_m": round(d0, 4),
                   "depth_to_m": round(d1, 4), "rock_fraction": round(rock, 4)}
            if mtile is not None and rock < min_rock:
                dropped.append(rec)
                continue
            n_kept += 1
            kept_cols[info["starts"][k]:info["starts"][k] + tw] = True
            rel = Path("tiles") / _safe(r["hole_id"]) / f"{sid}.{t.get('image_format', 'png')}"
            mrel = Path("tiles") / _safe(r["hole_id"]) / f"{sid}_mask.png" if mtile is not None else None
            if write_images:
                images.save_rgb(tile, out_root / rel)
                if mrel is not None:
                    Image.fromarray(mtile.astype(np.uint8) * 255).save(out_root / mrel)
            recs.append({**rec, "tile_path": rel.as_posix(), "mask_path": mrel.as_posix() if mrel else ""})
        rc = info["rock_cols"]
        row_stats.append({
            "hole_id": r["hole_id"], "photo_id": r["photo_id"], "row_id": r["row_id"],
            "height_px": img.shape[0], "width_px": img.shape[1], "aspect": round(img.shape[1] / img.shape[0], 3),
            "depth_from_m": float(r["depth_from_m"]), "depth_to_m": float(r["depth_to_m"]),
            "scaled_width_px": info["scaled_w"], "n_windows": len(pieces), "n_kept": n_kept,
            "n_dropped_low_rock": len(pieces) - n_kept,
            # share of the row's rock (resized mask) that lies inside kept tiles; NaN without masks
            "rock_px_row": int(rc.sum()) if rc is not None else np.nan,
            "rock_covered_fraction": round(float(rc[kept_cols].sum() / rc.sum()), 4)
            if rc is not None and rc.sum() else np.nan})
    rs = pd.DataFrame(row_stats)
    out_root.joinpath("tiles").mkdir(parents=True, exist_ok=True)
    rs.to_csv(out_root / ROWS_TILING_CSV, index=False)
    import json
    (out_root / TILING_RUN_JSON).write_text(json.dumps(
        {"tile_px": [t["tile_height"], tw], "stride": t["stride"], "min_rock_fraction": min_rock,
         "rows": len(rs), "tiles_kept": len(recs), "tiles_dropped_low_rock": len(dropped)}, indent=2))
    if not recs:
        raise ValueError(_no_tiles_message(rs, t, len(dropped), min_rock))
    n_zero = int((rs["n_kept"] == 0).sum()) if len(rs) else 0
    if n_zero:
        print(f"[make-tiles] {n_zero} of {len(rs)} rows produced no kept tile (see {ROWS_TILING_CSV})")
    df = pd.DataFrame(recs).sort_values(["hole_id", "depth_from_m"]).reset_index(drop=True)
    if df["slice_id"].duplicated().any():
        raise ValueError("duplicate slice_id: check photo_id/row_id uniqueness in rows.csv")
    out = out_root / TILES_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    pd.DataFrame(dropped, columns=[c for c in df.columns if c not in ("tile_path", "mask_path")]).to_csv(
        out_root / DROPPED_CSV, index=False)
    return df


def _no_tiles_message(rs: pd.DataFrame, t: dict, n_dropped: int, min_rock: float) -> str:
    th, tw = t["tile_height"], t["tile_width"]
    need = tw / th
    if rs.empty:
        return "no tiles produced: rows.csv has no active rows (all excluded?)"
    short = int((rs["scaled_width_px"] < tw).sum())
    msg = [f"no tiles produced from {len(rs)} rows with tile {th} x {tw} px (height x width).",
           f"Each row is resized to {th} px high, so a row must be at least {need:.2f}x longer than high "
           f"to give one tile; row length/height here: median {rs['aspect'].median():.2f}, "
           f"max {rs['aspect'].max():.2f}; {short} rows are too short."]
    if n_dropped:
        msg.append(f"{n_dropped} tiles were dropped by tiling.min_rock_fraction={min_rock}.")
    msg.append("Per-row details: " + ROWS_TILING_CSV + " under paths.output_root. Options: the alternative "
               "geometry (--config configs/experiments/tiles_128.yaml with its own output_root), a smaller "
               "tiling.tile_width, or a lower tiling.min_rock_fraction.")
    return " ".join(msg)


def read_tiles(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(output_root(cfg) / TILES_CSV, dtype={"hole_id": str, "photo_id": str, "row_id": str,
                                                           "mask_path": str}, keep_default_na=True)
