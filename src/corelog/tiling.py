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


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s))


def slice_id(hole: str, photo: str, row: str, k: int) -> str:
    return f"{_safe(hole)}__{_safe(photo)}__r{_safe(row)}__t{k:03d}"


def tile_row_masked(img: np.ndarray, mask: np.ndarray | None, depth_from: float, depth_to: float,
                    tile_h: int, tile_w: int, stride: int, min_last_fraction: float = 0.5,
                    trim_fraction: float = 0.0, direction: str = "ltr"):
    """Like tile_row, plus the mask (bool H, W) transformed identically.
    Returns [(tile, mask_tile | None, depth_from_m, depth_to_m), ...]."""
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
    recs, dropped, cache_path, cache_img = [], [], None, None
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
        pieces = tile_row_masked(img, mask, float(r["depth_from_m"]), float(r["depth_to_m"]),
                                 t["tile_height"], t["tile_width"], t["stride"],
                                 t.get("min_last_fraction", 0.5), t.get("trim_fraction", 0.0),
                                 r.get("direction", "ltr") if isinstance(r.get("direction", "ltr"), str) else "ltr")
        for k, (tile, mtile, d0, d1) in enumerate(pieces):
            sid = slice_id(r["hole_id"], r["photo_id"], r["row_id"], k)
            rock = float(mtile.mean()) if mtile is not None else float("nan")
            rec = {"slice_id": sid, "hole_id": r["hole_id"], "photo_id": r["photo_id"],
                   "row_id": r["row_id"], "tile_index": k, "depth_from_m": round(d0, 4),
                   "depth_to_m": round(d1, 4), "rock_fraction": round(rock, 4)}
            if mtile is not None and rock < min_rock:
                dropped.append(rec)
                continue
            rel = Path("tiles") / _safe(r["hole_id"]) / f"{sid}.{t.get('image_format', 'png')}"
            mrel = Path("tiles") / _safe(r["hole_id"]) / f"{sid}_mask.png" if mtile is not None else None
            if write_images:
                images.save_rgb(tile, out_root / rel)
                if mrel is not None:
                    Image.fromarray(mtile.astype(np.uint8) * 255).save(out_root / mrel)
            recs.append({**rec, "tile_path": rel.as_posix(), "mask_path": mrel.as_posix() if mrel else ""})
    if not recs:
        raise ValueError("no tiles produced: rows shorter than one tile, or all below tiling.min_rock_fraction")
    df = pd.DataFrame(recs).sort_values(["hole_id", "depth_from_m"]).reset_index(drop=True)
    if df["slice_id"].duplicated().any():
        raise ValueError("duplicate slice_id: check photo_id/row_id uniqueness in rows.csv")
    out = out_root / TILES_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    pd.DataFrame(dropped, columns=[c for c in df.columns if c not in ("tile_path", "mask_path")]).to_csv(
        out_root / DROPPED_CSV, index=False)
    return df


def read_tiles(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(output_root(cfg) / TILES_CSV, dtype={"hole_id": str, "photo_id": str, "row_id": str,
                                                           "mask_path": str}, keep_default_na=True)
