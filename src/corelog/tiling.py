"""Cut core rows into fixed-size, depth-registered tiles (paper §3.1.3).

Each row from rows.csv is cropped (optional pixel box), optionally trimmed at both
ends, flipped if depth runs right-to-left, resized to `tile_height` keeping the aspect
ratio, and cut into `tile_width` windows. Depth is interpolated linearly along the row
(same approach as the paper's Fig. 16). A trailing partial window is kept by shifting
it back to end exactly at the row end (no padding), if it covers >= min_last_fraction.
Every tile gets a stable `slice_id`; all later steps join on slice_id, never on array
position (audit DL-2 / T4c).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import images
from .config import data_path, data_root, output_root
from .manifest import BBOX, read_rows

TILES_CSV = "tiles/tiles.csv"


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s))


def slice_id(hole: str, photo: str, row: str, k: int) -> str:
    return f"{_safe(hole)}__{_safe(photo)}__r{_safe(row)}__t{k:03d}"


def tile_row(img: np.ndarray, depth_from: float, depth_to: float, tile_h: int, tile_w: int,
             stride: int, min_last_fraction: float = 0.5, trim_fraction: float = 0.0,
             direction: str = "ltr") -> list[tuple[np.ndarray, float, float]]:
    """Return [(tile, depth_from_m, depth_to_m), ...] for one row image (H, W, 3)."""
    if direction == "rtl":
        img = img[:, ::-1]
    W = img.shape[1]
    if trim_fraction > 0:
        t = int(round(W * trim_fraction))
        img = img[:, t:W - t]
        span = depth_to - depth_from
        depth_from, depth_to = depth_from + span * trim_fraction, depth_to - span * trim_fraction
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * tile_h / h)))
    row = np.asarray(Image.fromarray(np.ascontiguousarray(img)).resize((new_w, tile_h), Image.Resampling.LANCZOS))
    m_per_px = (depth_to - depth_from) / new_w
    starts = list(range(0, new_w - tile_w + 1, stride))
    end_covered = (starts[-1] + tile_w) if starts else 0
    if new_w - end_covered >= min_last_fraction * tile_w and new_w >= tile_w:
        starts.append(new_w - tile_w)
    return [(row[:, s:s + tile_w], depth_from + s * m_per_px, depth_from + (s + tile_w) * m_per_px)
            for s in starts]


def make_tiles(cfg: dict, write_images: bool = True) -> pd.DataFrame:
    t = cfg["tiling"]
    rows = read_rows(data_path(cfg, "rows_manifest"))
    if "exclude" in rows:
        rows = rows[rows["exclude"] == 0]
    out_root = output_root(cfg)
    recs, cache_path, cache_img = [], None, None
    for _, r in rows.sort_values(["image_path"]).iterrows():
        p = data_root(cfg) / r["image_path"]
        if p != cache_path:
            cache_img, cache_path = images.load_rgb(p), p
        img = cache_img
        if all(c in r and not pd.isna(r[c]) for c in BBOX):
            x0, y0, x1, y1 = (int(r[c]) for c in BBOX)
            img = img[y0:y1, x0:x1]
        pieces = tile_row(img, float(r["depth_from_m"]), float(r["depth_to_m"]),
                          t["tile_height"], t["tile_width"], t["stride"],
                          t.get("min_last_fraction", 0.5), t.get("trim_fraction", 0.0),
                          r.get("direction", "ltr") if isinstance(r.get("direction", "ltr"), str) else "ltr")
        for k, (tile, d0, d1) in enumerate(pieces):
            sid = slice_id(r["hole_id"], r["photo_id"], r["row_id"], k)
            rel = Path("tiles") / _safe(r["hole_id"]) / f"{sid}.{t.get('image_format', 'png')}"
            if write_images:
                images.save_rgb(tile, out_root / rel)
            recs.append({"slice_id": sid, "hole_id": r["hole_id"], "photo_id": r["photo_id"],
                         "row_id": r["row_id"], "tile_index": k, "depth_from_m": round(d0, 4),
                         "depth_to_m": round(d1, 4), "tile_path": rel.as_posix()})
    df = pd.DataFrame(recs).sort_values(["hole_id", "depth_from_m"]).reset_index(drop=True)
    if df["slice_id"].duplicated().any():
        raise ValueError("duplicate slice_id: check photo_id/row_id uniqueness in rows.csv")
    out = out_root / TILES_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def read_tiles(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(output_root(cfg) / TILES_CSV, dtype={"hole_id": str, "photo_id": str, "row_id": str})
