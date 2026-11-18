"""Dataset manifests (the contract with the research assistants) and their validation.

rows.csv (one line per core ROW visible in a photograph, or per pre-cropped row image)
    required: hole_id, photo_id, image_path, row_id, depth_from_m, depth_to_m
    optional: x0, y0, x1, y1   pixel box of the row in the image (omit -> whole image is the row)
              direction        ltr (default; depth increases left->right) | rtl
              exclude          1 to skip the row (broken, mislabelled...); default 0
              notes            free text
log_intervals.csv (geological journal, one line per logged interval)
    required: hole_id, depth_from_m, depth_to_m, lithology_code
    optional: description

Full specification with examples: docs/DATASET_MANIFEST.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import images

ROWS_REQUIRED = ["hole_id", "photo_id", "image_path", "row_id", "depth_from_m", "depth_to_m"]
ROWS_OPTIONAL = ["x0", "y0", "x1", "y1", "direction", "exclude", "notes"]
LOGS_REQUIRED = ["hole_id", "depth_from_m", "depth_to_m", "lithology_code"]
BBOX = ["x0", "y0", "x1", "y1"]
DEPTH_TOL_M = 0.005


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_markdown(self) -> str:
        out = ["# Dataset validation report", "",
               f"**Result: {'PASS' if self.ok else 'FAIL'}**: {len(self.errors)} errors, "
               f"{len(self.warnings)} warnings", ""]
        for title, items in (("Errors", self.errors), ("Warnings", self.warnings), ("Info", self.info)):
            out += [f"## {title}", ""] + ([f"- {i}" for i in items] or ["- none"]) + [""]
        return "\n".join(out)


def read_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"hole_id": str, "photo_id": str, "row_id": str, "image_path": str})
    df.columns = [c.strip() for c in df.columns]
    for c in ("direction",):
        if c in df:
            df[c] = df[c].fillna("ltr").astype(str).str.strip().str.lower()
    if "exclude" in df:
        df["exclude"] = df["exclude"].fillna(0).astype(int)
    return df


def read_logs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"hole_id": str, "lithology_code": str})
    df.columns = [c.strip() for c in df.columns]
    if "lithology_code" in df:
        df["lithology_code"] = df["lithology_code"].astype(str).str.strip()
    return df


def _overlaps(df: pd.DataFrame, what: str, rep: Report) -> None:
    for hole, g in df.sort_values(["hole_id", "depth_from_m"]).groupby("hole_id"):
        f, t = g["depth_from_m"].to_numpy(), g["depth_to_m"].to_numpy()
        ov = np.where(f[1:] < t[:-1] - DEPTH_TOL_M)[0]
        for i in ov[:10]:
            rep.errors.append(f"{what}: hole {hole}: intervals overlap at {f[i+1]:.3f} m "
                              f"(previous ends {t[i]:.3f} m)")
        gaps = np.where(f[1:] > t[:-1] + DEPTH_TOL_M)[0]
        if len(gaps):
            total = float((f[1:][gaps] - t[:-1][gaps]).sum())
            rep.info.append(f"{what}: hole {hole}: {len(gaps)} gaps totalling {total:.2f} m")


def _depth_checks(df: pd.DataFrame, what: str, rep: Report) -> pd.DataFrame:
    for c in ("depth_from_m", "depth_to_m"):
        bad = pd.to_numeric(df[c], errors="coerce").isna()
        if bad.any():
            rep.errors.append(f"{what}: {bad.sum()} non-numeric values in {c} (lines {list(df.index[bad][:5] + 2)})")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    inv = df["depth_to_m"] <= df["depth_from_m"]
    if inv.any():
        rep.errors.append(f"{what}: {inv.sum()} intervals with depth_to_m <= depth_from_m "
                          f"(lines {list(df.index[inv][:5] + 2)})")
    return df


def validate(rows: pd.DataFrame, logs: pd.DataFrame | None, data_root: Path,
             class_map: dict | None = None, check_images: bool = True) -> Report:
    rep = Report()
    miss = [c for c in ROWS_REQUIRED if c not in rows]
    if miss:
        rep.errors.append(f"rows.csv: missing required columns {miss}")
        return rep
    unknown = [c for c in rows.columns if c not in ROWS_REQUIRED + ROWS_OPTIONAL]
    if unknown:
        rep.info.append(f"rows.csv: extra columns ignored: {unknown}")
    empty = rows[ROWS_REQUIRED].isna().any(axis=1)
    if empty.any():
        rep.errors.append(f"rows.csv: {empty.sum()} lines with empty required fields (lines {list(rows.index[empty][:5] + 2)})")
    rows = _depth_checks(rows.copy(), "rows.csv", rep)
    dup = rows.duplicated(["photo_id", "row_id"])
    if dup.any():
        rep.errors.append(f"rows.csv: duplicate (photo_id, row_id) on lines {list(rows.index[dup][:5] + 2)}")
    if "direction" in rows:
        bad = ~rows["direction"].isin(["ltr", "rtl"])
        if bad.any():
            rep.errors.append(f"rows.csv: direction must be ltr or rtl (lines {list(rows.index[bad][:5] + 2)})")
    has_bbox = [c for c in BBOX if c in rows]
    if has_bbox and len(has_bbox) < 4:
        rep.errors.append("rows.csv: give all of x0, y0, x1, y1 or none of them")
    active = rows[rows.get("exclude", 0) == 0] if "exclude" in rows else rows
    _overlaps(active, "rows.csv", rep)

    if check_images:
        sizes: dict[str, tuple[int, int]] = {}
        for i, r in rows.iterrows():
            p = data_root / str(r["image_path"])
            if str(r["image_path"]) not in sizes:
                if not p.exists():
                    rep.errors.append(f"rows.csv line {i+2}: image not found: {r['image_path']}")
                    continue
                try:
                    sizes[str(r["image_path"])] = images.image_size(p)
                except Exception as e:  # noqa: BLE001
                    rep.errors.append(f"rows.csv line {i+2}: cannot read {r['image_path']}: {e}")
                    continue
            w, h = sizes[str(r["image_path"])]
            if len(has_bbox) == 4 and not r[BBOX].isna().all():
                x0, y0, x1, y1 = (r[c] for c in BBOX)
                if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                    rep.errors.append(f"rows.csv line {i+2}: box ({x0},{y0},{x1},{y1}) outside image {w}x{h}")
        rep.info.append(f"rows.csv: {len(rows)} rows, {rows['photo_id'].nunique()} photos, "
                        f"{len(sizes)} image files checked, holes: {sorted(rows['hole_id'].unique())}")
    if "exclude" in rows:
        rep.info.append(f"rows.csv: {int(rows['exclude'].sum())} rows marked exclude=1")

    if logs is None:
        rep.warnings.append("no log_intervals.csv: only SOM labelling / inference will be possible")
        return rep
    miss = [c for c in LOGS_REQUIRED if c not in logs]
    if miss:
        rep.errors.append(f"log_intervals.csv: missing required columns {miss}")
        return rep
    logs = _depth_checks(logs.copy(), "log_intervals.csv", rep)
    _overlaps(logs, "log_intervals.csv", rep)
    if class_map is not None:
        unmapped = sorted(set(logs["lithology_code"]) - set(map(str, class_map)))
        if unmapped:
            rep.warnings.append(f"log_intervals.csv: codes not in labels.class_map (treated as unlabeled): {unmapped}")
    for hole in sorted(set(rows["hole_id"]) - set(logs["hole_id"])):
        rep.warnings.append(f"hole {hole} has photos but no log intervals")
    for hole in sorted(set(logs["hole_id"]) - set(rows["hole_id"])):
        rep.warnings.append(f"hole {hole} has log intervals but no photos")
    for hole, g in active.groupby("hole_id"):
        lg = logs[logs["hole_id"] == hole]
        if lg.empty:
            continue
        cov = sum(max(0.0, min(t, b) - max(f, a)) for f, t in zip(g["depth_from_m"], g["depth_to_m"])
                  for a, b in zip(lg["depth_from_m"], lg["depth_to_m"]))
        tot = float((g["depth_to_m"] - g["depth_from_m"]).sum())
        rep.info.append(f"hole {hole}: {tot:.2f} m of photographed core, {100 * cov / max(tot, 1e-9):.1f}% covered by log intervals")
    return rep
