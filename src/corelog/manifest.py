"""Dataset manifests (the contract with the research assistants) and their validation.

rows.csv (one line per core ROW visible in a photograph, or per pre-cropped row image)
    required: hole_id, photo_id, image_path, row_id, depth_from_m, depth_to_m
    optional: x0, y0, x1, y1   pixel box of the row in the image (omit -> whole image is the row)
              direction        ltr (default; depth increases left->right) | rtl
              exclude          1 to skip the row (broken, mislabelled...); default 0
              mask_path        rock mask (white = rock), same size as the image; or derive it
                               for every row with `tiling.mask_template` (see mask_path_for)
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
ROWS_OPTIONAL = ["x0", "y0", "x1", "y1", "direction", "exclude", "mask_path", "notes"]
LOGS_REQUIRED = ["hole_id", "depth_from_m", "depth_to_m", "lithology_code"]
BBOX = ["x0", "y0", "x1", "y1"]
DEPTH_TOL_M = 0.005
SCALE_TOL = (0.75, 1.33)  # row px-per-metre relative to the hole median before a warning


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


def mask_path_for(row, template: str | None) -> str | None:
    """Relative path of the rock mask of one rows.csv line, or None.

    A non-empty `mask_path` column wins. Otherwise `template` (config `tiling.mask_template`)
    is formatted with: hole_id, photo_id, row_id, image_stem (file name without extension),
    image_name, image_dir (folder of image_path) and image_subdir (image_dir without its
    first component). Example for photos/<hole>/<row>.png -> masks/<hole>/<row>_mask.png:
    "masks/{image_subdir}/{image_stem}_mask.png".
    """
    mp = row.get("mask_path") if hasattr(row, "get") else None
    if isinstance(mp, str) and mp.strip():
        return mp.strip()
    if not template:
        return None
    ip = Path(str(row["image_path"]))
    parts = ip.parent.parts
    return template.format(hole_id=row["hole_id"], photo_id=row["photo_id"], row_id=row["row_id"],
                           image_stem=ip.stem, image_name=ip.name, image_dir=ip.parent.as_posix(),
                           image_subdir=Path(*parts[1:]).as_posix() if len(parts) > 1 else "")


def read_rows(path: Path) -> pd.DataFrame:
    # utf-8-sig: Excel "CSV UTF-8" writes a byte-order mark that would otherwise corrupt the first header
    df = pd.read_csv(path, dtype={"hole_id": str, "photo_id": str, "row_id": str, "image_path": str,
                                  "mask_path": str}, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    for c in ("direction",):
        if c in df:
            df[c] = df[c].fillna("ltr").astype(str).str.strip().str.lower()
    if "exclude" in df:
        df["exclude"] = df["exclude"].fillna(0).astype(int)
    return df


def read_logs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"hole_id": str, "lithology_code": str}, encoding="utf-8-sig",
                     keep_default_na=False, na_values={"depth_from_m": [""], "depth_to_m": [""]})
    df.columns = [c.strip() for c in df.columns]
    if "lithology_code" in df:  # codes kept verbatim (e.g. "NA" stays a code); only whitespace stripped
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


def class_metres(rows: pd.DataFrame, logs: pd.DataFrame, class_map: dict | None = None) -> pd.DataFrame:
    """Metres of photographed (non-excluded) core per hole and journal code / model class.

    Only the overlap of log intervals with rows.csv depth ranges is counted, so the table
    shows what can actually become labelled tiles. Columns: hole_id, code, class, metres.
    """
    rows = rows.copy()
    if "exclude" in rows:
        rows = rows[rows["exclude"] == 0]
    cmap = {str(k): v for k, v in (class_map or {}).items()}
    out = []
    for hole, lg in logs.groupby("hole_id"):
        rg = rows[rows["hole_id"] == hole]
        r0, r1 = rg["depth_from_m"].to_numpy(float)[None], rg["depth_to_m"].to_numpy(float)[None]
        for code, cg in lg.groupby("lithology_code"):
            a0, a1 = cg["depth_from_m"].to_numpy(float)[:, None], cg["depth_to_m"].to_numpy(float)[:, None]
            m = float(np.clip(np.minimum(a1, r1) - np.maximum(a0, r0), 0, None).sum()) if len(rg) else 0.0
            out.append({"hole_id": hole, "code": code, "class": cmap.get(str(code), ""),
                        "n_intervals": len(cg), "metres": round(m, 3)})
    return pd.DataFrame(out, columns=["hole_id", "code", "class", "n_intervals", "metres"])


def _class_checks(rows: pd.DataFrame, logs: pd.DataFrame, class_map: dict, classes: list[str] | None,
                  rep: Report) -> pd.DataFrame:
    cm = class_metres(rows, logs, class_map)
    mapped = cm[cm["class"] != ""]
    if mapped.empty:
        return cm
    tab = mapped.pivot_table(index="class", columns="hole_id", values="metres", aggfunc="sum", fill_value=0.0)
    holes = sorted(rows["hole_id"].astype(str).unique())
    tab = tab.reindex(columns=holes, fill_value=0.0)
    for c in classes or []:
        if c not in tab.index and c != "non_core":
            rep.warnings.append(f"class {c!r} has no photographed core in any hole (no training data)")
    for c, r in tab.iterrows():
        absent = [h for h in holes if r[h] <= 0]
        if absent:
            rep.warnings.append(f"class {c!r} is absent from hole(s) {absent}: it cannot be both trained and "
                                f"tested by hole ({', '.join(f'{h}: {r[h]:.1f} m' for h in holes)})")
    rep.info.append("photographed metres per class and hole: " +
                    "; ".join(f"{c}: " + ", ".join(f"{h} {r[h]:.1f} m" for h in holes) for c, r in tab.iterrows()))
    return cm


def validate(rows: pd.DataFrame, logs: pd.DataFrame | None, data_root: Path,
             class_map: dict | None = None, check_images: bool = True,
             mask_template: str | None = None, classes: list[str] | None = None) -> Report:
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
    elif has_bbox:
        n_set = rows[BBOX].notna().sum(axis=1)
        part = (n_set > 0) & (n_set < 4)
        if part.any():
            rep.errors.append(f"rows.csv: {part.sum()} lines with only some of x0, y0, x1, y1 filled "
                              f"(lines {list(rows.index[part][:5] + 2)}): fill all four or leave all empty")
        if (n_set == 0).all():
            rep.info.append("rows.csv: no pixel boxes; every image is treated as one pre-cropped row")
    active = rows[rows.get("exclude", 0) == 0] if "exclude" in rows else rows
    _overlaps(active, "rows.csv", rep)

    use_masks = ("mask_path" in rows and rows["mask_path"].notna().any()) or bool(mask_template)
    if check_images:
        sizes: dict[str, tuple[int, int]] = {}
        scale = []  # per active row: hole, height px, width px, px per metre along the row
        n_masks = 0
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
            rw, rh = w, h
            if len(has_bbox) == 4 and not r[BBOX].isna().all():
                x0, y0, x1, y1 = (r[c] for c in BBOX)
                if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                    rep.errors.append(f"rows.csv line {i+2}: box ({x0},{y0},{x1},{y1}) outside image {w}x{h}")
                rw, rh = x1 - x0, y1 - y0
            if use_masks:
                mp = mask_path_for(r, mask_template)
                if mp is None:
                    rep.errors.append(f"rows.csv line {i+2}: no mask_path (other rows have masks)")
                elif not (data_root / mp).exists():
                    rep.errors.append(f"rows.csv line {i+2}: mask not found: {mp}")
                else:
                    try:
                        if images.image_size(data_root / mp) != (w, h):
                            rep.errors.append(f"rows.csv line {i+2}: mask {mp} size {images.image_size(data_root / mp)} "
                                              f"differs from image size {(w, h)}")
                        n_masks += 1
                    except Exception as e:  # noqa: BLE001
                        rep.errors.append(f"rows.csv line {i+2}: cannot read mask {mp}: {e}")
            span = float(r["depth_to_m"]) - float(r["depth_from_m"]) if pd.notna(r["depth_to_m"]) else float("nan")
            if not r.get("exclude", 0) and span > 0:
                scale.append((str(r["hole_id"]), rh, rw, rw / span, i + 2))
        rep.info.append(f"rows.csv: {len(rows)} rows, {rows['photo_id'].nunique()} photos, "
                        f"{len(sizes)} image files checked, holes: {sorted(rows['hole_id'].unique())}")
        if use_masks:
            rep.info.append(f"rows.csv: {n_masks} rock masks found and size-checked")
        if scale:
            sc = pd.DataFrame(scale, columns=["hole", "h", "w", "ppm", "line"])
            for hole, g in sc.groupby("hole"):
                med = g["ppm"].median()
                off = g[(g["ppm"] < SCALE_TOL[0] * med) | (g["ppm"] > SCALE_TOL[1] * med)]
                if len(off):
                    rep.warnings.append(
                        f"hole {hole}: {len(off)} rows whose image length per metre of depth is outside "
                        f"{SCALE_TOL[0]:.2f}-{SCALE_TOL[1]:.2f} x the hole median ({med:.0f} px/m), lines "
                        f"{off['line'].tolist()[:8]}. If all photos share one scale, the row depth interval "
                        "does not match the core in the image (core loss, wrong box interval, depth split, "
                        "or a row cropped to the rock): tiles there get wrong depths")
                rep.info.append(f"hole {hole}: row height {g['h'].median():.0f} px (min {g['h'].min()}, max {g['h'].max()}), "
                                f"row length {g['w'].median():.0f} px, scale {g['ppm'].median():.0f} px/m "
                                f"(5-95%: {g['ppm'].quantile(.05):.0f}-{g['ppm'].quantile(.95):.0f})")
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
    blank = logs["lithology_code"].isin(["", "nan", "None"])
    if blank.any():
        rep.warnings.append(f"log_intervals.csv: {blank.sum()} intervals with an empty code (treated as unlabeled; "
                            f"lines {list(logs.index[blank][:5] + 2)})")
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
    if class_map:
        _class_checks(rows, logs, class_map, classes, rep)
    return rep
