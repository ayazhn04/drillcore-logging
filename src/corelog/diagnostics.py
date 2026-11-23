"""Dataset inspection after make-tiles / make-labels: what the model will actually see.

Writes <output_root>/reports/inspection.md plus CSV tables. Nothing here trains a model.
Checks that matter with very few holes:
  * labelled tiles per class x hole and per class x split (train / val / test)
  * classes missing from the training split (cannot be learned) or from the test split
    (cannot be evaluated), and classes with fewer than `min_tiles` tiles or fewer than
    `min_intervals` independent journal intervals in a split
  * rock fraction of tiles (rock masks), dropped tiles, depth gaps between tiles
  * tile geometry: metres per tile, and whether tiles match model.input_size
  * tiling summary (reports/tiling_summary.json) for comparing tile geometries with
    `corelog compare-tiling`: tiles generated and kept, % rows with zero tiles, rock-mask
    coverage, depth represented per tile, and the share of ambiguous (unlabeled) tiles by cause
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import splits
from .config import classes as cfg_classes, data_path, output_root
from .manifest import read_logs
from .tiling import DROPPED_CSV, ROWS_TILING_CSV, TILING_RUN_JSON, read_tiles

SUMMARY_JSON = "reports/tiling_summary.json"


def _q(v: pd.Series, qs=(0.05, 0.5, 0.95)) -> dict:
    v = pd.to_numeric(v, errors="coerce").dropna()
    return {f"p{int(q * 100):02d}": round(float(v.quantile(q)), 4) for q in qs} if len(v) else {}


def _union_length(tiles: pd.DataFrame) -> float:
    """Metres of depth covered by tiles, overlaps counted once (the shifted-back last window
    of a row overlaps the previous tile)."""
    total = 0.0
    for _, g in tiles.sort_values("depth_from_m").groupby("hole_id"):
        end = -np.inf
        for a, b in zip(g["depth_from_m"], g["depth_to_m"]):
            if b > end:
                total += b - max(a, end)
                end = b
    return float(total)


def tiling_summary(cfg: dict, tiles: pd.DataFrame) -> dict:
    """Geometry-comparison metrics for one tiling run (one output_root)."""
    root = output_root(cfg)
    t = cfg["tiling"]
    out: dict = {"output_root": str(root), "tile_px": [t["tile_height"], t["tile_width"]],
                 "stride": t["stride"], "min_rock_fraction": t.get("min_rock_fraction", 0.0)}
    rp = root / ROWS_TILING_CSV
    rs = pd.read_csv(rp, dtype={"hole_id": str}) if rp.exists() else pd.DataFrame()
    dp = root / DROPPED_CSV
    n_drop = int(len(pd.read_csv(dp))) if dp.exists() else 0
    out["tiles_generated"] = int(len(tiles) + n_drop)
    out["tiles_kept"] = int(len(tiles))
    out["tiles_dropped_low_rock"] = n_drop
    if len(rs):
        out["rows"] = int(len(rs))
        out["rows_zero_windows_pct"] = round(100 * float((rs["n_windows"] == 0).mean()), 2)
        out["rows_zero_kept_tiles_pct"] = round(100 * float((rs["n_kept"] == 0).mean()), 2)
        out["row_height_px"] = _q(rs["height_px"])
        out["row_aspect_length_over_height"] = _q(rs["aspect"])
        out["per_hole"] = {h: {"rows": int(len(g)), "tiles_kept": int(g["n_kept"].sum()),
                               "rows_zero_kept_tiles_pct": round(100 * float((g["n_kept"] == 0).mean()), 2)}
                           for h, g in rs.groupby("hole_id")}
        span = (rs["depth_to_m"] - rs["depth_from_m"]).sum()
        out["depth_covered_by_kept_tiles_pct"] = round(100 * _union_length(tiles) / span, 2) if span > 0 else None
        if rs["rock_px_row"].notna().any():
            out["rock_in_kept_tiles_pct_of_row_rock"] = round(
                100 * float((rs["rock_covered_fraction"] * rs["rock_px_row"]).sum() / rs["rock_px_row"].sum()), 2)
    if "rock_fraction" in tiles and tiles["rock_fraction"].notna().any():
        out["tile_rock_fraction"] = _q(tiles["rock_fraction"])
    out["metres_per_tile"] = _q(tiles["depth_to_m"] - tiles["depth_from_m"])
    lp = root / "labels" / "labels_log.csv"
    if lp.exists():
        lab = tiles[["slice_id"]].merge(pd.read_csv(lp), on="slice_id", how="left")
        unl = cfg["labels"].get("unlabeled", "unlabeled")
        mp = float(cfg["labels"].get("min_purity", 0.8))
        u = lab["label"].fillna(unl) == unl
        cov = lab["log_coverage"].fillna(0)
        pur = lab["purity"].fillna(0)
        n = max(len(lab), 1)
        out["labels"] = {
            "tiles": int(len(lab)), "labelled": int((~u).sum()),
            "ambiguous_pct": round(100 * float(u.mean()), 2),
            "mixed_contact_pct": round(100 * float((u & (cov >= mp) & (pur < mp)).sum() / n), 2),
            "contact_buffer_pct": round(100 * float((u & (pur >= mp)).sum() / n), 2),
            "unlogged_or_unmapped_pct": round(100 * float((u & (cov < mp)).sum() / n), 2),
            "min_purity": mp, "contact_buffer_m": cfg["labels"].get("contact_buffer_m", 0.0)}
    out["data_provenance"] = _prov(cfg)
    return out


def _summary_without_tiles(root: Path) -> dict:
    run = json.loads((root / TILING_RUN_JSON).read_text())
    rs = pd.read_csv(root / ROWS_TILING_CSV)
    out = {"tile_px": run["tile_px"], "tiles_generated": int(rs["n_windows"].sum()) if len(rs) else 0,
           "tiles_kept": run["tiles_kept"], "tiles_dropped_low_rock": run["tiles_dropped_low_rock"],
           "data_provenance": "see tiles/tiling_run.json (inspect not run: no tiles)"}
    if len(rs):
        out["rows_zero_windows_pct"] = round(100 * float((rs["n_windows"] == 0).mean()), 2)
        out["rows_zero_kept_tiles_pct"] = round(100 * float((rs["n_kept"] == 0).mean()), 2)
    return out


COMPARE_ROWS = [  # (label, path in summary)
    ("tile size (h x w px)", ("tile_px",)),
    ("tiles generated", ("tiles_generated",)),
    ("tiles kept (after rock filter)", ("tiles_kept",)),
    ("tiles dropped (low rock)", ("tiles_dropped_low_rock",)),
    ("rows with zero windows (%)", ("rows_zero_windows_pct",)),
    ("rows with zero kept tiles (%)", ("rows_zero_kept_tiles_pct",)),
    ("tile rock fraction p05 / p50 / p95", ("tile_rock_fraction",)),
    ("row rock inside kept tiles (%)", ("rock_in_kept_tiles_pct_of_row_rock",)),
    ("photographed depth inside kept tiles (%)", ("depth_covered_by_kept_tiles_pct",)),
    ("metres per tile p05 / p50 / p95", ("metres_per_tile",)),
    ("ambiguous (unlabeled) tiles (%)", ("labels", "ambiguous_pct")),
    ("  mixed at a contact (%)", ("labels", "mixed_contact_pct")),
    ("  within contact buffer (%)", ("labels", "contact_buffer_pct")),
    ("  unlogged / unmapped (%)", ("labels", "unlogged_or_unmapped_pct")),
    ("data provenance", ("data_provenance",)),
]


def compare_tiling(roots: list) -> str:
    """Markdown table comparing reports/tiling_summary.json of several output roots."""
    sums = []
    for r in roots:
        f = Path(r) / SUMMARY_JSON
        if f.exists():
            sums.append(json.loads(f.read_text(encoding="utf-8")))
        elif (Path(r) / TILING_RUN_JSON).exists():
            sums.append(_summary_without_tiles(Path(r)))  # make-tiles ran but produced no (kept) tiles
        else:
            raise FileNotFoundError(f"{f} not found: run `corelog make-tiles` and `corelog inspect` "
                                    "for that output root first")

    head = "| metric | " + " | ".join(Path(r).name for r in roots) + " |"
    lines = [head, "|---|" + "---|" * len(roots)]
    lines += [f"| {lab} | " + " | ".join(_get(s, path) for s in sums) + " |" for lab, path in COMPARE_ROWS]
    return "\n".join(lines)


def _intervals_per_class(logs: pd.DataFrame, cmap: dict, tiles: pd.DataFrame, split: pd.Series) -> pd.DataFrame:
    """Number of distinct journal intervals (per class and split) that contain >= 1 labelled tile."""
    lg = logs.assign(cls=logs["lithology_code"].astype(str).map({str(k): v for k, v in cmap.items()})).dropna(subset=["cls"])
    out = []
    for (hole, part), g in tiles.assign(split=split.to_numpy()).groupby(["hole_id", "split"]):
        l = lg[lg["hole_id"] == hole]
        mid = ((g["depth_from_m"] + g["depth_to_m"]) / 2).to_numpy()
        for _, iv in l.iterrows():
            inside = (mid >= iv["depth_from_m"]) & (mid < iv["depth_to_m"]) & (g["label"].to_numpy() == iv["cls"])
            if inside.any():
                out.append({"split": part, "class": iv["cls"], "hole_id": hole})
    d = pd.DataFrame(out, columns=["split", "class", "hole_id"])
    return d.groupby(["class", "split"]).size().unstack(fill_value=0) if len(d) else pd.DataFrame()


def contact_sheets(df: pd.DataFrame, root: Path, out: Path, n: int, seed: int) -> list[str]:
    """One PNG per label with up to n random tiles (and their depth), for a visual check of
    scale, background, masks and label plausibility."""
    from PIL import Image, ImageDraw
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for lab, g in df.groupby("label"):
        g = g.sample(min(n, len(g)), random_state=seed).sort_values(["hole_id", "depth_from_m"])
        ims = [Image.open(root / p).convert("RGB") for p in g["tile_path"]]
        w, h = max(i.width for i in ims), max(i.height for i in ims)
        cols = max(1, min(4, 1200 // max(w, 1)))
        sheet = Image.new("RGB", (cols * (w + 6), ((len(ims) + cols - 1) // cols) * (h + 18) + 4), "white")
        d = ImageDraw.Draw(sheet)
        for k, (im, (_, r)) in enumerate(zip(ims, g.iterrows())):
            x, y = (k % cols) * (w + 6), (k // cols) * (h + 18)
            d.text((x + 2, y + 2), f"{r['hole_id']} {r['depth_from_m']:.2f}-{r['depth_to_m']:.2f} m", fill="black")
            sheet.paste(im, (x, y + 14))
        f = out / f"{str(lab).replace('/', '_')}.png"
        sheet.save(f)
        made.append(str(f))
    return made


def inspect(cfg: dict, min_tiles: int = 30, min_intervals: int = 3, sheets: int = 0) -> dict:
    root = output_root(cfg)
    rep_dir = root / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    tiles = read_tiles(cfg)
    unl = cfg["labels"].get("unlabeled", "unlabeled")
    lab_p = root / "labels" / "labels_log.csv"
    labels = pd.read_csv(lab_p)[["slice_id", "label"]] if lab_p.exists() else pd.DataFrame(columns=["slice_id", "label"])
    df = tiles.merge(labels, on="slice_id", how="left")
    df["label"] = df["label"].fillna(unl)
    cls = cfg_classes(cfg)
    warn, info = [], []
    try:
        df["split"] = splits.assign(df, cfg["split"], cfg["project"]["seed"]).to_numpy()
    except Exception as e:  # noqa: BLE001
        warn.append(f"split cannot be built: {e}")
        df["split"] = "none"

    by_hole = pd.crosstab(df["label"], df["hole_id"]).reindex(cls + [unl], fill_value=0)
    by_split = pd.crosstab(df["label"], df["split"]).reindex(index=cls + [unl], fill_value=0)
    for p in ("train", "val", "test", "none"):
        if p not in by_split:
            by_split[p] = 0
    by_split = by_split[["train", "val", "test", "none"]]
    by_hole.to_csv(rep_dir / "tiles_class_by_hole.csv")
    by_split.to_csv(rep_dir / "tiles_class_by_split.csv")

    ivs = pd.DataFrame()
    lp = data_path(cfg, "log_intervals")
    if lp.exists():
        ivs = _intervals_per_class(read_logs(lp), cfg["labels"].get("class_map", {}), df, df["split"])
        if len(ivs):
            ivs.to_csv(rep_dir / "journal_intervals_class_by_split.csv")
    for c in cls:
        if c == "non_core" and by_hole.loc[c].sum() == 0:
            continue
        tr, te, va = (int(by_split.loc[c, p]) for p in ("train", "test", "val"))
        if tr == 0:
            warn.append(f"class {c!r}: 0 labelled training tiles (cannot be learned)")
        elif tr < min_tiles:
            warn.append(f"class {c!r}: only {tr} labelled training tiles")
        if te == 0:
            warn.append(f"class {c!r}: 0 labelled test tiles (cannot be evaluated)")
        if va == 0 and tr > 0:
            info.append(f"class {c!r}: 0 validation tiles (early stopping blind to it)")
        if len(ivs) and c in ivs.index:
            for p in ("train", "test"):
                k = int(ivs.loc[c, p]) if p in ivs.columns else 0
                if 0 < k < min_intervals:
                    warn.append(f"class {c!r}: {k} journal interval(s) in {p}: tiles are not independent samples")
    if by_hole.loc[unl].sum():
        info.append(f"{int(by_hole.loc[unl].sum())} of {len(df)} tiles unlabeled (contacts, buffer, unmapped codes, gaps)")

    geo = {}
    span = df["depth_to_m"] - df["depth_from_m"]
    geo["tiles"] = len(df)
    geo["metres_per_tile_median"] = round(float(span.median()), 4)
    geo["metres_per_tile_5_95"] = [round(float(span.quantile(.05)), 4), round(float(span.quantile(.95)), 4)]
    t = cfg["tiling"]
    geo["tile_px"] = [t["tile_height"], t["tile_width"]]
    geo["model_input_px"] = list(cfg["model"]["input_size"])
    if geo["tile_px"] != geo["model_input_px"]:
        warn.append(f"tiles are {geo['tile_px']} px but model.input_size is {geo['model_input_px']}: tiles are resized")
    if "rock_fraction" in df and df["rock_fraction"].notna().any():
        rf = df["rock_fraction"]
        geo["rock_fraction_quantiles"] = {q: round(float(rf.quantile(q)), 3) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
        dp = root / DROPPED_CSV
        geo["dropped_low_rock_tiles"] = int(len(pd.read_csv(dp))) if dp.exists() else 0
        rk = df.groupby("label")["rock_fraction"].median().round(3).to_dict()
        geo["median_rock_fraction_by_label"] = rk
        if len(rk) > 1 and max(rk.values()) - min(rk.values()) > 0.15:
            warn.append(f"median rock fraction differs strongly between labels {rk}: the CNN can learn "
                        "piece outlines / rubble amount instead of rock appearance")
    gaps = []
    for hole, g in df.sort_values("depth_from_m").groupby("hole_id"):
        d = g["depth_from_m"].to_numpy()[1:] - g["depth_to_m"].to_numpy()[:-1]
        gaps.append({"hole_id": hole, "tiles": len(g), "depth_top_m": float(g["depth_from_m"].min()),
                     "depth_bottom_m": float(g["depth_to_m"].max()), "gaps_over_0.5m": int((d > 0.5).sum()),
                     "largest_gap_m": round(float(d.max()), 3) if len(d) else 0.0})
    geo["holes"] = gaps

    summ = tiling_summary(cfg, tiles)
    (root / SUMMARY_JSON).write_text(json.dumps(summ, indent=2, default=str), encoding="utf-8")
    if summ.get("rows_zero_kept_tiles_pct"):
        warn.append(f"{summ['rows_zero_kept_tiles_pct']} % of rows produced no kept tile "
                    f"(tile {summ['tile_px'][0]} x {summ['tile_px'][1]} px; see {ROWS_TILING_CSV})")
    md = ["# Dataset inspection", "", f"Data provenance: {_prov(cfg)}", "",
          f"**{len(warn)} warnings**", ""] + [f"- {w}" for w in warn] + ["", "## Info", ""] + [f"- {i}" for i in info]
    md += ["", "## Labelled tiles by class and hole", "", by_hole.to_string(),
           "", "## Labelled tiles by class and split", "", by_split.to_string(), ""]
    if len(ivs):
        md += ["## Journal intervals represented, by class and split", "", ivs.to_string(), ""]
    md += ["## Geometry", "", "```", json.dumps(geo, indent=2, default=str), "```", ""]
    md += ["## Tiling summary (for `corelog compare-tiling`)", "", compare_tiling_one(summ), ""]
    if sheets:
        made = contact_sheets(df, root, rep_dir / "tile_sheets", sheets, cfg["project"]["seed"])
        md += ["## Contact sheets", ""] + [f"- {m}" for m in made] + [""]
    (rep_dir / "inspection.md").write_text("\n".join(md), encoding="utf-8")
    return {"warnings": warn, "info": info, "geometry": geo, "tiling_summary": summ,
            "by_split": by_split.to_dict(), "report": str(rep_dir / "inspection.md")}


def _get(d, path) -> str:
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
    if isinstance(d, dict):
        return " / ".join(str(d.get(k, "")) for k in ("p05", "p50", "p95"))
    return " x ".join(map(str, d)) if isinstance(d, list) else ("" if d is None else str(d))


def compare_tiling_one(summ: dict) -> str:
    return "\n".join(["| metric | value |", "|---|---|"] + [f"| {lab} | {_get(summ, p)} |" for lab, p in COMPARE_ROWS])


def _prov(cfg):
    from .runinfo import provenance
    return provenance(cfg)["data_provenance"]
