#!/usr/bin/env python3
"""Private drill-core prediction galleries for the corelog repository.

Requires: pandas, Pillow (already installed in corelog_lab).
No training or model loading. It joins eval_<split>/predictions.csv with
<output_root>/tiles/tiles.csv on slice_id to find the actual tile image files.

Examples:
  python make_prediction_montages.py --run /media/ra/KINGSTON/drillcore-logging-work/outputs/bassgold_2026-11_t128/runs/pilot_t128_extended_20261123_145424
  python make_prediction_montages.py --run .../runs/diag_weighted_60_mid_20261123_151926 --split val

Outputs remain under the run directory on Kingston. These images are derived
from potentially confidential drill-core photos: DO NOT push them to public Git.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

BG = "#F3F5F8"
WHITE = "#FFFFFF"
INK = "#172235"
MUTED = "#58677B"
GREEN = "#147D52"
RED = "#C63D45"
BORDER = "#D9E0E9"


def font(size: int, bold: bool = False):
    names = (["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"] if bold else
             ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"])
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


F_TITLE = font(27, True)
F_SUB = font(15)
F_LINE = font(16)
F_LABEL = font(17, True)
F_TINY = font(13)


def short(s):
    return str(s).removeprefix("lith_")


def sample_spread(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Sample across hole/depth blocks rather than a wall of adjacent tiles."""
    if df.empty or len(df) <= n:
        return df.sort_values(["hole_id", "depth_from_m"]).copy()
    x = df.copy()
    x["_depth_block"] = (x["depth_from_m"] // 2).astype(int)
    groups = [g.sample(frac=1, random_state=seed).index.tolist()
              for _, g in x.groupby(["hole_id", "_depth_block"], sort=True)]
    selected = []
    while len(selected) < n and any(groups):
        for g in groups:
            if g and len(selected) < n:
                selected.append(g.pop())
    return x.loc[selected].sort_values(["hole_id", "depth_from_m"]).drop(columns="_depth_block")


def make_sheet(df: pd.DataFrame, title: str, subtitle: str, dest: Path,
               *, max_items: int = 24, columns: int = 4, seed: int = 42):
    df = sample_spread(df, max_items, seed)
    card_w, card_h, gap, margin = 296, 364, 15, 24
    cols = max(1, columns)
    n_rows = max(1, math.ceil(len(df) / cols))
    top = 135
    W = margin * 2 + cols * card_w + (cols - 1) * gap
    H = top + n_rows * card_h + (n_rows - 1) * gap + 35
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 19), title, font=F_TITLE, fill=INK)
    draw.text((margin, 63), subtitle, font=F_SUB, fill=MUTED)
    draw.text((margin, 89), f"Displayed {len(df)} of {int(df.attrs.get('count', len(df)))} tiles. "
                               "Depth-spread sampling; labels are journal labels.", font=F_TINY, fill=MUTED)
    if df.empty:
        draw.text((margin, top + 28), "No examples available in this evaluation split.", font=F_LINE, fill=MUTED)
        canvas.save(dest)
        return

    for idx, (_, r) in enumerate(df.iterrows()):
        cx = margin + (idx % cols) * (card_w + gap)
        cy = top + (idx // cols) * (card_h + gap)
        correct = bool(r["correct"])
        edge = GREEN if correct else RED
        draw.rounded_rectangle([cx, cy, cx + card_w, cy + card_h], radius=13,
                               fill=WHITE, outline=BORDER, width=2)
        draw.rounded_rectangle([cx + 13, cy + 12, cx + card_w - 13, cy + 214],
                               radius=9, fill="#E9EDF3")
        img_path = Path(r["full_path"])
        try:
            with Image.open(img_path) as source:
                im = source.convert("RGB")
                im.thumbnail((card_w - 40, 186), Image.Resampling.LANCZOS)
                ix = cx + (card_w - im.width) // 2
                iy = cy + 21 + (186 - im.height) // 2
                canvas.paste(im, (ix, iy))
        except (OSError, ValueError):
            draw.text((cx + 22, cy + 90), "Unreadable tile image", font=F_LINE, fill=RED)
        draw.rounded_rectangle([cx + 12, cy + 224, cx + 105, cy + 250],
                               radius=6, fill=edge)
        draw.text((cx + 19, cy + 229), "CORRECT" if correct else "WRONG",
                  font=F_TINY, fill=WHITE)
        draw.text((cx + 12, cy + 257), f"True: {short(r['label'])}", font=F_LABEL, fill=INK)
        draw.text((cx + 12, cy + 282), f"Pred: {short(r['predicted'])}", font=F_LINE, fill=INK)
        conf = r.get("confidence", float("nan"))
        conf_s = f"Model confidence: {conf:.1%}" if pd.notna(conf) else "Model confidence: n/a"
        depth_text = f"{float(r['depth_from_m']):.2f}–{float(r['depth_to_m']):.2f} m"
        draw.text((cx + 12, cy + 306), f"{short(r['hole_id'])} | {depth_text}",
                  font=F_TINY, fill=MUTED)
        draw.text((cx + 12, cy + 327), conf_s, font=F_TINY, fill=MUTED)
    canvas.save(dest, optimize=True)


def combined_overview(left: Path, right: Path, dest: Path):
    with Image.open(left) as a, Image.open(right) as b:
        a, b = a.convert("RGB"), b.convert("RGB")
        # Both pages use 2 columns, so are moderately sized when joined.
        gap = 18
        out = Image.new("RGB", (a.width + b.width + gap, max(a.height, b.height)), BG)
        out.paste(a, (0, 0))
        out.paste(b, (a.width + gap, 0))
        out.save(dest, optimize=True)


def depth_tracks(df: pd.DataFrame, dest: Path):
    """Depth-ordered true/predicted class tracks, one panel per drill hole.

    Only evaluated tile intervals are shown. This is a tile-level picture,
    not an inferred continuous borehole log. Tile overlaps are allowed.
    """
    holes = sorted(df["hole_id"].astype(str).unique())
    classes = sorted(set(df["label"]) | set(df["predicted"]))
    palette = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#EECA3B"]
    colors = {c: palette[i % len(palette)] for i, c in enumerate(classes)}
    panel_w, top, margin = 310, 155, 28
    depths = [(float(df.loc[df.hole_id == h, "depth_from_m"].min()),
               float(df.loc[df.hole_id == h, "depth_to_m"].max())) for h in holes]
    max_span = max((hi - lo for lo, hi in depths), default=1)
    chart_h = max(600, min(1700, int(max_span * 7)))
    W = margin * 2 + len(holes) * panel_w
    H = top + chart_h + 85
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((margin, 18), "Where predictions match or fail along depth", font=F_TITLE, fill=INK)
    d.text((margin, 65), "Each colored bar represents an evaluated tile; blank spaces are not evaluated.",
           font=F_SUB, fill=MUTED)
    legend_x = margin
    for c in classes:
        d.rectangle([legend_x, 105, legend_x + 13, 118], fill=colors[c])
        d.text((legend_x + 19, 101), short(c), font=F_TINY, fill=INK)
        legend_x += max(76, len(short(c)) * 9 + 38)

    for idx, h in enumerate(holes):
        sub = df[df.hole_id == h].sort_values("depth_from_m")
        lo, hi = depths[idx]
        base_x = margin + idx * panel_w
        d.text((base_x, top - 22), h, font=F_LABEL, fill=INK)
        x_true, x_pred, x_ok = base_x + 59, base_x + 134, base_x + 209
        for x, text in [(x_true, "Journal"), (x_pred, "Model"), (x_ok, "Match")]:
            d.text((x - 6, top + 5), text, font=F_TINY, fill=MUTED)
            d.rectangle([x, top + 29, x + 27, top + 29 + chart_h], fill=WHITE, outline=BORDER)
        for step in range(int(lo // 20) * 20, int(hi) + 21, 20):
            if not lo <= step <= hi:
                continue
            y = top + 29 + round((step - lo) / (hi - lo) * chart_h)
            d.line([base_x + 41, y, base_x + 250, y], fill="#D8DEE8", width=1)
            d.text((base_x, y - 8), str(step), font=F_TINY, fill=MUTED)
        for _, row in sub.iterrows():
            y0 = top + 29 + round((float(row.depth_from_m) - lo) / (hi - lo) * chart_h)
            y1 = top + 29 + round((float(row.depth_to_m) - lo) / (hi - lo) * chart_h)
            y1 = max(y0 + 1, y1)
            for x, color in [(x_true, colors[row.label]),
                             (x_pred, colors[row.predicted]),
                             (x_ok, GREEN if row.correct else RED)]:
                d.rectangle([x, y0, x + 27, y1], fill=color)
        d.text((base_x + 40, top + chart_h + 49),
               f"{len(sub)} tiles | {sub.correct.mean():.1%} correct",
               font=F_TINY, fill=MUTED)
    im.save(dest, optimize=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path, help="A completed corelog run folder")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--output-root", type=Path, help="Override output_root if run folder was moved")
    p.add_argument("--out", type=Path, help="Output gallery directory (default: run/visual_review_<split>)")
    p.add_argument("--n", type=int, default=24, help="Maximum examples per individual gallery")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run = args.run.expanduser().resolve()
    root = (args.output_root.expanduser().resolve() if args.output_root else run.parent.parent)
    pred_path = run / f"eval_{args.split}" / "predictions.csv"
    tile_csv = root / "tiles" / "tiles.csv"
    for f in [pred_path, tile_csv]:
        if not f.is_file():
            p.error(f"Required file missing: {f}")
    out = args.out.expanduser().resolve() if args.out else run / f"visual_review_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(pred_path, dtype={"slice_id": str, "hole_id": str})
    tiles = pd.read_csv(tile_csv, dtype={"slice_id": str, "hole_id": str})
    required_pred = {"slice_id", "label", "predicted", "hole_id", "depth_from_m", "depth_to_m"}
    required_tiles = {"slice_id", "tile_path"}
    if missing := required_pred - set(pred.columns):
        p.error(f"Predictions missing columns: {sorted(missing)}")
    if missing := required_tiles - set(tiles.columns):
        p.error(f"Tile manifest missing columns: {sorted(missing)}")
    if pred.slice_id.duplicated().any() or tiles.slice_id.duplicated().any():
        p.error("Duplicate slice_id detected; refusing an unsafe join")
    df = pred.merge(tiles[["slice_id", "tile_path"]], on="slice_id", how="left", validate="one_to_one")
    if df.tile_path.isna().any():
        examples = df.loc[df.tile_path.isna(), "slice_id"].head(5).tolist()
        p.error(f"Could not find {df.tile_path.isna().sum()} predictions in tiles.csv: {examples}")
    df["full_path"] = df.tile_path.map(lambda s: str(root / Path(s)))
    missing_files = [x for x in df.full_path if not Path(x).is_file()]
    if missing_files:
        p.error(f"{len(missing_files)} tile image(s) are missing. First: {missing_files[0]}")
    df["correct"] = df.label == df.predicted
    df["confidence"] = [float(row[f"p_{row.predicted}"]) if
                        f"p_{row.predicted}" in df.columns and pd.notna(row[f"p_{row.predicted}"])
                        else float("nan") for _, row in df.iterrows()]
    print(f"Matched {len(df)} predictions to tile images in {root}")
    print(f"Correct: {int(df.correct.sum())}; wrong: {int((~df.correct).sum())}")

    def sheet(subset, filename, title, explanation, n=None, columns=4):
        subset = subset.copy()
        count = len(subset)
        subset.attrs["count"] = count
        path = out / filename
        make_sheet(subset, title, explanation, path,
                   max_items=args.n if n is None else n, columns=columns, seed=args.seed)
        print(f"  {filename}: {count} available")
        return path

    # Individual galleries sampled across hole/depth blocks.
    sheet(df[df.correct], "01_correct_examples.png", "Correct predictions", "Green = model matches the journal label")
    sheet(df[~df.correct], "02_wrong_examples.png", "Incorrect predictions", "Red = model disagrees with the journal label")
    sheet(df[(df.label == "lith_TDPAND") & (df.predicted == "lith_TPAND")],
          "03_true_TDPAND_pred_TPAND.png", "TDPAND mistaken for TPAND", "Journal says TDPAND; model says TPAND")
    sheet(df[(df.label == "lith_TPAND") & (df.predicted == "lith_TDPAND")],
          "04_true_TPAND_pred_TDPAND.png", "TPAND mistaken for TDPAND", "Journal says TPAND; model says TDPAND")
    for code in ["lith_TDPAND", "lith_TPAND", "lith_TRCRHYP", "lith_RS", "lith_CLY"]:
        sheet(df[(df.label == code) & df.correct], f"05_correct_{short(code)}.png",
              f"Correct: {short(code)}", "Only evaluated tiles with matching journal and model labels")
    sheet(df[~df.correct].sort_values("confidence", ascending=False).head(300),
          "06_high_confidence_errors.png", "High-confidence errors",
          "Potentially informative disagreement; confidence is not geological certainty")
    # A side-by-side summary (6 examples in each half).
    a = sheet(df[df.correct], "_overview_correct.png", "Correct", "Matching journal and model labels", n=6, columns=2)
    b = sheet(df[~df.correct], "_overview_wrong.png", "Incorrect", "Disagreement with journal label", n=6, columns=2)
    combined_overview(a, b, out / "00_correct_vs_wrong.png")
    a.unlink()
    b.unlink()
    depth_tracks(df, out / "07_predictions_by_depth.png")
    df.drop(columns=["full_path"]).to_csv(out / "all_evaluated_tiles_with_paths_PRIVATE.csv", index=False)
    (out / "README_PRIVATE.txt").write_text(
        f"Run: {run}\nSplit: {args.split}\nTiles root: {root}\n"
        "Correct means agreement with journal, not proof the rock type is geologically correct.\n"
        "Confidence is the model softmax probability of its predicted class.\n"
        "These images contain rock-photo tiles and must not be uploaded to public GitHub.\n")
    print(f"\nDONE. Open first: {out / '00_correct_vs_wrong.png'}")
    print(f"Then review: {out / '03_true_TDPAND_pred_TPAND.png'}")
    print(f"Depth overview: {out / '07_predictions_by_depth.png'}")


if __name__ == "__main__":
    main()
