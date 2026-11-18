"""SOM clustering and cluster labelling by geologists (paper §3.2.2), corrected.

Input: 48 colour-histogram PCs + Gabor features, standardised. ONLY descriptor columns
are used (no row index; audit LB-6) and everything is keyed by slice_id.
Initialisation: MiniSom pca_weights_init (reproducible; audit LB-9). Training samples
are drawn in random order with a fixed seed (the authors trained in depth order; LB-10).

Workflow:
  1. train()          -> som.joblib, assignments.csv, node_summary.csv (+ node_vs_log.csv
                         if log labels exist: how well SOM clusters match the journals)
  2. export_sheets()  -> one contact sheet per node + overview + labelling template
  3. geologists fill  labeling/<name>.csv (copies of the template)
  4. apply_labels()   -> majority vote per node -> labels/labels_som.csv
"""
from __future__ import annotations

import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from minisom import MiniSom
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..config import classes as cfg_classes, output_root
from ..features.descriptors import load as load_features
from ..labels import LOG_LABELS_CSV, SOM_LABELS_CSV

SOM_DIR = "som"


class SomInput:
    """Fitted transform: descriptors -> standardised SOM input matrix."""

    def __init__(self, n_color_pcs: int, seed: int = 0):
        self.n_color_pcs, self.seed = n_color_pcs, seed

    def fit(self, gabor: np.ndarray, color: np.ndarray) -> "SomInput":
        self.color_scaler = StandardScaler().fit(color)
        n = min(self.n_color_pcs, *color.shape)
        # exact solver: deterministic and independent of row order (randomized SVD is not)
        self.pca = PCA(n_components=n, svd_solver="covariance_eigh").fit(self.color_scaler.transform(color))
        self.final = StandardScaler().fit(self._raw(gabor, color))
        return self

    def _raw(self, gabor, color):
        return np.hstack([gabor, self.pca.transform(self.color_scaler.transform(color))])

    def transform(self, gabor, color) -> np.ndarray:
        return self.final.transform(self._raw(gabor, color))

    @property
    def n_features(self) -> int:
        return self.final.n_features_in_


def train(cfg: dict) -> dict:
    s, seed = cfg["som"], cfg["project"]["seed"]
    F = load_features(cfg)
    tr = SomInput(cfg["features"]["color"]["pca_components"], seed).fit(F["gabor"], F["color"])
    X = tr.transform(F["gabor"], F["color"])
    gx, gy = s["grid"]
    som = MiniSom(gx, gy, X.shape[1], sigma=s["sigma"], learning_rate=s["learning_rate"], random_seed=seed)
    if s.get("init", "pca") == "pca":
        som.pca_weights_init(X)
    else:
        som.random_weights_init(X)
    som.train(X, int(s["iterations"]), random_order=bool(s.get("random_order", True)), verbose=False)
    out = output_root(cfg) / SOM_DIR
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"som": som, "input": tr, "grid": (gx, gy)}, out / "som.joblib")
    a = assign(som, X, F["slice_id"], (gx, gy))
    a.to_csv(out / "assignments.csv", index=False)
    summ = a.groupby(["node_row", "node_col"]).size().rename("n_tiles").reset_index()
    summ.to_csv(out / "node_summary.csv", index=False)
    res = {"n_tiles": len(a), "n_features": int(X.shape[1]), "nodes_used": int(len(summ)),
           "quantization_error": float(som.quantization_error(X)),
           "topographic_error": float(som.topographic_error(X))}
    log_p = output_root(cfg) / LOG_LABELS_CSV
    if log_p.exists():
        res["vs_log"] = node_vs_log(a, pd.read_csv(log_p), cfg["labels"].get("unlabeled", "unlabeled"), out)
    return res


def assign(som: MiniSom, X: np.ndarray, slice_ids, grid) -> pd.DataFrame:
    W = som.get_weights().reshape(-1, X.shape[1])
    d = ((X[:, None, :] - W[None]) ** 2).sum(-1)
    k = d.argmin(1)
    r, c = np.unravel_index(k, grid)
    return pd.DataFrame({"slice_id": slice_ids, "node_row": r, "node_col": c,
                         "distance": np.sqrt(d[np.arange(len(k)), k]).round(4)})


def node_vs_log(a: pd.DataFrame, log_labels: pd.DataFrame, unlabeled: str, out: Path) -> dict:
    """Agreement of SOM nodes with journal labels (the check the paper could not do)."""
    m = a.merge(log_labels[["slice_id", "label"]], on="slice_id")
    m = m[m["label"] != unlabeled]
    if m.empty:
        return {}
    ct = pd.crosstab([m["node_row"], m["node_col"]], m["label"])
    ct["majority"] = ct.idxmax(axis=1)
    ct["purity"] = ct.drop(columns="majority").max(axis=1) / ct.drop(columns="majority").sum(axis=1)
    ct.to_csv(out / "node_vs_log.csv")
    n = ct.drop(columns=["majority", "purity"]).sum(axis=1)
    return {"weighted_node_purity": float((ct["purity"] * n).sum() / n.sum()), "n_log_labeled": int(n.sum())}


def export_sheets(cfg: dict, tiles: pd.DataFrame) -> Path:
    """Contact sheet per node + overview + empty labelling template for geologists."""
    out = output_root(cfg) / SOM_DIR / "sheets"
    out.mkdir(parents=True, exist_ok=True)
    a = pd.read_csv(output_root(cfg) / SOM_DIR / "assignments.csv").merge(tiles[["slice_id", "tile_path"]], on="slice_id")
    k, th = cfg["som"].get("thumbnails_per_node", 12), cfg["tiling"]["tile_height"]
    tw = cfg["tiling"]["tile_width"]
    rnd = random.Random(cfg["project"]["seed"])
    gx, gy = cfg["som"]["grid"]
    overview = Image.new("RGB", (gy * (tw // 2 + 4), gx * (th // 2 * 3 + 20)), "white")
    draw_o = ImageDraw.Draw(overview)
    rows = []
    for (r, c), g in a.groupby(["node_row", "node_col"]):
        pick = rnd.sample(list(g["tile_path"]), min(k, len(g)))
        sheet = Image.new("RGB", (tw, (th + 4) * len(pick) + 24), "white")
        ImageDraw.Draw(sheet).text((4, 4), f"node ({r},{c})  n={len(g)}", fill="black")
        for i, p in enumerate(pick):
            sheet.paste(Image.open(output_root(cfg) / p).convert("RGB"), (0, 24 + i * (th + 4)))
        sheet.save(out / f"node_r{r:02d}_c{c:02d}.png")
        x0, y0 = c * (tw // 2 + 4), r * (th // 2 * 3 + 20)
        draw_o.text((x0 + 2, y0 + 2), f"({r},{c}) n={len(g)}", fill="black")
        for i, p in enumerate(pick[:3]):
            overview.paste(Image.open(output_root(cfg) / p).convert("RGB").resize((tw // 2, th // 2)),
                           (x0, y0 + 14 + i * (th // 2 + 2)))
        rows.append({"node_row": r, "node_col": c, "n_tiles": len(g), "label": ""})
    overview.save(out / "overview.png")
    tmpl = output_root(cfg) / SOM_DIR / "labeling" / "TEMPLATE_copy_me.csv"
    tmpl.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(tmpl, index=False)
    (tmpl.parent / "README.txt").write_text(
        "Copy TEMPLATE_copy_me.csv to <your_name>.csv in this folder, look at ../sheets/node_*.png,\n"
        f"and write one label per node using exactly these class names: {cfg_classes(cfg)}.\n"
        "Leave the label empty (or write 'unlabeled') for ambiguous or mixed nodes.\n")
    return out


def vote(labeler_tables: list[pd.DataFrame], classes: list[str], unlabeled: str = "unlabeled") -> pd.DataFrame:
    """Majority vote per node; ties (or no votes) -> unlabeled."""
    allv = []
    for t in labeler_tables:
        t = t[["node_row", "node_col", "label"]].copy()
        t["label"] = t["label"].fillna("").astype(str).str.strip().replace("", unlabeled)
        bad = set(t["label"]) - set(classes) - {unlabeled}
        if bad:
            raise ValueError(f"unknown labels in a labeller file: {sorted(bad)} (allowed: {classes})")
        allv.append(t)
    v = pd.concat(allv)
    out = []
    for (r, c), g in v.groupby(["node_row", "node_col"]):
        counts = g["label"].value_counts()
        top = counts[counts == counts.max()]
        lab = top.index[0] if len(top) == 1 else unlabeled
        out.append({"node_row": r, "node_col": c, "label": lab, "n_labelers": len(g),
                    "agreement": round(float(counts.max() / len(g)), 3)})
    return pd.DataFrame(out)


def apply_labels(cfg: dict, labeler_files: list[Path]) -> pd.DataFrame:
    unl = cfg["labels"].get("unlabeled", "unlabeled")
    if not labeler_files:
        raise ValueError("no labeller CSV files given")
    nodes = vote([pd.read_csv(f) for f in labeler_files], cfg_classes(cfg), unl)
    nodes.to_csv(output_root(cfg) / SOM_DIR / "node_labels.csv", index=False)
    a = pd.read_csv(output_root(cfg) / SOM_DIR / "assignments.csv")
    lab = a.merge(nodes, on=["node_row", "node_col"], how="left")
    lab["label"] = lab["label"].fillna(unl)
    lab["source"] = "som"
    p = output_root(cfg) / SOM_LABELS_CSV
    p.parent.mkdir(parents=True, exist_ok=True)
    lab[["slice_id", "label", "source", "node_row", "node_col", "agreement"]].to_csv(p, index=False)
    return lab
