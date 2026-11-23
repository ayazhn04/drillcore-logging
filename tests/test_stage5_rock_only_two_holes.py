"""Stage 5: rock-only pre-cropped deliveries (black background + masks) and two-hole splits.

Synthetic data only (corelog.synthetic.make_rock_only_dataset). Regression tests for:
BOM-encoded CSVs, mask resolution and validation, per-row pixel-box completeness, empty
journal codes, class coverage per hole, row scale (px/m) consistency, mask-aware tiling
and rock-fraction filtering, contact buffer, mask-aware descriptors, within-hole
validation blocks with a buffer, depth-block bootstrap, and the end-to-end two-hole run.
"""
import json

import numpy as np
import pandas as pd
import pytest

from corelog import manifest, splits
from corelog.config import REPO_ROOT, load_config, output_root
from corelog.features.descriptors import color_histogram, gabor_features
from corelog.labels import log_labels
from corelog.synthetic import _texture, make_rock_only_dataset
from corelog.tiling import tile_row_masked

CFG = REPO_ROOT / "configs" / "rock_only_synthetic.yaml"


def rcfg(data, out, *over):
    return load_config([CFG], [f"paths.data_root={data}", f"paths.output_root={out}", *over])


@pytest.fixture(scope="module")
def rock_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("rock_only")
    make_rock_only_dataset(root)
    return root


def _read(root):
    return (manifest.read_rows(root / "manifests/rows.csv"),
            manifest.read_logs(root / "manifests/log_intervals.csv"))


# ---------------------------------------------------------------- manifest / validator
def test_bom_csv_headers_are_clean(rock_root):
    raw = (rock_root / "manifests/rows.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")                      # the fixture really has a BOM
    rows, logs = _read(rock_root)
    assert rows.columns[0] == "hole_id" and logs.columns[0] == "hole_id"
    assert logs["description"].iloc[0] == "синтетика"            # UTF-8 Cyrillic preserved


def test_mask_path_resolution():
    r = pd.Series({"hole_id": "H", "photo_id": "H_Box01", "row_id": "2",
                   "image_path": "photos/H/H_Box01_row2.png", "mask_path": np.nan})
    t = "masks/{image_subdir}/{image_stem}_mask.png"
    assert manifest.mask_path_for(r, t) == "masks/H/H_Box01_row2_mask.png"
    assert manifest.mask_path_for(r, None) is None
    assert manifest.mask_path_for(r.copy().replace({np.nan: "m/x.png"}), t) == "m/x.png"  # column wins


def test_rock_only_delivery_validates(rock_root):
    rows, logs = _read(rock_root)
    cfg = rcfg(rock_root, "unused")
    rep = manifest.validate(rows, logs, rock_root, cfg["labels"]["class_map"],
                            mask_template=cfg["tiling"]["mask_template"], classes=cfg["labels"]["classes"])
    assert rep.ok, rep.errors
    info = " ".join(rep.info)
    assert "no pixel boxes" in info and "rock masks found" in info and "px/m" in info
    assert "photographed metres per class and hole" in info


def test_validator_catches_rock_only_problems(rock_root, tmp_path):
    rows, logs = _read(rock_root)
    cfg = rcfg(rock_root, "unused")
    bad = rows.copy()
    bad["x0"] = bad["x0"].astype(object)
    bad.loc[0, "x0"] = 5                                        # only one of four box fields
    bad.loc[1, "mask_path"] = "masks/nowhere.png"               # missing mask
    from PIL import Image
    Image.new("L", (7, 7)).save(tmp_path / "small_mask.png")
    (rock_root / "masks" / "wrong").mkdir(exist_ok=True)
    Image.new("L", (7, 7)).save(rock_root / "masks" / "wrong" / "m.png")
    bad.loc[2, "mask_path"] = "masks/wrong/m.png"               # mask of the wrong size
    blogs = logs.copy()
    blogs.loc[0, "lithology_code"] = ""                         # empty journal code
    blogs = blogs[~((blogs["hole_id"] == "SYN_B") & (blogs["lithology_code"] == "S3"))]  # class absent from a hole
    rep = manifest.validate(bad, blogs, rock_root, cfg["labels"]["class_map"],
                            mask_template=cfg["tiling"]["mask_template"], classes=cfg["labels"]["classes"])
    e, w = " ".join(rep.errors), " ".join(rep.warnings)
    assert "only some of x0, y0, x1, y1" in e
    assert "mask not found" in e and "differs from image size" in e
    assert "empty code" in w
    assert "'syn_class_3' is absent from hole(s) ['SYN_B']" in w


def test_validator_flags_inconsistent_row_scale(rock_root):
    """The synthetic last box is partly filled but keeps a full box interval: its rows get a
    much lower px/m than the rest -> the validator must point at them."""
    rows, logs = _read(rock_root)
    rep = manifest.validate(rows, logs, rock_root, mask_template="masks/{image_subdir}/{image_stem}_mask.png")
    assert any("px/m" in w and "outside" in w for w in rep.warnings)


# ---------------------------------------------------------------- tiling
def test_mask_follows_flip_trim_and_resize():
    img = np.zeros((40, 400, 3), np.uint8)
    mask = np.zeros((40, 400), bool)
    mask[:, :100] = True                                        # rock only at the shallow (left) end
    img[mask] = 200
    ltr = tile_row_masked(img, mask, 0, 1, 20, 50, 50)
    rtl = tile_row_masked(img[:, ::-1], mask[:, ::-1], 0, 1, 20, 50, 50, direction="rtl")
    assert [round(t[1].mean(), 2) for t in ltr] == [round(t[1].mean(), 2) for t in rtl]
    assert ltr[0][1].mean() == 1.0 and ltr[-1][1].mean() == 0.0
    trim = tile_row_masked(img, mask, 0, 1, 20, 50, 50, trim_fraction=0.25)  # 100 px off each end = all the rock
    assert trim[0][1].mean() == 0.0
    with pytest.raises(ValueError):
        tile_row_masked(img, mask[:, :10], 0, 1, 20, 50, 50)


def test_make_tiles_records_rock_fraction_and_drops_low_rock(rock_root, tmp_path):
    from corelog.tiling import DROPPED_CSV, make_tiles
    cfg = rcfg(rock_root, tmp_path, "tiling.min_rock_fraction=0.62")
    tiles = make_tiles(cfg)
    assert tiles["rock_fraction"].between(0.62, 1).all()
    dropped = pd.read_csv(output_root(cfg) / DROPPED_CSV)
    assert len(dropped) > 0 and (dropped["rock_fraction"] < 0.62).all()
    assert (output_root(cfg) / tiles["mask_path"].iloc[0]).exists()
    assert not set(dropped["slice_id"]) & set(tiles["slice_id"])


# ---------------------------------------------------------------- labels
def test_contact_buffer_unlabels_tiles_near_contacts():
    tiles = pd.DataFrame({"slice_id": list("abc"), "hole_id": "H",
                          "depth_from_m": [0.70, 0.40, 1.60], "depth_to_m": [0.95, 0.65, 1.85]})
    logs = pd.DataFrame({"hole_id": "H", "depth_from_m": [0.0, 1.0], "depth_to_m": [1.0, 2.0],
                         "lithology_code": ["G", "S"]})
    args = (logs, {"G": "g", "S": "s"}, ["g", "s"], 0.8)
    assert log_labels(tiles, *args)["label"].tolist() == ["g", "g", "s"]
    # tile a ends 5 cm above the contact: with a 10 cm buffer it is no longer pure
    assert log_labels(tiles, *args, contact_buffer_m=0.10)["label"].tolist() == ["unlabeled", "g", "s"]


# ---------------------------------------------------------------- descriptors
def test_masked_descriptors_ignore_black_background():
    rng = np.random.default_rng(0)
    tex = np.clip(_texture("S2", 60, 60, rng), 0, 255).astype(np.uint8)
    m1 = np.zeros((60, 60), bool); m1[15:45, 5:55] = True        # two different piece outlines,
    m2 = np.zeros((60, 60), bool); m2[10:50, 20:45] = True        # same rock texture
    a, b = tex.copy(), tex.copy()
    a[~m1] = 0; b[~m2] = 0
    h = color_histogram(a, 8, "none", m1)
    assert h[0] == 0                                              # no black bin from the background
    assert h[0] < color_histogram(a, 8, "none")[0]                # the unmasked one has it
    freqs, th = [0.09, 0.17], [0, 45, 90, 135]
    ga, gb = gabor_features(a, freqs, th, mask=m1), gabor_features(b, freqs, th, mask=m2)
    ua, ub = gabor_features(a, freqs, th), gabor_features(b, freqs, th)
    assert np.abs(ga - gb).sum() < 0.5 * np.abs(ua - ub).sum()   # outline matters far less with masks


# ---------------------------------------------------------------- splits
def _two_hole_tiles():
    d0 = np.arange(0, 20, 0.25)
    return pd.DataFrame({"hole_id": ["A"] * len(d0) + ["B"] * len(d0),
                         "depth_from_m": np.r_[d0, d0], "depth_to_m": np.r_[d0, d0] + 0.25})


def test_val_block_inside_train_hole_with_buffer():
    t = _two_hole_tiles()
    cfg = {"mode": "holes", "train_holes": ["A"], "test_holes": ["B"],
           "val_from_train": {"fraction": 0.2, "buffer_m": 1.0, "position": 1.0}}
    s = splits.assign(t, cfg)
    assert set(s[t["hole_id"] == "B"]) == {"test"}                # test hole whole and untouched
    a = t[t["hole_id"] == "A"].assign(s=s)
    val, tr = a[a["s"] == "val"], a[a["s"] == "train"]
    assert val["depth_from_m"].min() >= 16.0 - 1e-9 and val["depth_to_m"].max() <= 20.0
    assert val["depth_from_m"].min() - tr["depth_to_m"].max() >= 1.0 - 1e-9   # buffer respected
    assert (a["s"] == "none").sum() == 4                          # 1 m of 0.25 m tiles dropped
    s0 = splits.assign(t, {**cfg, "val_from_train": {"fraction": 0.2, "buffer_m": 0.5, "position": 0.0}})
    assert t.loc[s0 == "val", "depth_to_m"].max() <= 4.0          # position 0 = shallowest block


def test_two_hole_split_errors_and_loho():
    t = _two_hole_tiles()
    vb = {"fraction": 0.2, "buffer_m": 1.0}
    with pytest.raises(splits.SplitError):                        # both val mechanisms at once
        splits.assign(t, {"mode": "holes", "train_holes": ["A"], "val_holes": ["B"], "val_from_train": vb})
    with pytest.raises(splits.SplitError):                        # classic LOHO needs 3 holes
        splits.hole_split(["A", "B"], {"mode": "leave_one_hole_out"})
    folds = [splits.assign(t, {"mode": "leave_one_hole_out", "fold": k, "val_from_train": vb}) for k in (0, 1)]
    assert [sorted(set(t.loc[f == "test", "hole_id"])) for f in folds] == [["A"], ["B"]]
    assert all((f == "val").any() for f in folds)
    s = pd.Series(["train"] * 80 + ["test"] * 80)
    s.iloc[40] = "val"                                            # val tile surrounded by train tiles
    with pytest.raises(splits.SplitError):
        splits.check_disjoint(t, s, within_hole_buffer_m=1.0)


# ---------------------------------------------------------------- evaluation
def test_block_bootstrap_with_one_test_hole():
    from corelog.evaluate import bootstrap
    n = 200
    df = pd.DataFrame({"hole_id": "B", "depth_from_m": np.arange(n) * 0.25,
                       "y": np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int)})
    df["pred"] = df["y"]
    df.loc[:9, "pred"] = 1
    b = bootstrap(df, ["a", "b"], 100, 0, block_length_m=2.0)
    assert b["unit"].startswith("depth block") and b["n_blocks"] == 25
    assert bootstrap(df, ["a", "b"], 10, 0)["unit"].startswith("tile")


# ---------------------------------------------------------------- end to end (CPU, seconds)
def test_two_hole_rock_only_end_to_end(rock_root, tmp_path):
    from corelog.diagnostics import inspect
    from corelog.evaluate import evaluate_run
    from corelog.features.descriptors import extract
    from corelog.inference import predict_log
    from corelog.labels import make_log_labels
    from corelog.som import cluster
    from corelog.tiling import make_tiles
    from corelog.train.supervised import train_supervised
    cfg = rcfg(rock_root, tmp_path)
    tiles = make_tiles(cfg)
    make_log_labels(cfg, tiles)
    ins = inspect(cfg, sheets=6)
    assert (output_root(cfg) / "reports" / "inspection.md").exists()
    assert (output_root(cfg) / "reports" / "tile_sheets" / "syn_class_1.png").exists()
    assert ins["by_split"]["test"]["syn_class_1"] > 0 and ins["by_split"]["train"]["syn_class_1"] > 0
    F = extract(cfg, tiles)
    assert F["masked"].all()
    som = cluster.train(cfg)
    assert som.get("vs_log", {}).get("weighted_node_purity", 0) > 0
    run = train_supervised(cfg, "rock_only_sup")
    ds = pd.read_csv(run / "data_summary.csv").set_index("split")
    assert ds.loc["train", "holes"] == "SYN_A" and ds.loc["test", "holes"] == "SYN_B"
    assert ds.loc["val", "holes"] == "SYN_A" and ds.loc["val", "n"] > 0
    ev = evaluate_run(run, cfg)
    assert ev["data_provenance"] == "SYNTHETIC" and ev["holes"] == ["SYN_B"]
    assert ev["bootstrap"]["unit"].startswith("depth block")
    assert "note" in evaluate_run(run, cfg, "val")
    pl = predict_log(run, cfg, compare=True)
    assert pl["data_provenance"] == "SYNTHETIC" and pl["n_intervals"] > 0
    assert json.loads((run / "provenance.json").read_text())["data_provenance"] == "SYNTHETIC"


def test_two_hole_lab_template_has_placeholders_only():
    import yaml
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "lab.two_holes.example.yaml").read_text())
    holes = cfg["split"]["train_holes"] + cfg["split"]["test_holes"]
    assert set(holes) == {"HOLE_A", "HOLE_B"} and cfg["split"]["val_holes"] == []
    assert cfg["split"]["val_from_train"]["fraction"] > 0
    merged = load_config([REPO_ROOT / "configs" / "lab.two_holes.example.yaml"])
    assert set(merged["labels"]["class_map"].values()) <= set(merged["labels"]["classes"])


# ---------------------------------------------------------------- tile-geometry comparison
def test_no_tiles_error_explains_geometry(rock_root, tmp_path):
    """Paper geometry 60 x 300 on short, full-height rows: no row is 5x longer than high."""
    from corelog.tiling import ROWS_TILING_CSV, make_tiles
    cfg = rcfg(rock_root, tmp_path, "tiling.tile_height=60", "tiling.tile_width=300", "tiling.stride=300")
    with pytest.raises(ValueError) as e:
        make_tiles(cfg)
    msg = str(e.value)
    assert "at least 5.00x longer than high" in msg and "rows are too short" in msg
    assert "tiles_128.yaml" in msg
    from corelog.cli import main                                  # CLI: clean message, exit code 1
    assert main(["make-tiles", "--config", str(CFG), "--set", f"paths.data_root={rock_root}",
                 "--set", f"paths.output_root={tmp_path}", "--set", "tiling.tile_width=300",
                 "--set", "tiling.stride=300"]) == 1
    rs = pd.read_csv(output_root(cfg) / ROWS_TILING_CSV)        # diagnostics written before the error
    assert len(rs) == 28 and (rs["n_windows"] == 0).all()
    from corelog.diagnostics import compare_tiling                # failed geometry still comparable
    table = compare_tiling([output_root(cfg)])
    assert "| tile size (h x w px) | 60 x 300 |" in table
    assert "| rows with zero windows (%) | 100.0 |" in table and "| tiles kept (after rock filter) | 0 |" in table


def test_rows_tiling_stats_are_consistent(rock_root, tmp_path):
    from corelog.tiling import ROWS_TILING_CSV, make_tiles
    cfg = rcfg(rock_root, tmp_path, "tiling.min_rock_fraction=0.6")
    tiles = make_tiles(cfg)
    rs = pd.read_csv(output_root(cfg) / ROWS_TILING_CSV)
    assert len(rs) == len(manifest.read_rows(rock_root / "manifests/rows.csv"))
    assert rs["n_kept"].sum() == len(tiles)
    assert (rs["n_kept"] + rs["n_dropped_low_rock"] == rs["n_windows"]).all()
    assert rs["rock_covered_fraction"].between(0, 1).all()
    assert (rs["aspect"] == (rs["width_px"] / rs["height_px"]).round(3)).all()


def test_compare_two_tile_geometries(rock_root, tmp_path):
    from corelog.diagnostics import SUMMARY_JSON, compare_tiling, inspect
    from corelog.labels import make_log_labels
    from corelog.tiling import make_tiles
    roots = []
    for name, extra in (("t60", []), ("t128", [REPO_ROOT / "configs/experiments/tiles_128.yaml"])):
        out = tmp_path / name
        cfg = load_config([CFG, *extra], [f"paths.data_root={rock_root}", f"paths.output_root={out}"])
        tiles = make_tiles(cfg)
        make_log_labels(cfg, tiles)
        s = inspect(cfg)["tiling_summary"]
        assert (out / SUMMARY_JSON).exists() and s["data_provenance"] == "SYNTHETIC"
        lab = s["labels"]
        parts = lab["mixed_contact_pct"] + lab["contact_buffer_pct"] + lab["unlogged_or_unmapped_pct"]
        assert parts == pytest.approx(lab["ambiguous_pct"], abs=0.05)
        assert s["tiles_generated"] == s["tiles_kept"] + s["tiles_dropped_low_rock"]
        assert 0 < s["depth_covered_by_kept_tiles_pct"] <= 100.0    # overlapping last windows counted once
        roots.append((out, s))
    (o60, s60), (o128, s128) = roots
    assert s60["tile_px"] == [60, 60] and s128["tile_px"] == [128, 128]
    table = compare_tiling([o60, o128])
    for needle in ("| metric | t60 | t128 |", "rows with zero kept tiles (%)", "metres per tile",
                   "ambiguous (unlabeled) tiles (%)", "row rock inside kept tiles (%)", "SYNTHETIC"):
        assert needle in table


def test_tiles_128_experiment_is_optional_and_consistent():
    base = load_config()
    assert base["tiling"]["tile_height"] == 60 and base["tiling"]["tile_width"] == 300   # paper default kept
    alt = load_config([REPO_ROOT / "configs/experiments/tiles_128.yaml"])
    assert [alt["tiling"]["tile_height"], alt["tiling"]["tile_width"]] == alt["model"]["input_size"] == [128, 128]
    assert alt["tiling"]["stride"] == alt["tiling"]["tile_width"]
    assert "paths" not in __import__("yaml").safe_load((REPO_ROOT / "configs/experiments/tiles_128.yaml").read_text())
