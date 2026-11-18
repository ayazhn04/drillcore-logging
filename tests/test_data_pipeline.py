"""Config, manifest validation, tiling, log labels and hole splits."""
import numpy as np
import pandas as pd
import pytest

from corelog import config as C
from corelog import manifest, splits
from corelog.labels import log_labels
from corelog.tiling import tile_row


# ---------------------------------------------------------------- config
def test_config_layers_overrides_and_env(tmp_path, monkeypatch):
    extra = tmp_path / "x.yaml"
    extra.write_text("train:\n  lr: 0.5\nlabels:\n  classes: [a, b]\n")
    cfg = C.load_config([extra], ["train.epochs=3", "split.train_holes=[H1, H2]", "paths.data_root=rel/data"])
    assert cfg["train"]["lr"] == 0.5 and cfg["train"]["epochs"] == 3
    assert cfg["split"]["train_holes"] == ["H1", "H2"]
    assert cfg["model"]["name"] == "baseline"            # inherited from base.yaml
    monkeypatch.setenv("CORELOG_DATA_ROOT", str(tmp_path))
    cfg = C.load_config([extra])
    assert C.data_path(cfg, "rows_manifest") == tmp_path / "manifests" / "rows.csv"


def test_config_requires_data_root_and_unique_classes(monkeypatch):
    monkeypatch.delenv("CORELOG_DATA_ROOT", raising=False)
    cfg = C.load_config()
    with pytest.raises(C.ConfigError):
        C.data_root(cfg)
    cfg["labels"]["classes"] = ["a", "a"]
    with pytest.raises(C.ConfigError):
        C.classes(cfg)


def test_committed_configs_have_no_absolute_paths():
    import yaml
    for f in (C.REPO_ROOT / "configs").glob("*.yaml"):
        if f.name.endswith("example.yaml"):
            continue
        paths = (yaml.safe_load(f.read_text()) or {}).get("paths", {}) or {}
        for k, v in paths.items():
            assert v is None or not str(v).startswith(("/", "~")) and ":" not in str(v), (f.name, k, v)


# ---------------------------------------------------------------- manifest
def test_synthetic_manifest_validates(synth_root):
    rows = manifest.read_rows(synth_root / "manifests/rows.csv")
    logs = manifest.read_logs(synth_root / "manifests/log_intervals.csv")
    rep = manifest.validate(rows, logs, synth_root, {"S1": 1, "S2": 1, "S3": 1, "NC": 1})
    assert rep.ok, rep.errors


def test_manifest_detects_errors(synth_root):
    rows = manifest.read_rows(synth_root / "manifests/rows.csv")
    logs = manifest.read_logs(synth_root / "manifests/log_intervals.csv")
    bad = rows.copy()
    bad.loc[0, "image_path"] = "photos/missing.png"
    bad.loc[1, "x1"] = 10 ** 6                                 # box outside image
    bad.loc[2, "depth_to_m"] = bad.loc[2, "depth_from_m"] - 1  # inverted depths
    bad.loc[5, "depth_from_m"] = bad.loc[4, "depth_from_m"] + 0.1  # overlap (same hole)
    blogs = logs.copy()
    blogs.loc[0, "lithology_code"] = "ZZZ"
    rep = manifest.validate(bad, blogs, synth_root, {"S1": 1, "S2": 1, "S3": 1, "NC": 1})
    text = " ".join(rep.errors)
    assert not rep.ok
    for needle in ("not found", "outside image", "depth_to_m <= depth_from_m", "overlap"):
        assert needle in text, needle
    assert any("ZZZ" in w for w in rep.warnings)
    rep = manifest.validate(rows.drop(columns=["depth_to_m"]), logs, synth_root)
    assert any("missing required" in e for e in rep.errors)


# ---------------------------------------------------------------- tiling
def test_tile_row_depths_contiguous_and_direction():
    img = np.zeros((120, 1300, 3), np.uint8)
    img[:, :650] = 255                                       # left half white
    pieces = tile_row(img, 10.0, 11.3, tile_h=60, tile_w=300, stride=300)
    assert [p[0].shape for p in pieces] == [(60, 300, 3)] * 2   # 650 px wide after resize -> 2 tiles (+50 px dropped)
    assert pieces[0][1] == pytest.approx(10.0) and pieces[1][1] == pytest.approx(pieces[0][2])
    rtl = tile_row(img, 10.0, 11.3, 60, 300, 300, direction="rtl")
    assert rtl[0][0].mean() < 10 and pieces[0][0].mean() > 245   # first tile is the shallow end
    part = tile_row(np.zeros((60, 500, 3), np.uint8), 0, 1, 60, 300, 300, min_last_fraction=0.5)
    assert len(part) == 2 and part[1][2] == pytest.approx(1.0)  # last window shifted back to row end


def test_make_tiles_ids_unique_and_ordered(tiled):
    cfg, tiles, _ = tiled
    assert tiles["slice_id"].is_unique
    assert (C.output_root(cfg) / tiles["tile_path"].iloc[0]).exists()
    for _, g in tiles.groupby("hole_id"):
        assert g["depth_from_m"].is_monotonic_increasing


# ---------------------------------------------------------------- labels
def test_log_labels_purity_and_unmapped():
    tiles = pd.DataFrame({"slice_id": ["a", "b", "c", "d"], "hole_id": "H",
                          "depth_from_m": [0.0, 0.9, 2.0, 3.0], "depth_to_m": [0.3, 1.2, 2.3, 3.3]})
    logs = pd.DataFrame({"hole_id": "H", "depth_from_m": [0.0, 1.0, 2.0],
                         "depth_to_m": [1.0, 2.0, 3.0], "lithology_code": ["G", "S", "UNKNOWN"]})
    lab = log_labels(tiles, logs, {"G": "granite", "S": "schist"}, ["granite", "schist"], 0.8)
    assert lab["label"].tolist() == ["granite", "unlabeled", "unlabeled", "unlabeled"]
    assert lab.loc[1, "purity"] == pytest.approx(2 / 3, abs=1e-3)  # contact tile
    with pytest.raises(ValueError):
        log_labels(tiles, logs, {"G": "gneiss"}, ["granite"], 0.8)


def test_synthetic_log_labels_cover_all_classes(tiled):
    cfg, _, labels = tiled
    assert set(C.classes(cfg)) <= set(labels["label"])


# ---------------------------------------------------------------- splits
def test_hole_splits_disjoint_and_errors():
    tiles = pd.DataFrame({"hole_id": list("AABBCCDD")})
    s = splits.assign(tiles, {"mode": "holes", "train_holes": ["A", "B"], "val_holes": ["C"], "test_holes": ["D"]})
    assert s.tolist() == ["train"] * 4 + ["val"] * 2 + ["test"] * 2
    with pytest.raises(splits.SplitError):
        splits.hole_split(["A", "B"], {"mode": "holes", "train_holes": ["A"], "test_holes": ["A"]})
    folds = [splits.hole_split(list("ABCD"), {"mode": "leave_one_hole_out", "fold": k}) for k in range(4)]
    assert sorted(h for f in folds for h, p in f.items() if p == "test") == list("ABCD")
