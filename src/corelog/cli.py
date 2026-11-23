"""Command-line interface. Every command accepts:
  --config FILE   (repeatable; layered on top of configs/base.yaml)
  --set KEY=VALUE (repeatable; e.g. --set train.epochs=50 --set model.name=vgg16)

Typical lab sequence (see docs/WORKFLOW.md):
  corelog validate        -c configs/lab.yaml
  corelog make-tiles      -c configs/lab.yaml
  corelog make-labels     -c configs/lab.yaml
  corelog inspect         -c configs/lab.yaml     (class x hole x split counts, rock fraction)
  corelog compare-tiling  OUT_60x300 OUT_128      (after inspect in each output root)
  corelog train           -c configs/lab.yaml [--mean-teacher]
  corelog evaluate        -c configs/lab.yaml outputs/runs/<run>
  corelog predict-log     -c configs/lab.yaml outputs/runs/<run> [--holes H1 H2]
SOM labelling (optional, paper workflow):
  corelog extract-features / som-train / som-sheets / som-apply-labels FILES...
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path


def _cfg(a):
    from .config import load_config
    return load_config(a.config, a.set)


def cmd_env(a):
    from .runinfo import environment
    print(json.dumps(environment(), indent=2))


def cmd_validate(a):
    import pandas as pd

    from . import manifest, splits
    from .config import classes, data_path, data_root, output_root
    cfg = _cfg(a)
    rows = manifest.read_rows(data_path(cfg, "rows_manifest"))
    lp = data_path(cfg, "log_intervals")
    logs = manifest.read_logs(lp) if lp.exists() else None
    cls = classes(cfg)
    rep = manifest.validate(rows, logs, data_root(cfg), cfg["labels"].get("class_map", {}),
                            check_images=not a.no_images, mask_template=cfg["tiling"].get("mask_template"),
                            classes=cls)
    bad = set(cfg["labels"].get("class_map", {}).values()) - set(cls)
    if bad:
        rep.errors.append(f"labels.class_map maps to classes not in labels.classes: {sorted(bad)}")
    if logs is not None and not cfg["labels"].get("class_map"):
        rep.errors.append("labels.class_map is empty: map journal lithology codes to model classes")
    try:
        holes = rows["hole_id"].astype(str).unique().tolist()
        m = splits.hole_split(holes, cfg["split"]) if cfg["split"].get("mode") != "random" else {}
        rep.info.append(f"split: {json.dumps({p: sorted(h for h, s in m.items() if s == p) for p in ('train', 'val', 'test', 'none')})}")
        if not any(s == "test" for s in m.values()) and cfg["split"].get("mode") == "holes":
            rep.warnings.append("no test holes configured")
    except Exception as e:  # noqa: BLE001
        rep.errors.append(f"split: {e}")
    out = output_root(cfg) / "reports" / "validation.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    if logs is not None and all(c in logs for c in manifest.LOGS_REQUIRED):
        try:
            r2, l2 = rows.copy(), logs.copy()
            for d in (r2, l2):
                for c in ("depth_from_m", "depth_to_m"):
                    d[c] = pd.to_numeric(d[c], errors="coerce")
            manifest.class_metres(r2, l2, cfg["labels"].get("class_map", {})).to_csv(
                out.parent / "class_metres_by_hole.csv", index=False)
        except Exception as e:  # noqa: BLE001
            rep.warnings.append(f"class metres table not written: {e}")
    out.write_text(rep.to_markdown(), encoding="utf-8")
    print(rep.to_markdown())
    print(f"report written to {out}")
    return 0 if rep.ok else 1


def cmd_inspect(a):
    from .diagnostics import inspect
    r = inspect(_cfg(a), a.min_tiles, a.min_intervals, a.sheets)
    print(open(r["report"], encoding="utf-8").read())
    print(f"report written to {r['report']}")


def cmd_compare_tiling(a):
    from .diagnostics import compare_tiling
    table = compare_tiling(a.output_roots)
    print(table)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(table + "\n", encoding="utf-8")
        print(f"written to {a.out}")


def cmd_tiles(a):
    from .tiling import make_tiles
    try:
        df = make_tiles(_cfg(a))
    except ValueError as e:
        print(f"[make-tiles] FAILED: {e}", file=sys.stderr)
        return 1
    print(f"{len(df)} tiles from {df['hole_id'].nunique()} holes")


def cmd_labels(a):
    from .labels import make_log_labels
    from .tiling import read_tiles
    cfg = _cfg(a)
    lab = make_log_labels(cfg, read_tiles(cfg))
    print(lab["label"].value_counts().to_string())


def cmd_features(a):
    from .features.descriptors import extract
    from .tiling import read_tiles
    cfg = _cfg(a)
    F = extract(cfg, read_tiles(cfg))
    print(f"features: gabor {F['gabor'].shape}, color {F['color'].shape}")


def cmd_som_train(a):
    from .som.cluster import train
    print(json.dumps(train(_cfg(a)), indent=2))


def cmd_som_sheets(a):
    from .som.cluster import export_sheets
    from .tiling import read_tiles
    cfg = _cfg(a)
    print(f"contact sheets in {export_sheets(cfg, read_tiles(cfg))}")


def cmd_som_apply(a):
    from .som.cluster import apply_labels
    lab = apply_labels(_cfg(a), [Path(f) for f in a.files])
    print(lab["label"].value_counts().to_string())


def cmd_train(a):
    cfg = _cfg(a)
    if a.mean_teacher or cfg.get("mean_teacher", {}).get("enabled"):
        from .train.mean_teacher import train_mean_teacher as fn
    else:
        from .train.supervised import train_supervised as fn
    print(f"run saved to {fn(cfg, a.run_name)}")


def cmd_evaluate(a):
    from .evaluate import evaluate_run
    res = evaluate_run(a.run, _cfg(a), a.split)
    print(json.dumps({k: res[k] for k in res if k != "per_class"}, indent=2))


def cmd_predict_log(a):
    from .inference import predict_log
    print(json.dumps(predict_log(a.run, _cfg(a), a.holes, compare=not a.no_compare), indent=2, default=str))


def cmd_smoke(a):
    """Synthetic end-to-end run of every stage on CPU (about a minute)."""
    import pandas as pd

    from .config import REPO_ROOT, load_config
    from .evaluate import evaluate_run
    from .features.descriptors import extract
    from .inference import predict_log
    from .labels import make_log_labels
    from .som import cluster
    from .synthetic import make_dataset
    from .tiling import make_tiles
    from .train.mean_teacher import train_mean_teacher
    from .train.supervised import train_supervised

    base = Path(a.out) if a.out else Path(tempfile.mkdtemp(prefix="corelog_smoke_"))
    if base.exists() and any(base.iterdir()) and a.out:
        shutil.rmtree(base)
    data, out = base / "data", base / "outputs"
    t0 = time.time()
    steps = []

    def step(name, fn):
        s = time.time()
        r = fn()
        steps.append((name, round(time.time() - s, 1)))
        print(f"[smoke] {name:28s} ok ({steps[-1][1]} s)")
        return r

    make_dataset(data)
    cfg = load_config([REPO_ROOT / "configs" / "smoke.yaml"] + list(a.config),
                      [f"paths.data_root={data}", f"paths.output_root={out}"] + list(a.set))
    ns = argparse.Namespace(config=[REPO_ROOT / "configs" / "smoke.yaml"] + list(a.config),
                            set=[f"paths.data_root={data}", f"paths.output_root={out}"] + list(a.set), no_images=False)
    rc = step("validate", lambda: cmd_validate(ns))
    if rc:
        raise SystemExit("[smoke] validation failed")
    tiles = step("make-tiles", lambda: make_tiles(cfg))
    step("make-labels (journal)", lambda: make_log_labels(cfg, tiles))
    step("extract-features", lambda: extract(cfg, tiles))
    som = step("som-train", lambda: cluster.train(cfg))
    step("som-sheets", lambda: cluster.export_sheets(cfg, tiles))
    # stand-in geologist: labels each node with its journal majority (synthetic only)
    nv = pd.read_csv(out / "som" / "node_vs_log.csv").rename(columns={"majority": "label"})
    f = out / "som" / "labeling" / "synthetic_geologist.csv"
    nv[["node_row", "node_col", "label"]].to_csv(f, index=False)
    step("som-apply-labels", lambda: cluster.apply_labels(cfg, [f]))
    sup = step("train baseline (supervised)", lambda: train_supervised(cfg, "smoke_supervised"))
    mt = step("train baseline (mean teacher)", lambda: train_mean_teacher(cfg, "smoke_meanteacher"))
    ev = step("evaluate supervised", lambda: evaluate_run(sup, cfg))
    ev2 = step("evaluate mean teacher", lambda: evaluate_run(mt, cfg))
    pl = step("predict-log + compare", lambda: predict_log(sup, cfg, compare=True))
    summary = {"data_provenance": "SYNTHETIC - not geological results", "output": str(base), "seconds": round(time.time() - t0, 1), "tiles": len(tiles),
               "som_weighted_node_purity_vs_journal": som.get("vs_log", {}).get("weighted_node_purity"),
               "test_accuracy_supervised": ev["overall"]["accuracy"],
               "test_accuracy_mean_teacher": ev2["overall"]["accuracy"],
               "predicted_intervals": pl["n_intervals"],
               "journal_agreement": pl["comparison_with_journal"]}
    print(json.dumps(summary, indent=2, default=str))
    print("[smoke] PASS: the pipeline runs end to end. All numbers above come from SYNTHETIC images\n"
          "        and say nothing about real geological classification performance.")
    if not a.out and not a.keep:
        shutil.rmtree(base, ignore_errors=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="corelog", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        s = sub.add_parser(name, help=help_)
        s.add_argument("-c", "--config", action="append", default=[], help="YAML config (repeatable)")
        s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override (repeatable)")
        s.set_defaults(fn=fn)
        return s

    add("env-report", cmd_env, "print package versions, CUDA and git state")
    add("validate", cmd_validate, "check manifests, images, depths, class map and split").add_argument(
        "--no-images", action="store_true", help="skip opening image files (fast)")
    add("make-tiles", cmd_tiles, "cut rows into depth-registered tiles")
    add("make-labels", cmd_labels, "label tiles from the geological journals")
    s = add("inspect", cmd_inspect, "class x hole x split counts, rock fraction, geometry (after make-labels)")
    s.add_argument("--min-tiles", type=int, default=30)
    s.add_argument("--min-intervals", type=int, default=3)
    s.add_argument("--sheets", type=int, default=0, metavar="N",
                   help="write reports/tile_sheets/<label>.png with N random tiles per label")
    s = sub.add_parser("compare-tiling", help="compare tiling summaries of several output roots (after inspect)")
    s.add_argument("output_roots", nargs="+", help="paths.output_root of each tiling run")
    s.add_argument("--out", help="also write the table to this file")
    s.set_defaults(fn=cmd_compare_tiling)
    add("extract-features", cmd_features, "Gabor + colour descriptors for SOM")
    add("som-train", cmd_som_train, "train the SOM and assign tiles to nodes")
    add("som-sheets", cmd_som_sheets, "export node contact sheets + labelling template")
    add("som-apply-labels", cmd_som_apply, "majority-vote labeller CSVs -> labels_som.csv").add_argument("files", nargs="+")
    s = add("train", cmd_train, "train a model (supervised or --mean-teacher)")
    s.add_argument("--mean-teacher", action="store_true")
    s.add_argument("--run-name")
    for name, fn, h in (("evaluate", cmd_evaluate, "metrics on held-out holes"),
                        ("predict-log", cmd_predict_log, "depth-indexed predicted log for holes")):
        s = add(name, fn, h)
        s.add_argument("run", help="run folder (outputs/runs/<run>)")
        if name == "evaluate":
            s.add_argument("--split", default="test", choices=["train", "val", "test"])
        else:
            s.add_argument("--holes", nargs="*")
            s.add_argument("--no-compare", action="store_true")
    s = add("smoke-test", cmd_smoke, "synthetic end-to-end test of all stages (CPU)")
    s.add_argument("--out", help="keep outputs in this folder (deleted and recreated)")
    s.add_argument("--keep", action="store_true", help="keep the temporary folder")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    rc = a.fn(a)
    return rc if isinstance(rc, int) else 0


if __name__ == "__main__":
    sys.exit(main())
