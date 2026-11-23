#!/usr/bin/env python3
"""BASSGOLD corelog exploratory diagnostics. Run from the activated corelog_lab env.

Uses existing tiles/labels; keeps all datasets, run outputs and logs on Kingston.
Never commits or uploads data. Produces a compact private review ZIP.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

REPO = Path.home() / "code" / "drillcore-logging"
WORK = Path("/media/ra/KINGSTON/drillcore-logging-work")
ROOT60 = WORK / "outputs/bassgold_2026-11"
ROOT128 = WORK / "outputs/bassgold_2026-11_t128"
HOLE_A, HOLE_B = "DH26UMN_17a", "DH26UVD_11"

# These runs are *diagnostics*, not a publishable classification benchmark.
# Independent held-out-hole evaluation cannot support all five classes with two holes.
CASES = [
    # One-variable comparisons against the already completed unweighted baselines.
    ("weighted_60_mid", ROOT60, "both", 0.5, "balanced", 60, 25, "val"),
    ("weighted_128_mid", ROOT128, "both", 0.5, "balanced", 60, 25, "val"),
    # Same model, unweighted, but different contiguous validation regions.
    ("unweighted_128_shallow", ROOT128, "both", 0.0, "none", 50, 20, "val"),
    ("unweighted_128_deep", ROOT128, "both", 1.0, "none", 50, 20, "val"),
    # Honest held-out-hole stress tests: their unseen classes must be flagged.
    ("holdout_a_to_b_60", ROOT60, "a_to_b", 0.5, "none", 40, 15, "test"),
    ("holdout_b_to_a_60", ROOT60, "b_to_a", 0.5, "none", 40, 15, "test"),
]

EXISTING = [
    ("original_60_mid", ROOT60 / "runs/pilot_fixed_20261123_143119", "val"),
    ("original_128_mid", ROOT128 / "runs/pilot_t128_20261123_143600", "val"),
    ("extended_128_mid", ROOT128 / "runs/pilot_t128_extended_20261123_145424", "val"),
]


def run_stream(cmd: list[str], *, env: dict, outfile: Path) -> int:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    print("\n$", " ".join(cmd), flush=True)
    with outfile.open("w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="", flush=True)
            f.write(line)
        return p.wait()


def summarize(rows: list[tuple[str, Path, str]], target: Path) -> None:
    data = []
    for name, run, split in rows:
        mfile = run / f"eval_{split}/metrics.json"
        if not mfile.exists():
            print("NO METRICS:", name, mfile)
            continue
        m = json.loads(mfile.read_text())
        p = m["overall"]["per_class"]
        supported = [v["f1"] for v in p.values() if v["support"] > 0]
        ds = run / "data_summary.csv"
        counts = {}
        if ds.exists():
            with ds.open(newline="") as fp:
                counts = {r["split"]: r for r in csv.DictReader(fp)}
        record = {
            "run": name,
            "split": split,
            "n": m["overall"]["n"],
            "accuracy": round(m["overall"]["accuracy"], 4),
            "macro_f1_all5": round(m["overall"]["macro_f1"], 4),
            "macro_f1_supported": round(sum(supported) / len(supported), 4) if supported else "NA",
            "tdpand_recall": round(p.get("lith_TDPAND", {}).get("recall", 0), 4),
            "tpand_recall": round(p.get("lith_TPAND", {}).get("recall", 0), 4),
            "trcrhyp_recall": round(p.get("lith_TRCRHYP", {}).get("recall", 0), 4),
            "rs_recall": round(p.get("lith_RS", {}).get("recall", 0), 4),
            "clay_recall": round(p.get("lith_CLY", {}).get("recall", 0), 4),
            "tdpand_support": p.get("lith_TDPAND", {}).get("support", 0),
            "rs_support": p.get("lith_RS", {}).get("support", 0),
            "clay_support": p.get("lith_CLY", {}).get("support", 0),
            "train_tiles": counts.get("train", {}).get("n", ""),
            "test_tiles": counts.get("test", {}).get("n", ""),
            "run_path": str(run),
        }
        data.append(record)
    csv_path = target / "all_runs_summary.csv"
    if data:
        with csv_path.open("w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    print("\n======== FINISHED RUN SUMMARY ========")
    for r in data:
        print(f"{r['run']:26} {r['split']:4} n={r['n']:5} "
              f"acc={r['accuracy']:.3f} F1_5={r['macro_f1_all5']:.3f} "
              f"F1_supported={r['macro_f1_supported']} "
              f"TDPAND_recall={r['tdpand_recall']:.3f} (n={r['tdpand_support']}) "
              f"RS_n={r['rs_support']} CLY_n={r['clay_support']}")
    (target / "README.txt").write_text(
        "Exploratory BASSGOLD CNN experiments, 23 Nov 2026.\n"
        "Training and validation on depth blocks within same boreholes are not independent\n"
        "held-out-hole evaluation. Tests with held-out boreholes contain unseen classes.\n"
        "The very rare CLY class may be excessively weighted in balanced runs.\n"
        "Macro-F1-all5 includes classes with zero test/validation support; use per-class\n"
        "support and recall. Adjacent tiles are highly correlated. Do not publish these\n"
        "results as a validated geological benchmark. Private company data: DO NOT\n"
        "upload the diagnostic ZIP or CSVs to public GitHub.\n",
        encoding="utf-8",
    )
    zip_path = target / "corelog_diagnostic_review_PRIVATE.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for pth in (target / "README.txt", csv_path):
            if pth.exists():
                z.write(pth, pth.relative_to(target))
        for name, run, split in rows:
            for rel in ("history.csv", "data_summary.csv", "config.yaml", "train_summary.json",
                        f"eval_{split}/metrics.json", f"eval_{split}/confusion.csv"):
                path = run / rel
                if path.exists():
                    z.write(path, f"runs/{name}/{rel}")
    print("\nSummary CSV:", csv_path)
    print("Private review ZIP:", zip_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-only", action="store_true", help="Only rebuild report from saved registry")
    args = ap.parse_args()
    if not REPO.is_dir():
        print("ERROR: repository missing:", REPO)
        return 1
    if not ROOT60.is_dir() or not ROOT128.is_dir():
        print("ERROR: Kingston output directories missing. SSD must be mounted.")
        return 1
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    target = WORK / "diagnostics" / f"finish_{stamp}"
    target.mkdir(parents=True, exist_ok=True)
    print("Storing ALL new results on Kingston:", target)
    index_file = WORK / "diagnostics" / "latest_finish_registry.json"
    completed: list[tuple[str, Path, str]] = EXISTING.copy()
    if args.summary_only:
        if not index_file.is_file():
            print("No registry found:", index_file)
            return 1
        for d in json.loads(index_file.read_text()):
            completed.append((d["name"], Path(d["run"]), d["split"]))
        summarize(completed, target)
        return 0
    for root in (ROOT60, ROOT128):
        for rel in ("tiles/tiles.csv", "labels/labels_log.csv"):
            if not (root / rel).is_file():
                print("ERROR: missing precomputed file:", root / rel)
                return 1
    if not args.summary_only:
        env0 = dict(os.environ, CORELOG_OUTPUT_ROOT=str(ROOT60))
        rc = run_stream(["corelog", "validate", "-c", "configs/lab.yaml"],
                        env=env0, outfile=target / "validation_60.log")
        if rc:
            print("Validation returned failure; STOP before training. Log:", target / "validation_60.log")
            return rc
    registry: list[dict] = []
    for name, root, mode, pos, weight, epochs, patience, split in CASES:
        runname = f"diag_{name}_{stamp}"
        run_dir = root / "runs" / runname
        cfg_args = ["-c", "configs/lab.yaml"]
        if root == ROOT128:
            cfg_args += ["-c", "configs/experiments/tiles_128.yaml"]
        if mode == "both":
            trainholes, testholes = f"[{HOLE_A},{HOLE_B}]", "[]"
        elif mode == "a_to_b":
            trainholes, testholes = f"[{HOLE_A}]", f"[{HOLE_B}]"
        else:
            trainholes, testholes = f"[{HOLE_B}]", f"[{HOLE_A}]"
        overrides = [
            "split.mode=holes", f"split.train_holes={trainholes}",
            "split.val_holes=[]", f"split.test_holes={testholes}",
            f"split.val_from_train={{fraction: 0.2, buffer_m: 2.0, position: {pos}}}",
            "model.name=baseline", "train.device=cuda", "train.num_workers=0",
            f"train.epochs={epochs}", f"train.patience={patience}",
            "train.batch_growth_factor=1", "train.max_steps_per_epoch=null",
            "train.batch_size=16", "train.lr=0.0001", f"train.class_weighting={weight}",
        ]
        cmd = ["corelog", "train", *cfg_args, *sum((["--set", o] for o in overrides), []),
               "--run-name", runname]
        env = dict(os.environ, CORELOG_OUTPUT_ROOT=str(root))
        print("\n" + "=" * 72 + f"\nCASE: {name}  (output: {root})\n" + "=" * 72, flush=True)
        rc = run_stream(cmd, env=env, outfile=target / f"{name}_train.log")
        entry = {"name": name, "run": str(run_dir), "split": split, "train_returncode": rc}
        registry.append(entry)
        index_file.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        if rc:
            print(f"WARNING: {name} training returned {rc}; continuing to other cases.")
            continue
        eval_cmd = ["corelog", "evaluate", *cfg_args, str(run_dir), "--split", split]
        rc = run_stream(eval_cmd, env=env, outfile=target / f"{name}_evaluate.log")
        entry["evaluate_returncode"] = rc
        index_file.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        if rc:
            print(f"WARNING: {name} evaluation failed; other cases will still run.")
        completed.append((name, run_dir, split))
    summarize(completed, target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
