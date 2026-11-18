"""Stage 4: evaluation, depth-log generation, CLI and the end-to-end smoke test."""
import json

import numpy as np
import pandas as pd
import pytest

from corelog import cli
from corelog.inference import compare_with_log, to_log

CLS = ["a", "b", "non_core"]


def _pred(seq, probs=0.9, step=0.3, start=10.0, hole="H"):
    rows = []
    for i, c in enumerate(seq):
        p = {f"p_{k}": (probs if k == c else (1 - probs) / 2) for k in CLS}
        if c == "non_core":
            p["p_b"] = (1 - probs) / 2 + 0.01  # b is the second-best class
        rows.append({"hole_id": hole, "depth_from_m": start + i * step, "depth_to_m": start + (i + 1) * step, **p})
    return pd.DataFrame(rows)


def test_to_log_merges_and_non_core_policies():
    pred = _pred(["a", "a", "b", "b", "b", "non_core", "a"])
    log = to_log(pred, CLS, non_core_policy="keep")
    assert log["predicted_class"].tolist() == ["a", "b", "non_core", "a"]
    assert log.loc[0, "depth_from_m"] == 10.0 and log.loc[0, "depth_to_m"] == pytest.approx(10.6)
    sb = to_log(pred, CLS, non_core_policy="second_best")
    assert sb["predicted_class"].tolist() == ["a", "b", "a"] and sb.loc[1, "n_tiles"] == 4
    dr = to_log(pred, CLS, non_core_policy="drop")
    assert "non_core" not in set(dr["predicted_class"]) and dr["n_tiles"].sum() == 6


def test_to_log_min_interval_and_gaps():
    pred = _pred(["a", "a", "a", "b", "a", "a"])
    assert to_log(pred, CLS, min_interval_m=0.5)["predicted_class"].tolist() == ["a"]
    gap = pd.concat([_pred(["a"] * 2), _pred(["a"] * 2, start=20.0)])
    assert len(to_log(gap, CLS)) == 2                              # missing core -> separate intervals
    lone = pd.concat([_pred(["a"] * 4), _pred(["b"], start=30.0)])
    assert to_log(lone, CLS, min_interval_m=0.5)["predicted_class"].tolist() == ["a", "b"]  # terminates


def test_compare_with_log_depth_weighted():
    plog = pd.DataFrame({"hole_id": "H", "depth_from_m": [0.0, 1.0], "depth_to_m": [1.0, 2.0], "predicted_class": ["a", "b"]})
    jlog = pd.DataFrame({"hole_id": "H", "depth_from_m": [0.0, 1.5], "depth_to_m": [1.5, 2.0], "lithology_code": ["A", "B"]})
    r = compare_with_log(plog, jlog, {"A": "a", "B": "b"})["H"]
    assert r["depth_weighted_agreement"] == pytest.approx(0.75, abs=0.01) and r["compared_m"] == pytest.approx(2.0, abs=0.02)


def test_cli_validate_fails_on_bad_dataset(tmp_path, synth_root):
    import shutil
    bad = tmp_path / "bad"
    shutil.copytree(synth_root, bad)
    rows = pd.read_csv(bad / "manifests/rows.csv")
    rows.loc[0, "image_path"] = "photos/nope.png"
    rows.to_csv(bad / "manifests/rows.csv", index=False)
    rc = cli.main(["validate", "-c", "configs/smoke.yaml", "--set", f"paths.data_root={bad}",
                   "--set", f"paths.output_root={tmp_path / 'out'}"])
    assert rc == 1 and "FAIL" in (tmp_path / "out/reports/validation.md").read_text()


@pytest.mark.slow
def test_smoke_end_to_end(tmp_path):
    summary = cli.cmd_smoke(cli.build_parser().parse_args(["smoke-test", "--out", str(tmp_path / "smoke")]))
    assert summary["tiles"] > 100
    assert 0 <= summary["test_accuracy_supervised"] <= 1 and summary["predicted_intervals"] > 0
    runs = tmp_path / "smoke/outputs/runs"
    for r in ("smoke_supervised", "smoke_meanteacher"):
        m = json.loads((runs / r / "eval_test/metrics.json").read_text())
        assert m["holes"] == ["SYN_D"]                             # test hole only
        assert {"accuracy", "macro_f1", "per_class"} <= set(m["overall"])
    assert (tmp_path / "smoke/outputs/predicted_logs/smoke_supervised/predicted_log.csv").exists()
    env = json.loads((runs / "smoke_supervised/environment.json").read_text())
    assert env["packages"]["torch"]
    assert summary["data_provenance"].startswith("SYNTHETIC")
    assert (runs / "smoke_supervised/SYNTHETIC_DATA_NOT_GEOLOGICAL_RESULTS.txt").exists()
    m = json.loads((runs / "smoke_supervised/eval_test/metrics.json").read_text())
    assert m["data_provenance"] == "SYNTHETIC" and "NOT a geological" in m["warning"]
    s = json.loads((tmp_path / "smoke/outputs/predicted_logs/smoke_supervised/summary.json").read_text())
    assert s["data_provenance"] == "SYNTHETIC"


def test_provenance_real_dataset_is_not_flagged(tmp_path):
    from corelog.runinfo import provenance
    (tmp_path / "manifests").mkdir()
    assert provenance({"paths": {"data_root": str(tmp_path)}})["data_provenance"].startswith("real")


def test_trained_model_evaluates_and_logs_above_chance(tiled):
    """A model that learned (see stage 1) must also score well through evaluate_run and
    predict_log - guards against label/prediction misalignment in the evaluation path."""
    import copy

    from corelog.evaluate import evaluate_run
    from corelog.inference import predict_log
    from corelog.train.supervised import train_supervised
    cfg = copy.deepcopy(tiled[0])
    cfg["train"].update({"epochs": 10, "max_steps_per_epoch": None, "patience": 10, "lr": 1e-3})
    run = train_supervised(cfg, run_name="pytest_eval_path")
    res = evaluate_run(run, cfg)
    assert res["holes"] == ["SYN_D"] and res["overall"]["accuracy"] > 0.6     # chance = 0.25
    pred = pd.read_csv(run / "eval_test/predictions.csv")
    assert (pred["label"] == pred["predicted"]).mean() == pytest.approx(res["overall"]["accuracy"])
    pl = predict_log(run, cfg, holes=["SYN_D"])
    assert pl["comparison_with_journal"]["SYN_D"]["depth_weighted_agreement"] > 0.5
