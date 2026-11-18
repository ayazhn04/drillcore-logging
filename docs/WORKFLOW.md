# Lab workflow: from the RAs' dataset to evaluated models and predicted logs

All commands take `-c configs/lab.yaml` (your local, git-ignored config) and optional `--set key=value` overrides. Outputs go to `paths.output_root`.

## 0. One-time setup
1. Install: [INSTALL.md](INSTALL.md); `pytest` and `corelog smoke-test` must pass.
2. Create your local config:
   ```bash
   cp configs/lab.example.yaml configs/lab.yaml
   ```
   Then edit `paths.data_root`, `paths.output_root`, `labels.classes`, `labels.class_map` and `split.*`. This file is git-ignored.

## 1. Import and validate the dataset
The RAs deliver a folder in the format of [DATASET_MANIFEST.md](DATASET_MANIFEST.md).
```bash
corelog validate -c configs/lab.yaml
```
Fix every **error** before continuing; read the warnings (unmapped lithology codes, holes without logs, depth coverage). The report is saved to `outputs/reports/validation.md`.

## 2. Tiles and journal labels
```bash
corelog make-tiles  -c configs/lab.yaml    # outputs/tiles/<hole>/*.png + tiles/tiles.csv (slice_id, depths)
corelog make-labels -c configs/lab.yaml    # outputs/labels/labels_log.csv (label, purity per tile)
```
Look at a few tiles per hole, and check `labels_log.csv` (share of `unlabeled` and class balance) before training.

## 3. (Optional) SOM labelling, as in the paper
Use this when journal labels are missing or too coarse, or to test the paper's weak-labelling idea against the journals.
```bash
corelog extract-features -c configs/lab.yaml --set features.n_jobs=16
corelog som-train        -c configs/lab.yaml          # prints weighted node purity vs journal labels
corelog som-sheets       -c configs/lab.yaml          # outputs/som/sheets/*.png + labeling/TEMPLATE_copy_me.csv
```
Each geologist copies `TEMPLATE_copy_me.csv` to `<name>.csv` and fills one class per node (empty = ambiguous). Then:
```bash
corelog som-apply-labels -c configs/lab.yaml outputs/som/labeling/geo1.csv outputs/som/labeling/geo2.csv
```
To train on SOM labels, use `--set split.label_source=som`, or `log_then_som` to fill journal gaps with SOM labels.

## 4. Train
Splits are always **by drill hole** (`split.mode: holes` or `leave_one_hole_out`). Never report results from `split.mode: random`.
```bash
corelog train -c configs/lab.yaml                                    # baseline CNN, supervised
corelog train -c configs/lab.yaml --set model.name=vgg16             # VGG16 transfer (frozen base)
corelog train -c configs/lab.yaml --mean-teacher                     # Mean Teacher (baseline student)
corelog train -c configs/lab.yaml --run-name loho_fold0 --set split.mode=leave_one_hole_out --set split.fold=0
```
Each run folder `outputs/runs/<run>/` contains `model.pt`, `history.csv`, `config.yaml`, `environment.json`, `data_summary.csv` and `train_summary.json`.

Useful overrides: `train.epochs`, `train.lr`, `train.batch_size`, `train.patience`, `train.class_weighting=balanced`, `model.head=notebook`, `model.input_size=[224,224]`, `mean_teacher.consistency_max`.

For experiment variants, keep small YAML files in `configs/experiments/` (committed, no paths) and stack them:
`-c configs/lab.yaml -c configs/experiments/vgg_224.yaml`.

## 5. Evaluate on held-out holes
```bash
corelog evaluate -c configs/lab.yaml outputs/runs/<run>
```
This writes `eval_test/metrics.json`: accuracy, macro-F1, per-class precision/recall/F1, metrics excluding `non_core`, per-hole accuracy and bootstrap 95% CIs (by hole when there are ≥ 3 test holes). It also writes `confusion.csv` and `predictions.csv`. Classes and splits are always taken from the run's own `config.yaml`.

With 4 holes, run all 4 leave-one-hole-out folds and report the mean and range across folds.

## 6. Depth-indexed predicted logs
```bash
corelog predict-log -c configs/lab.yaml outputs/runs/<run> --holes HOLE_A
```
This writes `outputs/predicted_logs/<run>/predicted_log.csv` (hole, depth_from_m, depth_to_m, predicted_class, mean_probability, n_tiles), `tile_predictions.csv`, and `summary.json` with the **depth-weighted agreement** against the journal. Controls:
- `inference.smoothing_window_tiles` (e.g. 5): moving average of class probabilities along depth
- `inference.non_core_policy`: `second_best` (paper), `keep` or `drop`
- `inference.min_interval_m`: absorb predicted intervals shorter than this

New holes without journals can be logged too: add them to `rows.csv`, run `make-tiles`, then `predict-log --holes NEW_HOLE --no-compare`.

## 7. What to keep and share
- Commit code and configs only. **Never** commit data, tiles, runs or checkpoints (`.gitignore` enforces this).
- Archive each reported run folder (it contains its config and environment) with the dataset version it used.
