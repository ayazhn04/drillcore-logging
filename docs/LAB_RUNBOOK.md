# Lab runbook: first real-data run (Ubuntu, two drill holes)

Copy-paste procedure for the lab computer. It assumes a **rock-only, pre-cropped delivery**: one image per core row, black background, separate rock masks, and `rows.csv` with empty `x0..y1`. It also assumes **two holes**. Every command is safe to rerun.

Placeholders, which you replace in your shell or in `configs/lab.yaml` and never commit:

| Placeholder | Meaning |
|---|---|
| `<DATASET_ROOT>` | folder that contains `manifests/`, `photos/`, `masks/` |
| `<OUT_ROOT>` | output folder on a **local Linux disk** (not exFAT, not inside the repo) |
| `<GITHUB_URL>` | the private repository URL |
| `HOLE_A`, `HOLE_B` | the two hole IDs, spelled exactly as in `rows.csv` |
| `CODE_*`, `class_*` | journal codes and model classes agreed with the geologists |

Stop at every **CHECK** and write the result into the observation sheet (section 8) before going on.

---

## 0. Before you start

- Do not copy the dataset into the repository, and do not `git add` anything under `<DATASET_ROOT>` or `<OUT_ROOT>`.
- Clone the code into your home folder (ext4). External drives formatted exFAT cannot hold a Python virtual environment.
- Put `<OUT_ROOT>` on a local disk: tiles are many small files, and a removable drive is slow and can be unplugged mid-run.
- On a **shared** account, use a read-only deploy key (or a fine-grained token limited to this repository) for cloning, not your personal GitHub login.

## 1. GPU and Python

```bash
nvidia-smi
```
**CHECK 1:** GPU model, driver version, and the "CUDA Version" shown.

```bash
python3.12 --version || python3.13 --version
```
**CHECK 2:** which Python 3.12 or 3.13 is available. The pins were tested on 3.12. Use 3.13 only if 3.12 is missing.

## 2. Clone and install

```bash
mkdir -p ~/code && cd ~/code && git clone <GITHUB_URL> drillcore-logging && cd drillcore-logging
```
```bash
git log -1 --oneline
```
Pick the CUDA tag from <https://pytorch.org/get-started/locally/> for torch 2.15. **RTX 50-series (Blackwell) cards need a CUDA 12.8 or newer build** (`cu128` or newer). Then:
```bash
bash scripts/setup_ubuntu.sh <CUDA_TAG>
```
With only Python 3.13:
```bash
PYTHON=python3.13 bash scripts/setup_ubuntu.sh <CUDA_TAG>
```
```bash
source .venv/bin/activate
```
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
```
**CHECK 3:** `True`, the GPU name, and an arch list that includes the card's compute capability (`sm_120` for RTX 50-series). If `is_available()` is `False`, or a warning says the GPU is not supported, the CUDA tag is wrong.

```bash
pytest -q && corelog smoke-test
```
**CHECK 4:** the number of passed tests, and `[smoke] PASS`. These are synthetic numbers, not results.

## 3. Configure the dataset (local only)

```bash
cp configs/lab.two_holes.example.yaml configs/lab.yaml
```
Edit `configs/lab.yaml`:
- `paths.data_root: <DATASET_ROOT>` and `paths.output_root: <OUT_ROOT>`
- `split.train_holes: [HOLE_A]`, `split.test_holes: [HOLE_B]`
- `labels.classes` and `labels.class_map`: from the delivery's `manifests/codes.csv`, agreed with the geologists

Then check that Git still ignores the file:
```bash
git status --short && git check-ignore -v configs/lab.yaml
```
**CHECK 5:** `configs/lab.yaml` is reported as ignored, and `git status` shows no data files.

## 4. Validate (no images changed, nothing trained)

```bash
corelog validate -c configs/lab.yaml
```
Writes `<OUT_ROOT>/reports/validation.md` and `class_metres_by_hole.csv`. The RAs' own validator passing does **not** replace this step.

**CHECK 6:**
- PASS or FAIL, with the number of errors and warnings. Copy every error and warning.
- "rock masks found and size-checked": the count must equal the number of rows.
- "row height / row length / scale px/m" per hole.
- Any warning about **rows whose px/m is outside the hole median**: list the lines. They point at row depth intervals that do not match the core in the image.
- "photographed metres per class and hole": which classes are **absent from one hole**.
- Unmapped codes and empty codes.

If a class exists in only one hole, it cannot be both trained and tested. Merge it, drop it, or accept that it is not evaluated, and decide this with the geologists before training.

Repeat for the second attribute, if one is delivered:
```bash
corelog validate -c configs/lab.yaml --set paths.log_intervals=manifests/log_<attribute>.csv
```

## 5. Tiles, labels, inspection

```bash
corelog make-tiles -c configs/lab.yaml
```
```bash
corelog make-labels -c configs/lab.yaml
```
```bash
corelog inspect -c configs/lab.yaml --sheets 24
```
**CHECK 7** (from `<OUT_ROOT>/reports/inspection.md`):
- Number of tiles per hole, and metres per tile (median, 5–95 %).
- `dropped_low_rock_tiles` and the rock-fraction quantiles. Adjust `tiling.min_rock_fraction` if needed and rerun.
- The tables of labelled tiles by class × hole and by class × split (train / val / test / none).
- Journal intervals per class and split. One or two intervals per class means the tiles are **not independent samples**.
- Unlabeled share, which comes from contacts, the contact buffer, unmapped codes and gaps.
- Warnings about classes with 0 training or 0 test tiles.
- A warning that **median rock fraction differs strongly between labels**: the CNN could then learn piece outlines instead of rock.
- Open `reports/tile_sheets/*.png`. Is the core visible at a useful resolution? Does the label match what you see? Are there leftovers of the black background, tags or wrap?

## 6. Pilot (small, minutes)

Purpose: to check that the pipeline runs on real data and that numbers are plausible. **This is not the experiment.** Keep `max_steps_per_epoch` unset, and use few epochs.

Fold 1 (train HOLE_A, test HOLE_B):
```bash
corelog train -c configs/lab.yaml --set train.epochs=40 --set train.patience=10 --run-name pilot_f1_baseline
```
```bash
corelog evaluate -c configs/lab.yaml <OUT_ROOT>/runs/pilot_f1_baseline
```
```bash
corelog predict-log -c configs/lab.yaml <OUT_ROOT>/runs/pilot_f1_baseline --holes HOLE_B
```
Fold 2 (swap the holes):
```bash
corelog train -c configs/lab.yaml --set "split.train_holes=[HOLE_B]" --set "split.test_holes=[HOLE_A]" --set train.epochs=40 --set train.patience=10 --run-name pilot_f2_baseline
```
```bash
corelog evaluate -c configs/lab.yaml <OUT_ROOT>/runs/pilot_f2_baseline
```
`evaluate` and `predict-log` always use the classes and the split stored with the run (`<run>/config.yaml`). Only `paths` are taken from the current config.

**CHECK 8:**
- `history.csv`: does the validation loss go down at all? At which epoch did training stop?
- `data_summary.csv`: tiles per class in train, val and test.
- `eval_test/metrics.json`: accuracy, macro-F1, per-class recall, `label_counts`, and the bootstrap interval with its unit (depth block).
- `eval_test/confusion.csv`.
- Compare the result with a **majority-class baseline**: the share of the most frequent class in the test hole, from `label_counts`.
- `predict-log`: predicted intervals versus journal intervals, and the depth-weighted agreement.
- Seconds per epoch on the GPU.

Tile geometry is the decision to test next. With 60 px high tiles, a full-height row image is shrunk many times, and texture may be lost. One comparison run:
```bash
corelog make-tiles -c configs/lab.yaml --set paths.output_root=<OUT_ROOT>_t128 --set tiling.tile_height=128 --set tiling.tile_width=128 --set tiling.stride=128
```
Then run `make-labels`, `inspect --sheets 24` and the same fold-1 training, each with `--set paths.output_root=<OUT_ROOT>_t128 --set "model.input_size=[128, 128]"`.

Optional SOM check (paper workflow, short):
```bash
corelog extract-features -c configs/lab.yaml && corelog som-train -c configs/lab.yaml --set som.iterations=200000
```
**CHECK 9:** `weighted_node_purity` against the journal, in the printed JSON and in `som/node_vs_log.csv`.

## 7. Keep results out of Git

```bash
git status --short
```
It must show no files from `<OUT_ROOT>` or `<DATASET_ROOT>`. Share results as the small report files (`reports/*.md`, `*.csv`, `metrics.json`) through approved channels only, never through the repository.

## 8. Observation sheet (fill in on the lab computer)

| # | Observation | Value |
|---|---|---|
| 1 | GPU, driver, CUDA version (`nvidia-smi`) | |
| 2 | Python version used for `.venv` | |
| 3 | torch version, CUDA tag, `cuda_available`, arch list | |
| 4 | `pytest` result, smoke test PASS | |
| 5 | `lab.yaml` ignored by Git | |
| 6 | `validate`: PASS/FAIL, errors, warnings (copy all) | |
| 6a | rows / masks found / images checked | |
| 6b | row height px, row length px, px/m per hole (median, 5–95 %) | |
| 6c | rows flagged for inconsistent px/m (lines) | |
| 6d | metres per class per hole (`class_metres_by_hole.csv`) | |
| 6e | classes absent from a hole; unmapped or empty codes | |
| 7 | tiles per hole; metres per tile; dropped low-rock tiles | |
| 7a | rock fraction quantiles; median rock fraction by label | |
| 7b | labelled tiles class × split; unlabeled share | |
| 7c | journal intervals per class in train and test | |
| 7d | visual check of tile sheets (resolution, label plausibility, leftovers) | |
| 8 | pilot fold 1 and fold 2: best epoch, macro-F1, per-class recall, majority baseline, bootstrap CI | |
| 8a | depth-weighted agreement of the predicted log | |
| 8b | seconds per epoch | |
| 8c | 60×300 versus 128×128 tiles: macro-F1, tile sheets | |
| 9 | SOM node purity against the journal (optional) | |
