# Implementation plan and stage reports

*Updated 2026-11-18. It supersedes §11 of the reproduction audit.*

## Context and constraints
- **Development** happens on a personal laptop (macOS, CPU). **Real experiments** run later on a **lab machine running Ubuntu** with an NVIDIA GPU. Windows/WSL is supported as a fallback only (see INSTALL appendix).
- The RAs prepare photographs and geological journals independently. The pipeline must accept their delivery through a documented manifest, with no code changes.
- No company data, raw photographs, credentials, generated files or checkpoints in Git. The authors' unlicensed code is kept outside the repository (NOTICE.md).
- Nothing is pushed to GitHub until the project owner approves.

## Design decisions
| Decision | Choice | Reason |
|---|---|---|
| Framework | **PyTorch** | Clean CUDA install on Ubuntu; explicit training loop makes the Mean Teacher gradient testable (the paper's MT bug was invisible in Keras). |
| Join key | `slice_id` everywhere | Removes the positional-alignment bugs found in the audit (DL-2, T4c). |
| Splits | **By drill hole** only (explicit, or leave-one-hole-out) | Random tile splits put a neighbour of 96% of test tiles in training (audit T4e). |
| Labels | Journal labels (depth overlap, purity ≥ 0.8) **and/or** SOM labels | Journals give an independent ground truth; SOM reproduces the paper and can be scored against the journals. |
| Tiles | 300×60 px, no overlap, depth-registered | Paper geometry; overlap leaks into evaluation. |
| Paths | `data_root` / `output_root` from a local config or env vars | Same code on laptop and lab, nothing hard-coded. |
| Formats | CSV for manifests, labels and logs; PNG tiles; NPZ features; `.pt` checkpoints | Readable by RAs and geologists; no heavy dependencies. |

## Deviations from the paper (intentional, configurable)
| Item | Paper / authors' code | Here (default) | Config key |
|---|---|---|---|
| Gabor | \|G\| ∗ I (code) / \|I ∗ G\| (Eq. 5) | \|I ∗ G\| via FFT | — |
| Gabor frequencies | up to 1.0 cyc/px (aliases) | 0.05 … 0.45 (< Nyquist) | `features.gabor.frequencies` |
| Gabor boundary | wrap | reflect | `features.gabor.boundary` |
| Colour normalization | equalize (code) / min-max (text) | min-max | `features.color.normalize` |
| SOM input | 201–202 cols incl. row index (code) | 48 Gabor + 48 colour PCs (paper) | `features.color.pca_components` |
| SOM init / order | PCA scores, depth order | `pca_weights_init`, random order, seeded | `som.init`, `som.random_order` |
| Split | random tiles 70/10/20 | by hole | `split.*` |
| Baseline head | +2×Dense128 (code) | Fig. 8 (single dense) | `model.head` |
| VGG16 input | 60×300 (code) / 224 (text) | 60×300 | `model.input_size` |
| Mean Teacher consistency | no gradient (code behind Table 3) | differentiable, independent noise, labeled+unlabeled | `mean_teacher.*` |
| Unlabeled pool | all holes | training holes only | — |
| Metrics | accuracy | + macro-F1, per class, excluding non-core, per hole, bootstrap CI, depth-weighted agreement with journals | `evaluation.*` |

## Stage reports

> **All numbers in these stage reports come from SYNTHETIC images** (`corelog.synthetic`, classes `syn_class_1..3` + `non_core`). They show that the code works. They say **nothing** about real geological classification performance. Every output computed on the synthetic set is stamped `"data_provenance": "SYNTHETIC"` (in `provenance.json`, `metrics.json` and the predicted-log `summary.json`), and run folders get a `SYNTHETIC_DATA_NOT_GEOLOGICAL_RESULTS.txt` marker.

All tests use **synthetic data only** (`corelog.synthetic`: 4 synthetic holes SYN_A–D × 3 photos × 3 rows, 144 tiles, 4 synthetic classes with distinct colour and texture). Environment: macOS arm64, Python 3.12.10, torch 2.15.0 (CPU), numpy 2.5.5, scipy 1.18.1, scikit-image 0.26.0, scikit-learn 1.9.1, pandas 3.1.1, MiniSom 2.3.6.

### Stage 0: Foundation (config, manifests, tiling, labels, splits)
- **Files:** `config.py`, `images.py`, `manifest.py`, `tiling.py`, `labels.py`, `splits.py`, `synthetic.py`, `configs/{base,smoke,lab.example}.yaml`
- **Tests:** `tests/test_data_pipeline.py` (10 tests)
- **Results:** all pass. Config layering, overrides and env vars work; committed configs contain no absolute paths. The validator catches missing images, boxes outside the image, inverted depths, overlapping rows, missing columns and unmapped codes. Tile depths are contiguous; `rtl` is handled; the trailing partial window is shifted back. Purity labelling marks contact tiles as unlabeled. Hole splits are disjoint, and leave-one-hole-out covers every hole once.
- **Limitations:** no automatic tray/row segmentation and no lens-distortion correction. Row boxes come from the RAs (manifest). Depth is linear within a row (core loss inside a row is not modelled).

### Stage 1: CNN baseline and supervised training
- **Files:** `models/__init__.py` (BaselineCNN, VGG16Transfer), `data/dataset.py`, `train/common.py`, `train/supervised.py`, `runinfo.py`
- **Tests:** `tests/test_stage1_cnn.py` (6 tests)
- **Results:** all pass. Architecture matches Fig. 8 (32/32/64 3×3 convs, single dense layer of 64·8·38 inputs). VGG16 builds without downloading, with a frozen conv base. The batch-size schedule and augmentation are correct. `prepare()` keeps unlabeled tiles from training holes only. **Learning check:** 8 epochs on the synthetic data reach validation accuracy 1.0 on a held-out hole (chance 0.25); the best epoch is restored and saved. Run folders contain config, environment, history and checkpoint.
- **Limitations:** VGG16 was tested with random weights only (no 530 MB download on the laptop). No GPU was tested; CUDA is exercised only on the lab machine. Mixed precision is not implemented.

### Stage 2: Gabor/colour features and SOM labelling
- **Files:** `features/descriptors.py`, `som/cluster.py`
- **Tests:** `tests/test_stage2_features_som.py` (7 tests)
- **Results:** all pass. The FFT Gabor equals `ndimage.convolve` exactly (reflect and wrap). On seam-free gratings it classifies orientation at ≥ 0.95, where the authors' code scored 0.30 (audit test 1b). Frequencies ≥ 0.5 are rejected. The SOM input is exactly descriptors (no index column) and is invariant to row order. *Found and fixed during testing:* scikit-learn's automatic randomized PCA made the colour PCs depend on row order, so the solver is now `covariance_eigh`. End to end: SOM node purity against synthetic journal labels > 0.7 (0.99 in the smoke run); contact sheets, template and majority vote (ties → unlabeled) work; SOM results are reproducible with a fixed seed.
- **Limitations:** the SOM's usefulness on real core is unknown until it is scored against the journals (`node_vs_log.csv`). The labelling UI is file-based (PNG sheets + CSV), not interactive.

### Stage 3: Mean Teacher
- **Files:** `train/mean_teacher.py`
- **Tests:** `tests/test_stage3_mean_teacher.py` (5 tests)
- **Results:** all pass. **The consistency loss has a non-zero gradient in the student and none in the teacher** (the regression test for the paper's bug). λ changes the update. The EMA follows θ′ ← αθ′ + (1−α)θ, including BN buffers. The ramp-up matches the paper (exp(−5) at epoch 0, 1 after the ramp). A training run records λ, EMA α, CE, consistency, and student and teacher validation metrics, and saves the teacher. *Found during testing:* synthetic log boundaries aligned with tile edges, so there were no unlabeled tiles. The generator now produces realistic contact tiles.
- **Limitations:** VGG16 as a Mean Teacher student is supported by config but only tested through the shared model builder. Whether MT helps on real data is an open experimental question (the paper's evidence for it is invalid, see audit §5).

### Stage 4: Evaluation, depth-indexed logs, CLI, smoke test
- **Files:** `evaluate.py`, `inference.py`, `cli.py`, `.github/workflows/ci.yml`
- **Tests:** `tests/test_stage4_eval_log_cli.py` (7 tests, including the end-to-end smoke test and synthetic-provenance stamping); `tests/test_repo_hygiene.py` (3 tests: no data, checkpoints or large files in Git; no source file hidden by `.gitignore`)
- **Results:** all pass. Interval merging, the three non-core policies, short-interval absorption (with guaranteed termination) and gap handling work. Depth-weighted agreement is checked against a hand-computed 0.75. `validate` returns exit code 1 on a broken dataset. A trained model evaluated through `evaluate_run`/`predict_log` scores above chance on the held-out hole (accuracy > 0.6, journal agreement > 0.5), which checks label/prediction alignment along the whole path. `corelog smoke-test` runs all 13 steps in about 9 s on the laptop CPU.
- **Limitations:** bootstrap CIs with fewer than 3 test holes are tile-level and therefore optimistic (labelled as such in the output). No plotting of strip logs yet (CSV only).

### Stage 5: Rock-only deliveries and two-hole splits
*Added 2026-11-23, after reading the RAs' delivery documentation. No real data was used; the delivery itself was not available on the development laptop.*
- **Files:** `manifest.py` (masks, BOM, per-row box completeness, empty codes, metres per class and hole, row scale check), `tiling.py` (mask-aware tiling, `rock_fraction`, `min_rock_fraction`, `dropped_tiles.csv`), `labels.py` (`contact_buffer_m`), `features/descriptors.py` (`use_mask`), `splits.py` (`val_from_train`, two-hole leave-one-hole-out), `evaluate.py` (depth-block bootstrap, `label_counts`, within-hole validation note), `diagnostics.py` + `corelog inspect` / `compare-tiling`, `synthetic.make_rock_only_dataset`, `configs/lab.two_holes.example.yaml`, `configs/rock_only_synthetic.yaml`, `docs/LAB_RUNBOOK.md`
- **Tests:** `tests/test_stage5_rock_only_two_holes.py` (18 tests, synthetic rock-only delivery with two holes)
- **Results:** all pass, and the 38 earlier tests and the smoke test are unchanged.
  - A CSV that starts with a byte-order mark now validates.
  - Masks are resolved from a column or a template and size-checked, and they follow the flip, trim and resize of the image.
  - Low-rock tiles are dropped and listed.
  - With a 0.10 m contact buffer, a tile 5 cm from a contact becomes unlabeled. *Found during testing:* a purity test on the widened span did not do this for long tiles, so the rule is now strict.
  - Masked descriptors: the black background adds no histogram mass, and the piece outline changes the Gabor features at most half as much as without the mask.
  - The validation block lies inside the training hole with the buffer respected, and the test hole stays whole.
  - The bootstrap resamples depth blocks when there is one test hole.
  - End to end on two synthetic holes: tiles, labels, inspect, masked features, SOM, training, evaluation and the predicted log all run, and every output is stamped SYNTHETIC.
- **Tile-geometry comparison (added after the lab command dry run):** on synthetic full-height rows, the paper's 60×300 tiles produced no tiles at all. Each row shrinks to 60 px high, so it needs a length/height ratio ≥ 5. The default stays 60×300 (paper reference); `configs/experiments/tiles_128.yaml` is an optional alternative. `make-tiles` now writes `tiles/rows_tiling.csv` (per row: size, ratio, windows, kept tiles, rock covered) and explains a no-tiles failure. `inspect` writes `reports/tiling_summary.json`, and `corelog compare-tiling` tabulates two geometries: tiles generated and kept, % rows with zero tiles, rock-mask coverage, metres per tile, and % ambiguous labels by cause. Tests: 4 more (18 in stage 5).
- **Limitations:**
  - Within-hole validation is not independent; it is used only for early stopping.
  - With two holes, every result describes one held-out hole. There is no estimate of between-hole variation.
  - The class balance, tile geometry and px/m of the real delivery are unknown until `validate` and `inspect` run on the lab machine.

## Remaining work (after RA delivery)
1. **P0 decisions:** class scheme and `class_map` with the geologists; which journal attribute(s) to model; hole split. **The first delivery has two holes:** 2 folds (train one hole, test the other), validation from a buffered depth block of the training hole (`split.val_from_train`). Four-fold leave-one-hole-out is not possible yet.
2. **Dataset QA:** run `validate`, inspect tiles per hole, check class balance and the share of unlabeled tiles.
3. **Baselines on the GPU:** baseline CNN, VGG16 (60×300 and 224), each over 4 folds × ≥ 3 seeds.
4. **Paper replication on our data:** SOM labelling by our geologists, scored against the journals; training on SOM labels vs journal labels.
5. **Mean Teacher** vs supervised under identical folds.
6. **Depth logs:** smoothing and minimum-interval tuning on validation holes; comparison with journals.
7. Nice to have: strip-log figures, row auto-detection helper for RAs, colour-chart calibration if charts are present, mixed-precision training.

Approval gates: (a) this implementation before the first push to GitHub; (b) the P0 decisions before the first real training run.
