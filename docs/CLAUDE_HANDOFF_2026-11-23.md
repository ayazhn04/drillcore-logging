# CLAUDE HANDOFF — BASSGOLD drill-core lithology classification

**Status:** 2026-11-23. **Stage:** baseline-CNN diagnostics finished; geological verification pending.
**Repository:** `ayazhn04/drillcore-logging`, branch `review/corelog-contact-sheets`.
**Important:** This repository is PUBLIC. The underlying BASSGOLD company data, row images,
rock masks, confidential journals, predictions with depths, model weights and private
experiment ZIPs must remain on the Kingston SSD unless their publication is authorized.

## Task and scientific boundary

We developed an independent clean-room implementation inspired by Serdoun et al. (2026).
It takes depth-indexed drill-core photographs, tiles them, assigns lithology labels
from journal intervals, trains a baseline CNN, and predicts one of five classes:
`lith_CLY`, `lith_RS`, `lith_TDPAND`, `lith_TPAND`, `lith_TRCRHYP`.
This is **not** a reproduction of the paper authors' empirical results.

Dataset: 2 holes (`DH26UMN_17a`, `DH26UVD_11`), 140 original photographs,
556 prepared row photographs and corresponding masks. Validator: PASS, 0 errors,
4 missing-class-by-hole warnings. Existing tiling: 60x300 (~1,659 kept tiles)
and 128x128 (~7,574 kept tiles). Tiles are correlated within rows and intervals.
Only `TPAND` appears in both holes; other classes are absent from one hole.
Therefore the current data cannot establish five-class generalization to a new hole.

## Local environment, paths and constraints

- Ubuntu repo checkout: `/home/ra/code/drillcore-logging`
- Conda env: `corelog_lab`
- Git branch: `review/corelog-contact-sheets` (NOT `main`)
- Source files: `/media/ra/KINGSTON/BASSGOLD` (do not publish)
- 60x300 output root: `/media/ra/KINGSTON/drillcore-logging-work/outputs/bassgold_2026-11`
- 128x128 output root: `/media/ra/KINGSTON/drillcore-logging-work/outputs/bassgold_2026-11_t128`
- Private diagnostic report: `/media/ra/KINGSTON/drillcore-logging-work/diagnostics/finish_20261123_151926/corelog_diagnostic_review_PRIVATE.zip`
- Existing visual ZIP in this branch: `review_artifacts/bassgold_visual_review.zip`
- Lab configuration is `configs/lab.yaml`, GIT-IGNORED; do not stage it.
- During training set `train.num_workers=0` (multiprocessing caused permission errors).
- Use `CORELOG_OUTPUT_ROOT` pointing at the appropriate Kingston root.
- Do NOT rerun all training: saved runs and predictions already exist.

## Completed runs: metrics (exploratory only)

These results use saved best-loss checkpoints, not necessarily maximum-accuracy epochs.
Within-hole validation blocks are not independent holes. Fixed-five-class macro-F1
counts absent classes as zero; always inspect supports and per-class metrics.

| Run | Split | n | Accuracy | Macro-F1 (5) | TDPAND recall |
|---|---|---:|---:|---:|---:|
| original_60_mid | val | 328 | 93.29% | 0.400 | 6.7% (15 support) |
| unweighted_60_matched | val | 328 | 93.29% | 0.400 | 6.7% (15 support) |
| weighted_60_mid | val | 328 | 91.77% | 0.455 | 46.7% (15 support) |
| original_128_mid | val | 1502 | 88.42% | 0.376 | 6.9% (72 support) |
| extended_128_mid | val | 1502 | 91.34% | 0.376 | 1.4% (72 support) |
| weighted_128_mid | val | 1502 | 88.62% | 0.379 | 8.3% (72 support) |
| unweighted_128_shallow | val | 1501 | 80.28% | 0.338 | 69.7% (409 support) |
| unweighted_128_deep | val | 1517 | 86.95% | 0.358 | N/A (0 support) |
| holdout_a_to_b_60 | test | 628 | 0.32% | 0.003 | 0.0% (277 support) |
| holdout_b_to_a_60 | test | 1011 | 1.68% | 0.021 | N/A (0 support) |

The CSV `handoff_public/run_metrics.csv` contains full support and F1/recall for each class,
all run directory paths, and best epochs where available. Exact run configs, training
histories, confusion matrices and checkpoint files remain under each Kingston run directory.

## Main findings

1. Extended 128x128 training (49 epochs, best epoch 19) gave ~91.34% validation
   accuracy, but TDPAND recall fell to 1/72 (~1.39%). More epochs are not a fix.
2. **Matched 60x300 comparison** (both max 60 epochs, patience 25):
   - Unweighted: accuracy 93.29%, fixed-five-class macro-F1 0.3996, TDPAND 1/15.
   - Balanced loss: accuracy 91.77%, macro-F1 0.4551, TDPAND 7/15.
   - Improvement is small-sample and correlated-tile evidence, NOT a statistical claim.
3. Weighted 128x128 did not materially fix TDPAND (6/72 recall).
4. Shallow/deep blocked validations had different class supports; their raw
   accuracies are not direct like-for-like comparisons.
5. Held-out-hole stress tests collapsed (A→B 2/628 correct; B→A 17/1011).
   This is primarily due to missing target classes in training; it does not alone
   establish a conventional domain-shift generalization error.
6. The training script selects checkpoints using unweighted validation loss even
   with balanced training weights, potentially prioritizing majority-class performance.
7. Prediction montages were generated for extended 128x128 and weighted 60x300.
   Their ZIP can be inspected on the existing review branch (publication permission
   for imagery should be confirmed). Gallery generation does not prove image-depth
   accuracy or geological label correctness.

## Pending external verification, NOT yet established errors

- Confirm precise journal meanings and photographic distinctions of TPAND vs TDPAND.
- Check recurrent disagreements around ~85–90 m and ~117–120 m in hole DH26UVD_11
  against ORIGINAL core boxes, journal contacts and row depth mapping.
- Validate row orientation and registration of contact intervals. Track uncertain
  contact boundaries and any corrections separately from original annotations.
- Determine if additional INDEPENDENT holes containing overlapping classes exist.
- Confirm number of independent lithological intervals per class, not tile counts.
- Review masks/black backgrounds, fragmentation, lighting and potential shortcuts.
- Check publication authorization for derived imagery in the PUBLIC repo.

## Decision and next work

**STOP additional unstructured CNN training for now.** The current diagnostic
phase is completed. Await geological verification and possible extra boreholes.
Afterwards: version only approved label/depth corrections, select an evaluation
protocol with class overlap and independent intervals/holes, and then consider
VGG16, SOM and Mean Teacher. These methods have NOT been run in this phase.

## How to resume without repeating work

1. Open this file and `handoff_public/run_metrics.csv`.
2. Review `docs/LAB_RUNBOOK.md`, `docs/REPRODUCTION_AUDIT.md`, `NOTICE.md`,
   `src/corelog/` and `configs/experiments/tiles_128.yaml`.
3. Check `scripts/corelog_review/finish_bassgold_diagnostics.py` for exact
   experiment override commands; `scripts/corelog_review/make_prediction_montages.py`
   for matching saved prediction CSVs to images by `slice_id`.
4. For visuals inspect the existing `review_artifacts/bassgold_visual_review.zip`.
5. On the lab computer verify the Kingston SSD is mounted before referring to
   the run directories above. Do not attempt expensive re-training by default.
6. If new geological evidence arrives, record source and approved corrections,
   and rerun only what needs changing. Never silently relabel to match a CNN.

## GitHub upload audit

The ZIP was already uploaded to the *existing* public repository on the review
branch. It is not on `main`. This handoff adds NO new geological imagery,
raw manifests, tile files, model weights or private evaluation archives.

Local ZIP CRC: PASS. Git blob equals local ZIP: True.
