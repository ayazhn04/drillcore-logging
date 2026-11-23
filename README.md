# drillcore-logging

Automatic, depth-indexed geological logging from drill-core photographs. This is a corrected, clean-room reimplementation of the weakly supervised workflow of **Serdoun et al. (2026)**, *Computers & Geosciences* 212, 106156 (see [NOTICE.md](NOTICE.md)).

```
photos + rows.csv ──► tiles (300×60 px, depth-registered) ──► labels from journals (depth overlap)
                                    │                                   │
                                    ├─► Gabor + colour ─► SOM ─► geologists label nodes (optional)
                                    ▼                                   ▼
                     CNN baseline / VGG16 / Mean Teacher  ◄── hole-based train/val/test split
                                    ▼
             evaluation on held-out holes ──► predicted depth log ──► agreement with journal
```

## Quick start

On the lab machine (Ubuntu + NVIDIA GPU), clone and set up the environment:
```bash
git clone <repo-url> drillcore-logging && cd drillcore-logging
```
```bash
bash scripts/setup_ubuntu.sh <CUDA_TAG>
```
`<CUDA_TAG>` is the PyTorch tag for your driver, e.g. `cu128`; see [docs/INSTALL.md](docs/INSTALL.md).

Check the installation:
```bash
source .venv/bin/activate && pytest && corelog smoke-test
```

`pytest` and `smoke-test` use only generated synthetic images. Their numbers are marked `SYNTHETIC` and are not geological results.

Point the pipeline at the RAs' dataset:
```bash
cp configs/lab.example.yaml configs/lab.yaml
```
Edit `configs/lab.yaml` to set `data_root`, `output_root`, classes, `class_map` and holes.

Run the pipeline:
```bash
corelog validate    -c configs/lab.yaml
```
```bash
corelog make-tiles  -c configs/lab.yaml
```
```bash
corelog make-labels -c configs/lab.yaml
```
```bash
corelog train       -c configs/lab.yaml
```

Evaluate a run and produce its depth log:
```bash
corelog evaluate    -c configs/lab.yaml outputs/runs/<run>
```
```bash
corelog predict-log -c configs/lab.yaml outputs/runs/<run> --holes HOLE_A
```

## Documentation
| Document | For |
|---|---|
| [docs/DATASET_MANIFEST.md](docs/DATASET_MANIFEST.md) | **Research assistants:** how to deliver photos and journals |
| [docs/INSTALL.md](docs/INSTALL.md) | Ubuntu GPU setup, laptop setup, WSL appendix, reproducibility |
| [docs/LAB_RUNBOOK.md](docs/LAB_RUNBOOK.md) | **First real-data run:** copy-paste commands for two holes and rock-only deliveries, with an observation sheet |
| [docs/WORKFLOW.md](docs/WORKFLOW.md) | Step-by-step lab procedure: validate, tiles, SOM, training, evaluation, logs |
| [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | Design decisions, deviations from the paper, stage test reports, next steps |
| [docs/REPRODUCTION_AUDIT.md](docs/REPRODUCTION_AUDIT.md) | Audit of the paper and the authors' code, with numerical evidence |

## Repository layout
```
configs/        base.yaml (all parameters), smoke.yaml, lab.example.yaml, lab.two_holes.example.yaml,
                rock_only_synthetic.yaml (tests), experiments/*.yaml
src/corelog/    config, manifest, images, tiling, labels, splits, synthetic,
                features/ (Gabor, colour), som/ (SOM + labelling), data/ (dataset, augmentation),
                models/ (baseline CNN, VGG16), train/ (supervised, mean_teacher),
                evaluate.py, inference.py (depth logs), diagnostics.py (inspect), cli.py
tests/          unit tests per stage + end-to-end smoke test (synthetic data only)
docs/           documentation and RA templates
scripts/        setup_ubuntu.sh
requirements/   pinned dependencies
```
Data, tiles, features, runs and checkpoints are written to `paths.output_root`, outside Git (see `.gitignore`).

## Data policy
No company photographs, raw files (`.CR2`), journals, manifests with company data, credentials, generated tiles or model checkpoints are ever committed. Machine-specific paths live only in the git-ignored `configs/lab.yaml` or in the environment variables `CORELOG_DATA_ROOT` and `CORELOG_OUTPUT_ROOT`.
