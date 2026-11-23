#!/usr/bin/env python3
"""Publish a curated, code-and-metrics-only Claude handoff to the EXISTING
public ayazhn04/drillcore-logging repository, review/corelog-contact-sheets branch.

No new Git repository or branch. No raw images, tiles, masks, predictions.csv,
logs/manifests, model weights, private diagnostic ZIPs or ignored lab config
are staged by this script. Existing uploaded visualization ZIP is verified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

HOME = Path.home()
REPO = HOME / 'code/drillcore-logging'
WORK = Path('/media/ra/KINGSTON/drillcore-logging-work')
ROOT60 = WORK / 'outputs/bassgold_2026-11'
ROOT128 = WORK / 'outputs/bassgold_2026-11_t128'
BRANCH = 'review/corelog-contact-sheets'
REMOTE = 'ayazhn04/drillcore-logging'
ZIP = WORK / 'diagnostics/bassgold_visual_review.zip'
ZIP_GIT_PATH = 'review_artifacts/bassgold_visual_review.zip'

# Preserved run names and evaluation types (no training is performed).
RUNS = [
    ('original_60_mid', ROOT60, 'pilot_fixed_20261123_143119', 'val'),
    ('unweighted_60_matched', ROOT60, 'diag_unweighted_60_matched_20261123_153158', 'val'),
    ('weighted_60_mid', ROOT60, 'diag_weighted_60_mid_20261123_151926', 'val'),
    ('original_128_mid', ROOT128, 'pilot_t128_20261123_143600', 'val'),
    ('extended_128_mid', ROOT128, 'pilot_t128_extended_20261123_145424', 'val'),
    ('weighted_128_mid', ROOT128, 'diag_weighted_128_mid_20261123_151926', 'val'),
    ('unweighted_128_shallow', ROOT128, 'diag_unweighted_128_shallow_20261123_151926', 'val'),
    ('unweighted_128_deep', ROOT128, 'diag_unweighted_128_deep_20261123_151926', 'val'),
    ('holdout_a_to_b_60', ROOT60, 'diag_holdout_a_to_b_60_20261123_151926', 'test'),
    ('holdout_b_to_a_60', ROOT60, 'diag_holdout_b_to_a_60_20261123_151926', 'test'),
]


def cmd(*args, cwd=REPO, capture=True):
    p = subprocess.run(list(args), cwd=cwd, text=True, capture_output=capture)
    if p.returncode:
        raise RuntimeError(f"Command failed: {' '.join(args)}\n{p.stderr if capture else ''}")
    return p.stdout.strip() if capture else ''


def git_preflight(push):
    if not (REPO / '.git').exists():
        raise RuntimeError(f'Git repository not found: {REPO}')
    remote = cmd('git', 'remote', 'get-url', 'origin')
    allowed = ('github.com/ayazhn04/drillcore-logging', 'github.com:ayazhn04/drillcore-logging')
    if not any(x in remote for x in allowed):
        raise RuntimeError(f'Unexpected origin: {remote}')
    if cmd('git', 'diff', '--cached', '--name-only'):
        raise RuntimeError('Existing staged changes detected. Commit/unstage them first.')
    branch = cmd('git', 'branch', '--show-current')
    if branch != BRANCH:
        if cmd('git', 'status', '--porcelain'):
            raise RuntimeError(f'Working tree is dirty on {branch}. Switch to {BRANCH} manually after saving work.')
        cmd('git', 'switch', BRANCH)
    if push:
        print('Syncing existing branch, without force pushes or branch creation...')
        cmd('git', 'pull', '--ff-only', 'origin', BRANCH, capture=False)
    print('Repo:', REPO)
    print('Branch:', cmd('git', 'branch', '--show-current'))
    print('Remote:', remote)


def verify_zip():
    result = {'path': str(ZIP), 'present': ZIP.is_file(), 'repository_path': ZIP_GIT_PATH}
    if not ZIP.is_file():
        print('WARNING: Local visualization ZIP not found, cannot inspect archive or compare Git blob.')
        return result
    names = []
    with ZipFile(ZIP) as z:
        bad = z.testzip()  # verifies every archived member's CRC
        if bad:
            raise RuntimeError(f'ZIP CRC check failed for {bad}')
        names = [x.filename for x in z.infolist() if not x.is_dir()]
        for prefix in ('128x128/', '60x300_weighted/'):
            must = {'00_correct_vs_wrong.png', '03_true_TDPAND_pred_TPAND.png',
                    '04_true_TPAND_pred_TDPAND.png', '07_predictions_by_depth.png'}
            missing = {prefix + x for x in must} - set(names)
            if missing:
                raise RuntimeError(f'ZIP missing expected files: {sorted(missing)}')
        result['member_count'] = len(names)
        result['groups'] = dict(Counter(x.split('/')[0] for x in names))
        result['crc_check'] = 'PASS'
        result['size_bytes'] = ZIP.stat().st_size
    result['sha256'] = hashlib.sha256(ZIP.read_bytes()).hexdigest()
    try:
        public_blob = cmd('git', 'rev-parse', f'HEAD:{ZIP_GIT_PATH}')
        local_blob = cmd('git', 'hash-object', str(ZIP))
        result['git_blob_matches_checkout'] = local_blob == public_blob
        result['git_blob_sha'] = public_blob
        if local_blob != public_blob:
            raise RuntimeError('Local ZIP differs from the committed GitHub ZIP. Stop and inspect before publishing.')
    except RuntimeError as e:
        if 'Local ZIP differs' in str(e):
            raise
        result['git_blob_matches_checkout'] = 'unverified'
    print('ZIP integrity:', result['crc_check'], 'members:', result['member_count'],
          'Git blob matches:', result['git_blob_matches_checkout'])
    return result


def collect_metrics(out: Path):
    rows = []
    missing = []
    for name, root, basename, split in RUNS:
        run = root / 'runs' / basename
        source = run / f'eval_{split}/metrics.json'
        if not source.exists():
            missing.append(str(source))
            continue
        metric = json.loads(source.read_text(encoding='utf-8'))
        overall = metric['overall']
        pc = overall.get('per_class', {})
        row = {'run': name, 'tile_geometry': '128x128' if root == ROOT128 else '60x300',
               'evaluation_split': split, 'run_dir': str(run),
               'n': overall['n'], 'accuracy': overall['accuracy'],
               'macro_f1_fixed_five': overall['macro_f1']}
        supported = [d.get('f1', 0) for d in pc.values() if d.get('support', 0) > 0]
        row['macro_f1_supported_only'] = sum(supported) / len(supported) if supported else ''
        for cls in ('CLY','RS','TDPAND','TPAND','TRCRHYP'):
            data = pc.get('lith_' + cls, {})
            row[cls + '_support'] = data.get('support', '')
            row[cls + '_recall'] = data.get('recall', '')
            row[cls + '_f1'] = data.get('f1', '')
        summary = run / 'train_summary.json'
        if summary.exists():
            try:
                j = json.loads(summary.read_text(encoding='utf-8'))
                row['best_epoch'] = j.get('best_epoch', '')
            except (json.JSONDecodeError, UnicodeDecodeError):
                row['best_epoch'] = ''
        else:
            row['best_epoch'] = ''
        rows.append(row)
    if not rows:
        raise RuntimeError('No experiment metrics found on Kingston. Is the SSD mounted?')
    with (out / 'run_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / 'missing_sources.txt').write_text('\n'.join(missing) + ('\n' if missing else ''), encoding='utf-8')
    print(f'Found {len(rows)}/{len(RUNS)} evaluated runs; CSV created.')
    if missing:
        print('WARNING: missing runs:', len(missing))
    return rows


def handoff_text(rows, check):
    display = []
    for r in rows:
        val = r['TDPAND_recall']
        valstr = f'{float(val):.1%}' if val != '' and r['TDPAND_support'] else 'N/A'
        display.append(f"| {r['run']} | {r['evaluation_split']} | {r['n']} | {r['accuracy']:.2%} | "
                       f"{r['macro_f1_fixed_five']:.3f} | {valstr} ({r['TDPAND_support']} support) |")
    md = '''# CLAUDE HANDOFF — BASSGOLD drill-core lithology classification

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
''' + '\n'.join(display) + '''

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
'''
    md += f'\nLocal ZIP CRC: {check.get("crc_check", "not checked")}. Git blob equals local ZIP: '
    md += str(check.get('git_blob_matches_checkout', 'not checked')) + '.\n'
    return md


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--push', action='store_true', help='Commit and push curated handoff to existing branch')
    args = parser.parse_args()
    git_preflight(args.push)
    audit = verify_zip()
    docs = REPO / 'docs'
    summary_dir = REPO / 'handoff_public'
    scripts = REPO / 'scripts/corelog_review'
    for folder in (docs, summary_dir, scripts):
        folder.mkdir(parents=True, exist_ok=True)
    records = collect_metrics(summary_dir)
    (docs / 'CLAUDE_HANDOFF_2026-11-23.md').write_text(handoff_text(records, audit), encoding='utf-8')
    (summary_dir / 'visual_zip_audit.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    for sname in ('finish_bassgold_diagnostics.py', 'make_prediction_montages.py'):
        src = HOME / 'Downloads' / sname
        if src.is_file():
            shutil.copy2(src, scripts / sname)
            print('Preserved script:', sname)
        else:
            print('WARNING: script not found in ~/Downloads:', sname)
    # Preserve the command used to create this handoff for future Claude sessions.
    this_file = Path(__file__).resolve()
    dest = scripts / 'publish_corelog_handoff.py'
    if this_file != dest:
        shutil.copy2(this_file, dest)
    print('\nCreated documents:')
    for p in (docs / 'CLAUDE_HANDOFF_2026-11-23.md', summary_dir / 'run_metrics.csv',
              summary_dir / 'visual_zip_audit.json'):
        print(' -', p)
    if args.push:
        # Explicit whitelist: NO git add -A, NO datasets or checkpoints.
        cmd('git', 'add', '--', 'docs/CLAUDE_HANDOFF_2026-11-23.md',
            'handoff_public/run_metrics.csv', 'handoff_public/visual_zip_audit.json',
            'handoff_public/missing_sources.txt', 'scripts/corelog_review/', capture=False)
        staged = cmd('git', 'diff', '--cached', '--name-only').splitlines()
        allowed_prefixes = ('docs/CLAUDE_HANDOFF_2026-11-23.md', 'handoff_public/', 'scripts/corelog_review/')
        if any(not path.startswith(allowed_prefixes) for path in staged):
            raise RuntimeError('Unexpected staged files; refusing to commit: ' + repr(staged))
        print('\nFiles staged for publication:', *staged, sep='\n - ')
        if staged:
            cmd('git', 'commit', '-m', 'Document November 2026 CNN diagnostics for future handoff', capture=False)
            cmd('git', 'push', 'origin', BRANCH, capture=False)
            print('PUSHED: https://github.com/' + REMOTE + '/tree/' + BRANCH)
        else:
            print('No changes: handoff already committed.')
    else:
        print('\nDRY RUN. Nothing pushed. To upload now, rerun with --push.')
    print('No new repository or branch was created, and no training was run.')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('ERROR:', e, file=sys.stderr)
        raise SystemExit(1)
