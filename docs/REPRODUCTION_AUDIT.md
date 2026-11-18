# Reproduction Audit — Serdoun et al. (2026)

> **Note (repository copy).** This audit was produced before the repository existed. The paths `upstream/Core-photographs`, `paper/` and `audit_tests/` refer to the separate workspace folder next to this repository (`../`), **not** to files in this repo: the authors' code has no license and is deliberately kept out of it. Real borehole IDs have been replaced with HOLE_A–D.

**Paper:** Serdoun, Sur, Milési, Obin, Mercadier (2026). *Automated classification of drill core photographs: A complete workflow for weakly supervised training.* Computers & Geosciences 212, 106156. https://doi.org/10.1016/j.cageo.2026.106156 (CC BY 4.0; full text read from HAL hal-05632256v1)
**Code:** https://github.com/MehdiSerdoun/Core-photographs, cloned read-only into `upstream/Core-photographs` at commit `f7b8b7c` (2025-08-28, the only branch, no tags or releases).
**Audit date:** 2026-11-18
**Scope:** Static audit of the paper against the code, plus (since 2026-11-18, second pass) **numerical tests on synthetic data** of the feature-extraction, labeling and data-handling findings (§12; scripts in `audit_tests/`). The test environment is separate (`.venv-audit`: numpy, scipy, scikit-image, scikit-learn, pandas, MiniSom, tqdm) and contains **no TensorFlow**, so the CNN and Mean Teacher findings remain static findings. The original repository was not modified. No company data was used.

> **Bottom line.** The public notebooks cannot run end to end, and in several places they do not implement what the paper describes. The paper's numbers cannot be reproduced: the dataset was never published, and the code that produced Table 3 was deleted from the repository. That code is still recoverable from git history (§5), and its saved outputs match two Table 3 values exactly. It shows that (a) the "semi-supervised" Mean Teacher used in the paper gets **no gradient from its consistency loss**, (b) VGG16 was **not** resized to 224×224 as the paper states, and (c) the split was 70/10/20, not 70/20/10. The paper's random slice-level split also makes its accuracy figures **optimistic** (§7).

---

## Contents
1. [Repository inventory](#1-repository-inventory)
2. [Dataset availability](#2-dataset-availability)
3. [What the paper describes (reference specification)](#3-what-the-paper-describes)
4. [Notebook-by-notebook audit](#4-notebook-by-notebook-audit)
5. [Evidence from git history (deleted notebooks)](#5-evidence-from-git-history)
6. [Paper ↔ code discrepancy table](#6-paper--code-discrepancy-table)
7. [Data-leakage and evaluation-validity risks](#7-data-leakage-and-evaluation-validity-risks)
8. [Internal inconsistencies within the paper](#8-internal-inconsistencies-within-the-paper)
9. [Environment and licensing](#9-environment-and-licensing)
10. [Reproducibility verdict](#10-reproducibility-verdict)
11. [Prioritized implementation plan](#11-prioritized-implementation-plan)
12. [Numerical verification on synthetic data](#12-numerical-verification-on-synthetic-data)

---

## 1. Repository inventory

| File | Size | Notes |
|---|---|---|
| `Feature_Extraction.ipynb` | 10 cells | No outputs saved. |
| `Labeling.ipynb` | 12 cells | No outputs saved. |
| `Deep_Learning.ipynb` | 16 cells | No outputs saved. |
| `README.md` | — | Says `little_images.npy` is provided. **It is not.** |
| `requirements.txt` | 146 pins | Windows-only packages; `minisom` missing (§9). |
| `.gitattributes` | — | LFS rule for `3_Clustering.ipynb`, a file that no longer exists. |

**Not in the repository at all**, although the paper describes them (§3.1):
- Lens-distortion correction (Santana-Cedrés et al. 2016)
- Core–tray separation (Canny edges, morphological operators, skeletonization)
- 5% lateral crop, vertical concatenation per drill hole, slicing into 300×60 px tiles
- Depth registration of slices (used for Fig. 16)
- Inference on unseen holes (Figs. 15–16), majority voting of four geologists, and the "strict" vs "large" label variants

Every commit is dated 2025-08-28, via "Add files via upload". `3_Clustering.ipynb` (an LFS pointer to a 29.9 MB file) and `5_Semi_supervised.ipynb` were committed and later deleted (§5).

## 2. Dataset availability

| Claim | Source | Reality |
|---|---|---|
| "A dataset used in the associated article is provided to run the code (little_images.npy)" | README | **No `.npy`/`.npz`/`.csv` file exists in any commit** (`git log --all --name-only`). No releases or tags. No LFS objects other than the deleted clustering notebook. |
| "Supporting data is made available in the associated GitHub repository" | Paper, Data availability | False as of the audit date. |
| "The source codes and data sets used in this study are available at github.com/MehdiSerdoun/Core-photographs" | Paper, Code availability | Code is partial and data is absent. |

The photographs belong to Orano Canada Inc. and may never be released. **Conclusion: the paper's numbers cannot be reproduced.** Only the methodology can be reproduced, on other data. Contacting the corresponding author (address in the paper) is the only route to the data.

File names in the code show several distinct data artefacts that the paper does not name:
`little_images.npy` (SST/basin, made with **50% overlap** between slices, see §5), `little_images_BST_300.npy` (basement), `little_images_WAT.npz` (107,922 images: a count that matches neither 95,983, nor 20,312, nor their sum), and label files `Image_labels_{SST,BST}_{Large,strict}.csv`.

## 3. What the paper describes

Reference specification extracted from the paper, used for the comparison in §6.

| Stage | Paper specification |
|---|---|
| Data | 23 NQ drill holes, 1,009 photos (861 basin, 148 basement), 4 boxes per photo |
| Pre-processing | Lens-distortion correction; Canny + morphology for core–tray separation; 5% crop on each side; concatenation per hole (~25,000×60 px); slices of **300×60 px ≈ 18 cm**. Basin: 95,983 slices. Basement: 20,312 slices (20,333 in Fig. 7). |
| Gabor | Grayscale; f = [0.05, 0.09, 0.17, 0.3, 0.55, 1] cycles/px; θ = [0, π/4, π/2, 3π/4]; **magnitude of the complex response \|I∗G\|** (Eq. 5); mean + std → **24 filters × 2 = 48 features** |
| Colour | RGB 8×8×8 = 512-bin 3D histogram, each channel "normalized to 0–1" before computing it; **PCA → 48 PCs** |
| SOM input | 48 Gabor + 48 colour PCs = **n = 96**, standardized |
| SOM | MiniSom, 10×10, 5,000,000 iterations, "h = 0.01; α = 1.25" (lr and sigma); **PCA initialization with PC loadings** |
| Labeling | 4 geologists, majority vote per node, unlabeled nodes become "non-attributed". Basin: 4 classes, 18,355 labeled. Basement: 5 classes, 4,673 labeled. |
| Split | Random; **70% train / 20% validation / 10% test**. Test sizes: 2,663 basin, 935 basement. |
| Baseline CNN | 3 conv blocks (32, 32, 64; 3×3; ReLU; BN; MaxPool) → dropout 0.5 → **one** dense softmax layer; 1,000 epochs; batch 16; early stopping, **patience 20** |
| VGG16 | ImageNet weights; **resized to 224×224**; **conv base frozen**, only the FC layers trained |
| Optimisation | **lr = 1e-5**; batch size "initially 32", increased "by two every 20 epochs" (contradicts batch 16 above) |
| Augmentation | Gaussian noise, horizontal and vertical flips, brightness, contrast |
| Mean Teacher | Student = baseline (or VGG16); **independent noise for student and teacher**; MSE consistency over labeled **and** unlabeled samples; **λ ramps to 10 over 20 epochs**; EMA α = 0.99 during ramp-up, then 0.999 |
| Results | Table 3, accuracy only. Basin: baseline-sup 0.932, VGG-sup 0.884, baseline-MT 0.931, VGG-MT 0.881. Basement: 0.740, 0.675, 0.751, 0.683. |
| External check | Qualitative only: Fig. 15 (Waterbury) and Fig. 16 (one hole, WF-100, visual comparison with the field log) |

## 4. Notebook-by-notebook audit

Severity: 🔴 crash or wrong result · 🟠 diverges from the paper or biases results · 🟡 hygiene or portability.

### 4.1 `Feature_Extraction.ipynb`

**What it does.** It loads an array of equally sized RGB slices. For each slice it computes a 512-bin 3D RGB histogram after per-channel histogram equalization, plus Gabor statistics computed in parallel with joblib. It then concatenates them into `df_unified` (Gabor columns first, then `rgb_0…rgb_511`).

| # | Sev | Cell | Issue |
|---|---|---|---|
| FE-1 | 🔴 | [4], [6], [8] | `data = np.load(...)` is assigned, but every later cell uses `little_images`, giving a **NameError**. History shows the original was `data = np.load('…WAT.npz'); little_images = data['arr_0']`; the second line was lost. |
| FE-2 | 🔴 | [4] | `little_images.npy` does not exist (§2). Hard-coded Windows placeholder path `r'path\to\data\little_images.npy'`. |
| FE-3 | 🔴 | [8] | **Gabor implemented incorrectly.** `np.absolute(gabor_kernel(f, θ))` takes the modulus of the *kernel* before convolution. The modulus of a complex Gabor kernel is just its Gaussian envelope, because \|e^{iφ}\| = 1. With skimage's default σx = σy, that envelope is **isotropic and carries no frequency or orientation**. The 6 θ values per frequency therefore give near-identical features: they differ only through the θ-dependent kernel support size. In practice the "texture" descriptor is a set of Gaussian blurs at 6 scales, repeated 6 times. The paper's Eq. 5 requires \|I ∗ G\|, the modulus *after* convolution with the complex kernel. History shows the author wrote a comment saying to swap the order to "convolve then take the modulus", but that version crashed with a TypeError, and the abs-kernel version was the last one executed (§5). **Numerically confirmed (§12.1):** \|kernel\| is identical across θ (relative difference 0.0), θ-columns correlate at 0.98–1.00, and on seam-free gratings the original features classify orientation at chance (0.30 vs 0.25), while Eq. 5 scores 1.00. The 0.60–0.67 seen on ordinary tiles comes from boundary and wrap-seam artefacts, not orientation selectivity. |
| FE-4 | 🟠 | [8] | Filter bank differs from the paper: frequencies `unique(geomspace(0.05, 1.3, 6).round(2))` = **[0.05, 0.10, 0.18, 0.35, 0.68, 1.3]** (paper: up to 1.0); θ = `arange(0, 3, 0.5)` = **[0, 0.5, …, 2.5] rad**, 6 non-uniform angles (paper: 4 angles at π/4 steps). The result is **72 Gabor features, not 48**. |
| FE-5 | 🟠 | [8] | Frequencies above 0.5 cycles/px are above Nyquist (0.68 and 1.3 in the code; 0.55 and 1.0 in the paper too). They alias, and at f = 1 the complex carrier is constant on the integer grid, so those filters add no texture information. This is a flaw in both the paper and the code. |
| FE-6 | 🟠 | [8] | `ndi.convolve(..., mode='wrap')` on 60-px-high slices. At f = 0.05, σ ≈ 11 px and the kernel is about 69 px wide, which is larger than the image height. The response at low frequencies is dominated by top and bottom wrap-around, where the core edge meets the tray. |
| FE-7 | 🟠 | [8] | Per-image min–max normalization of each response before taking mean and std removes absolute texture contrast, and makes the features sensitive to single outlier pixels such as specular highlights. The paper does not mention it. |
| FE-8 | 🟠 | [6] | `equalize_hist` per channel is **histogram equalization**: it flattens each channel's distribution. The paper says channels are "independently normalized to values from 0 to 1", which describes a min–max stretch. Equalization removes most absolute colour (for example, the red-hematite vs bleached contrast that the basin classes depend on) and keeps mainly inter-channel co-variation. This is a major ambiguity for a colour-defined task. |
| FE-9 | 🟡 | [9] | `df_unified` is never written to disk, but `Labeling.ipynb` reads `image_descriptors.csv`. The export step is missing, and whether to write the index is undocumented, which matters because Labeling slices columns by position. |
| FE-10 | 🟡 | [2] | Unused or dead imports: `scipy.misc` (deprecated; still imports on SciPy 1.18.1 with a warning, removal planned for 2.0, see §12.5), `cv2`, `KMeans`, `AgglomerativeClustering`, `PCA`, `local_binary_pattern`, `skeletonize`, `area_opening`/`area_closing`, `binary_dilation`, plus duplicate imports. These look like leftovers from the missing pre-processing code. |
| FE-11 | 🟡 | [6] | `range=((0,255),…)` with 8 bins gives 31.875-wide bins. Value 255 lands in the last bin (correct), but the uint8 truncation of `equalize_hist(...)*255` is implicit. |

### 4.2 `Labeling.ipynb`

**What it does.** It loads slices and descriptors and subsamples both `[::2]`. It PCA-reduces the colour and Gabor blocks, standardizes, and initializes and trains a 10×10 MiniSom for 5M iterations. It then plots 5 random images per node on a 200×200-inch figure, collects node labels from the user through ipywidgets, and exports per-image labels, with "not_attributed" for unlabeled nodes.

| # | Sev | Cell | Issue |
|---|---|---|---|
| LB-1 | 🔴 | [4] | `np.load('path\to\little_images.npy')` is **not a raw string**, so `\t` becomes a TAB character. The path is a placeholder in any case. |
| LB-2 | 🔴 | [6] | `rgb = rgb_vector`: `rgb_vector` exists only in the Feature_Extraction kernel, giving a **NameError**. It would also not be subsampled `[::2]`, so its length (N) would mismatch `gabor` (N/2). It should be `data.iloc[:, -512:]`. |
| LB-3 | 🔴 | [6] | `PCA(n_components=128)` on the Gabor block, which has **72 columns** (48 per the paper). scikit-learn raises `ValueError: n_components=128 must be between 0 and min(n_samples, n_features)=72`. |
| LB-4 | 🔴 | [6] | Comment: "There are 512 gabor filters so access them from index 512". This is wrong: 512 is the number of **RGB** bins. `data.iloc[:, :-512]` does select the Gabor block by luck, but it also includes an `Unnamed: 0` index column if the CSV was saved with its index (pandas default). |
| LB-5 | 🔴 | [6] | `pca_gabor` is computed and then **never used**. `df_unified = concat([pca_rgb, gabor])` uses the **raw** Gabor block, not the PCs. |
| LB-6 | 🔴 | [6] | **Positional index leaks into the feature space.** `gabor.reset_index(inplace=True)` adds an `index` column (0, 2, 4, …), and so does `pca_rgb.reset_index`. `df_unified.iloc[:, 1:]` drops only the *first* one. The Gabor `index` column, which encodes slice order (that is, hole and depth position), is standardized and fed to the SOM as a feature. Clusters can then partly reflect position along the hole instead of appearance. **Numerically confirmed (§12.3).** With the pandas-default CSV, `Unnamed: 0` leaks as well (202 inputs). In a controlled experiment, the leak produced a small but significant depth imprint on SOM nodes (z = 4.1–11.1 above a shuffled baseline) without changing appearance clustering. |
| LB-7 | 🟠 | [6] | Resulting dimensionality: 128 colour PCs + 72 raw Gabor + 1 index = **201 features**, versus the paper's **96** (48 + 48). Colour gets 128 PCs, which over-weights colour relative to texture and contradicts the paper's "same number of descriptors" rationale. |
| LB-8 | 🔴 | [8] | `som._weights = init_weights_sst`: **NameError**; it should be `init_weights`. |
| LB-9 | 🟠 | [8] | The initialization does not match the paper. The paper says neurons are initialized with **PC loadings**. The code uses the **PCA scores of the first 100 samples** (`pca_components[(i*10+j) % N]`), which are coordinates in the *rotated* PCA space, while the SOM is trained in the *unrotated* standardized space. The init therefore lies in the wrong coordinate frame. The paper's own description is also infeasible (100 neurons ↔ "m first PCs" with only 96 dimensions). MiniSom's `pca_weights_init` (a grid spanned by the first 2 PCs) is the standard and reproducible choice. Setting the private `_weights` attribute also bypasses MiniSom's API. |
| LB-10 | 🟠 | [8] | `som.train(scaled_data, 5_000_000)` uses `random_order=False`, so it cycles through the data in file order (depth order). The paper describes drawing "the median sample", which does not match MiniSom's algorithm. The MiniSom version is not pinned, and its default decay functions have changed between releases, so the "same hyperparameters" can give a different map. |
| LB-11 | 🟠 | [4] | `[::2]` subsampling, which the paper does not mention. History explains it: the SST array was built with **50% slice overlap**, while labeling assumed none. Labels are therefore indexed against the subsampled array, and any downstream notebook must apply the identical subsampling (Deep_Learning does not: DL-2). For basement data `[::2]` must *not* be applied, yet it is hard-coded here. |
| LB-12 | 🟡 | [8] | `figsize=(200, 200)` inches: about 20,000×20,000 px at 100 dpi, several GB of RAM, impractical. `random.sample` is unseeded, so the displayed exemplars change on every run. The comment says "Select 10" but the code takes 5. `display` is not imported (works only inside IPython). |
| LB-13 | 🟠 | [8]–[11] | Labels exist only in an in-memory dict filled by widgets. Nothing is persisted: no SOM weights, BMU assignments, node→label map, or labeler ID. The labeling is not auditable or reproducible, and multi-geologist majority voting (as in the paper) is not supported. |
| LB-14 | 🟡 | [11] | Exports to `Image_labels_SST_Large.csv`, but Deep_Learning reads `Image_labels_BST_Large.csv`: different dataset tags (SST = sandstone/basin, BST = basement). |

### 4.3 `Deep_Learning.ipynb`

**What it does.** It loads images and labels, drops `not_attributed`, splits 70/10/20, and trains three models: (1) a baseline CNN, (2) VGG16 fine-tuning, and (3) a Mean Teacher with the baseline as student. It reports a confusion matrix and a classification report for whichever model was trained last.

| # | Sev | Cell | Issue |
|---|---|---|---|
| DL-1 | 🔴 | [4] | Hard-coded Windows paths: `'D:\Algo_Photos\little_images_BST_300.npy'` (not raw; `\A` and `\l` raise SyntaxWarnings) and `r'D:\Article_Photos\Image_labels_BST_Large.csv'`. |
| DL-2 | 🔴 | [4] | **Index alignment with labels is not guaranteed.** Labeling exports labels for `little_images[::2]`, while this notebook loads the full array. It works only if the basement array (which has no overlap) was used throughout. For SST data, labels would be attached to the wrong images. **Numerically confirmed (§12.4):** only 1 of 40 labels was paired with the image it was made for, and the label matched the true class of the paired image for 28% of tiles. |
| DL-3 | 🟠 | [6] | `unique_labels = list(set(...))`: string hashing is randomized per process, so the **class ↔ index mapping changes between runs**. Saved models and confusion matrices are then not comparable across sessions. The Mean Teacher section uses `LabelEncoder` (sorted order), a *different* mapping in the same notebook. |
| DL-4 | 🟠 | [6] | Split is **70/10/20** (test 0.2, then val 0.125×0.8) versus the paper's **70/20/10**. It is not stratified (diatexite has 196 samples) and is random at slice level (see §7). |
| DL-5 | 🟠 | [8] | Baseline differs from the paper (Fig. 8: 3 conv blocks → dropout → softmax). The code adds **two Dense(128) + Dropout + BN blocks**. Other differences: lr **1e-3** with AMSGrad (paper: 1e-5); early-stopping patience **50** (paper: 20); fixed batch 16 (paper: growing batch); no flip, brightness or contrast augmentation. |
| DL-6 | 🟠 | [8], [10] | Inputs are **raw uint8 0–255**, with no rescaling. `GaussianNoise(0.05)` on that scale is effectively no augmentation. The Mean Teacher section divides by 255, so models in the same notebook see differently scaled inputs. |
| DL-7 | 🔴 | [10] | `Image` (PIL) is not imported, giving a **NameError** in `resize_images`. The resize is also a no-op (60×300 → 60×300), whereas the paper says 224×224. |
| DL-8 | 🔴 | [10] | **Test set used as validation**: `validation_data=(X_test_resized, y_test)` with `EarlyStopping(restore_best_weights=True)`. The test set therefore picks the stopping epoch and the weights, which is test leakage. `X_val` is ignored. |
| DL-9 | 🟠 | [10] | VGG16 conv base is **not frozen**: all ~14.7M conv parameters are fine-tuned. The paper says only the FC layers are trained, and the deleted notebook did freeze them. The head is Dense(256)–Dense(256) (paper: VGG's FC layers). No `vgg16.preprocess_input` (ImageNet BGR mean subtraction). |
| DL-10 | 🔴 | [10] | `num_classes = 4` is hard-coded. The basement set has 5 classes, giving a shape-mismatch error. It should be `len(unique_labels)`. |
| DL-11 | 🟠 | [12] | The metrics cell evaluates whichever `model` was defined last (VGG16 overwrites the baseline), on `X_test` rather than `X_test_resized`. Baseline metrics are never reported, apart from a single accuracy print. |
| DL-12 | 🔴 | [14] | **Validation data trains the Mean Teacher.** `labeled_gen` is built from *all* `labeled_images`, and `train_test_split(...)` into X_train/X_val happens *afterwards*. Every validation image is therefore in the labeled training stream. |
| DL-13 | 🔴 | [14]–[15] | **The Mean Teacher has no test evaluation.** It uses its own 80/20 split, not the 70/10/20 split above, so supervised and semi-supervised results are not computed on the same held-out set and cannot be compared. Only (contaminated) validation accuracy is printed. |
| DL-14 | 🟠 | [15] | No input perturbation. Student and teacher receive *identical* inputs, and `create_base_model` has no `GaussianNoise`, so the only source of divergence is dropout (student training=True, teacher training=False). The paper (Fig. 9) applies independent noise η and η′. Consistency is computed on unlabeled data only (paper: N_l + N_u). |
| DL-15 | 🟠 | [15] | Hyperparameters differ from the paper: λ_max = **100**, ramp-up **80** epochs (paper: 10 and 20); EMA fixed at **0.999** (paper: 0.99 → 0.999). `sigmoid_rampdown` is defined but unused. |
| DL-16 | 🟠 | [15] | `create_base_model` is described as the "same architecture as above" but differs: no noise layer, and a different Dense/Dropout/BN order. |
| DL-17 | 🔴 | [15] | `create_vgg16_mean_teacher` uses `Input` and `Model`, which are never imported (NameError if called). It is never called, so the Table 3 "VGG16 semi-supervised" rows have no code path. Its comments are in French. |
| DL-18 | 🟡 | [14]–[15] | The generators do not shuffle: batches arrive in file (depth) order, highly non-i.i.d., and BatchNorm statistics follow that order. The classification loss is a per-sample vector added to a scalar consistency term, so it is implicitly summed and the loss scale depends on batch size. The comment "classification loss on unlabeled data" is wrong (it is computed on labeled data). Evaluation uses the student, not the teacher. |
| DL-19 | 🟡 | [2] | Unused imports: `TSNE`, `pydot`, `ReduceLROnPlateau`, `OffsetImage`. `keras.applications` and `tensorflow.keras` imports are mixed (works in TF 2.20 / Keras 3, but fragile). In Keras 3, passing `input_shape` to a non-first layer is ignored with a warning. |
| DL-20 | 🟡 | all | No seeds set (`tf`, `numpy`, `random`). Single run, no variance estimate. |

## 5. Evidence from git history

These are the most informative findings of the audit. The notebooks that actually produced the paper's numbers were committed and then deleted.

**`5_Semi_supervised.ipynb`** (deleted in `2fbf2f0`, run under Python 3.11.7) contains saved outputs that **match Table 3**:
- VGG16 + "Mean Teacher", basement: test accuracy **0.6834** (Table 3: 0.683). Test support **935** (paper: 935); diatexite support **40** (paper: 40).
- VGG16 fully supervised, sandstone: test accuracy **0.8840** (Table 3: 0.884).

What that code actually did, compared with the paper:

| Aspect | Deleted notebook (produced Table 3) | Paper says |
|---|---|---|
| Split | `test_size=0.2`, then `0.125` → **70/10/20**; one split shared across models (good) | 70/20/10 |
| VGG16 input | `input_shape = X_train.shape[1:]` = **(60, 300, 3)**, no resize | resized to 224×224 |
| VGG16 base | frozen ✔ | frozen |
| Augmentation | GaussianNoise, RandomBrightness(0.2), RandomContrast(0.2), RandomFlip(h+v) ✔ | same |
| Batch schedule | 16, **doubling every 50 epochs** (max 512) | starts at 32, "increased by two every 20 epochs" |
| Patience / lr | 20 / 1e-5 ✔ | 20 / 1e-5 |
| SST data | `little_images.npy[::2]`. Author comment (French): the array "was made with a 50% overlap while labeling was done without overlap" | not mentioned |
| Label files | `Image_labels_{SST,BST}_strict.csv` | not mentioned ("strict" vs "Large" variants) |
| Unlabeled batch | 25% resampled from **X_train** + 75% not_attributed | unlabeled data only |

**🔴 Critical: the Mean Teacher in the paper did not apply consistency regularization.** In the deleted `train_step`, the consistency term is computed *outside* the `GradientTape` with `teacher_model.predict(...)` and `student_model.predict(...)`, which return NumPy arrays, and passed through a stateful Keras `MeanSquaredError` *metric* that is never reset (a running average over all of training). Inside the tape, `loss += consistency_weight * mse` adds a **constant**, so its gradient with respect to the student weights is **zero**. The "semi-supervised" models were therefore trained on the supervised loss alone. The EMA teacher was updated but never used for evaluation, so it had no influence on the reported results. The reported MT gains (basement 0.740 → 0.751; basin 0.932 → 0.931) are run-to-run noise, not an effect of Mean Teacher. The large spikes in the saved training loss (0.82 → 2.82 → 0.82) come from this added constant. The public `Deep_Learning.ipynb` rewrite *does* compute consistency inside the tape (correctly differentiable), but it has the other defects listed in §4.3.

**`Feature_Extraction.ipynb`, first version (`733b25b`)** saved outputs show:
- `rgb_vector` shape **(107922, 512)** and a final `df_unified` of **107,922 × 584**, which means **72 Gabor features** (6×6×2), not 48. The executed feature set therefore did not match the paper's n = 96.
- A cell implementing the *correct* order (complex convolution, then `np.absolute`) using `geomspace(0.05, 1, 6)` and θ = `arange(0, 1.25, 0.25)` crashed with `TypeError`. The abs-kernel cell was executed after it (execution count 23 > 22) and produced the stored features.
- The original load was `np.load('D:\Collecte_Data\little_images_WAT.npz')['arr_0']` (Waterfound).

**`3_Clustering.ipynb`** survives only as an LFS pointer (29.9 MB, oid `c2a91d5b…`). Its content may still be fetchable from GitHub LFS. That was not attempted, as it is a large download; it would show the actual SOM run.

## 6. Paper ↔ code discrepancy table

| Item | Paper | Public code | Deleted code (produced Table 3) |
|---|---|---|---|
| Pre-processing (distortion, tray separation, slicing) | described | **absent** | absent |
| Gabor operation | \|I ∗ G\| | **I ∗ \|G\|** (orientation-blind) | same as public |
| Gabor f | 0.05…1.0 (6) | 0.05…1.3 (6) | 0.05…1.3 (executed) |
| Gabor θ | 0, π/4, π/2, 3π/4 (4) | 0…2.5 rad step 0.5 (6) | 6 (executed) |
| Gabor features | 48 | 72 | 72 |
| Colour pre-processing | "normalized 0–1" | histogram equalization | equalization |
| Colour PCs | 48 | 128 | — |
| Gabor PCA | none | computed with 128 PCs (crashes), then unused | — |
| SOM input dim | 96 | 201 (incl. leaked index) | — |
| SOM init | PC loadings | PCA scores of 100 samples (undefined variable) | — |
| SOM σ / lr / iters | 1.25 / 0.01 / 5M | same ✔ | — |
| Sub-sampling | — | `[::2]` | `[::2]` on SST only |
| Split train/val/test | 70/20/10 | 70/10/20 (MT: 80/20, contaminated) | 70/10/20 |
| Baseline head | conv×3 → dropout → softmax | + 2×Dense(128) | + 2×Dense(128) |
| Baseline lr | 1e-5 | 1e-3 AMSGrad | 1e-5 |
| Patience | 20 | 50 | 20 |
| Batch | 16, or 32 growing ×2 / 20 ep | 16 fixed | 16 doubling / 50 ep |
| Augmentation | noise, flips, brightness, contrast | noise only (on 0–255 scale) | all four ✔ |
| VGG input | 224×224 | 60×300 | 60×300 |
| VGG base | frozen | **trainable** | frozen ✔ |
| VGG validation | val set | **test set** | val set ✔ |
| MT λ / ramp-up | 10 / 20 ep | 100 / 80 ep | 1 (sigmoid capped at 1) / 20 ep |
| MT EMA | 0.99 → 0.999 | 0.999 | 0.99 → 0.999 ✔ |
| MT noise η, η′ | independent | none | augmentation layers (both) |
| MT consistency gradient | yes | yes | **no (constant)** |
| MT VGG16 variant | reported | not runnable | used |
| Metrics | accuracy + per-class accuracy | accuracy + classification report | + classification report |

Note that the deleted MT code used `sigmoid_rampup(...)`, which is capped at 1, so the effective λ_max was **1**, not 10. This is moot, because the term had no gradient anyway.

## 7. Data-leakage and evaluation-validity risks

Ordered by expected impact on the reported accuracy.

1. **Spatially autocorrelated random split (paper and code).** Neighbouring 18-cm slices from the same core run, box, photograph (same lighting and camera) and drill hole are scattered across train, validation and test. A model can score well by recognizing photo-level appearance (illumination, white balance, tray colour) instead of geology. The paper's own Fig. 16 extrapolation to an unseen hole is only qualitative. **A held-out drill hole (group split) is required for an honest estimate.**
2. **Overlapping slices.** The SST array was built with **50% overlap**. If `[::2]` is ever omitted or misaligned, half of each test image's pixels appear in a training image. Even with `[::2]`, consecutive slices are directly contiguous.
3. **Circular labels (weak supervision).** Test labels are SOM-node labels derived from handcrafted features computed on *all* images, including the test images. Test accuracy therefore measures how well a CNN reproduces the SOM partition, **not agreement with geologists' logs**. Only images in "unambiguous" nodes are evaluated, and ambiguous ones are excluded (about 81% of basin and 77% of basement slices are unlabeled). That selection makes the test set easier than the deployment distribution.
4. **Test set used for early stopping and weight restoration** (public VGG16 cell, DL-8).
5. **Validation set inside the Mean Teacher's training stream** (public notebook, DL-12). The deleted notebook also drew 25% of each "unlabeled" batch from X_train.
6. **Model selection on test.** The "best model" per subset is chosen from 8 configurations by test accuracy, with no separate selection set and no repeated runs or confidence intervals.
7. **Positional index used as a SOM feature** (LB-6). Clusters can encode slice order, and hence hole and depth.
8. **Unsupervised preprocessing fit on all data** (StandardScaler and PCA before any split). This is a mild, transductive leak, acceptable for a labeling tool but it should be stated.
9. **Transductive unlabeled pool.** Mean Teacher's unlabeled images include images from the test holes and photographs. To evaluate honestly, exclude held-out holes from the unlabeled pool.
10. **Metric choice.** Accuracy only, on imbalanced classes. The easy "non-core" class (about 96–97% accuracy) inflates overall accuracy. Macro-F1, per-class recall and results with non-core excluded should be reported.

## 8. Internal inconsistencies within the paper

- Split stated as 70/20/10, but the reported test sizes match **20%** test for the basement (4,673 × 0.2 ≈ 935; diatexite 196 × 0.2 ≈ 40). The basin test count of **2,663** matches neither 10% (1,836) nor 20% (3,671) of the 18,355 labeled basin slices. A different label set ("strict" vs "Large") was probably used without being stated.
- Basement slice count: 20,312 (text) vs 20,333 (Fig. 7 caption).
- "Four classes in both subsets" (§3.4), but the basement has five classes (Table 1).
- Batch size: "16" (§3.3.1) vs "initially set to 32 and increased… for all models".
- VGG input resized to 224×224, but the code that produced Table 3 used 60×300.
- SOM init with "m first principal components" is impossible for m = 100 > n = 96, and does not match MiniSom's `pca_weights_init`.
- Fig. 6/7 captions give "h = 0.01; α = 1.25", where α usually denotes the learning rate. In the code, sigma = 1.25 and lr = 0.01.
- Gabor: equation (5) magnitude-after-convolution vs the code's magnitude-before; 48 vs 72 features.
- "Pegmatite… very high partial melting" conflates a rock type with an anatectic degree. Geologically debatable, but outside this audit's scope.
- Data and code availability statements are not met (§2).

## 9. Environment and licensing

- `requirements.txt` pins **`pywin32==311`** and **`pywinpty==3.0.0`**, which are Windows-only, so `pip install -r` **fails on macOS and Linux**.
- **`minisom` is missing** from the requirements, although Labeling imports it.
- Python: the paper says 3.11 and the deleted notebook ran on 3.11.7, but the public notebook metadata says **3.13.7**. Pins (NumPy 2.2.6, SciPy 1.16.1, TF 2.20, Keras 3.11) need Python ≥ 3.11.
- The deleted MT code uses the Keras 2 API (`model.compiled_loss`), which is incompatible with the pinned Keras 3.
- **No LICENSE file.** By default, all rights are reserved. Copying this code into a company pipeline is legally unclear. Options: a clean-room reimplementation from the paper's description (the paper is CC BY), or asking the author to add a license.

## 10. Reproducibility verdict

| Component | Status |
|---|---|
| Numerical results (Table 3, Figs. 13–16) | ❌ **Not reproducible.** Data unavailable; generating code deleted; MT code defective. |
| Pre-processing (distortion, tray separation, slicing) | ❌ Not provided; must be reimplemented. Thesis Appendix A (Serdoun 2025, theses.fr/2025LORR0037) has details. |
| Colour histogram features | ⚠️ Reproducible after small fixes; choose between equalization and min–max normalization. |
| Gabor features | ⚠️ Code is wrong; reimplement following Eq. 5 with Nyquist-valid frequencies. |
| SOM clustering + labeling UI | ⚠️ Several crash bugs and an index leak; concept is reproducible with MiniSom. |
| Baseline CNN | ✅ Architecture is reproducible (after normalization and augmentation fixes). |
| VGG16 transfer | ⚠️ Fixable; follow the paper (freeze, preprocess_input; resize optional). |
| Mean Teacher | ⚠️ The public version is differentiable but has leakage and missing noise. The published results did not use a working MT. |
| Honest evaluation | ❌ Not present anywhere. Needs grouped (hole-level) splits and external validation against logs. |

What we *can* claim later: "we reimplemented the methodology and evaluated it on our data". What we *cannot* claim: "we reproduced Serdoun et al.'s results".

## 11. Prioritized implementation plan

Proposed only; nothing is implemented yet. The plan is designed so the pipeline is ready when the four boreholes (HOLE_A, HOLE_B, HOLE_C, HOLE_D) and journals arrive.

**P0 — Decisions needed before coding**
1. **Label source.** Our RAs are preparing geological journals (depth-interval logs). This gives a second, *stronger* labeling route than the paper's: depth-registered log labels. Proposal: support both, (a) log-derived labels, used for evaluation as ground truth, and (b) SOM cluster labeling, used to label at scale. Use (a) to measure how good (b) is, which the paper never did.
2. **Class scheme** for gold-hosted lithologies, agreed with the geologists. It should include a `non_core`/QC class.
3. **License.** Treat the upstream code as reference only; write a clean-room implementation from the paper.
4. **Photo metadata.** Confirm the depth-from/to convention in filenames, the number of boxes per photo, the presence of a colour chart or scale, and the camera and lighting setup.

**P1 — Data contract and pre-processing** (the paper's missing step i)
5. Define a per-slice manifest (Parquet/CSV): `hole_id, photo_id, box, row, depth_from, depth_to, x0, x1, tile_path, overlap_flag, qc_flags`. Every downstream step keys on `slice_id`, never on array position. This removes DL-2, LB-6 and LB-11 by design.
6. Tray and row segmentation: start with classical Canny + morphology as in the paper; keep SAM as an upgrade path, as the paper suggests. Add optional lens-distortion correction, and colour normalization if a chart is present.
7. Tiling at 300×60 px (configurable) **without overlap** for labeling and evaluation, and optional overlap for inference only. Map depth linearly within each core row, as in Fig. 16.
8. Journal ingestion: parse the RA logs into intervals and assign each slice a label by majority overlap, with a boundary-uncertainty flag for slices that straddle contacts.

**P2 — Corrected feature extraction and SOM labeling**
9. Gabor per Eq. 5 (complex convolution, then modulus), f ≤ 0.5 cycles/px, 4 orientations over [0, π), `mode='reflect'`. Write a unit test that orientation changes the response on a synthetic striped image (this guards against FE-3).
10. Colour: configurable `none | minmax | equalize`, plus optional HSV or Lab. Compare variants by SOM purity against log labels.
11. Standardize, then PCA with `n_components ≤ n_features` checks. Column selection by name, never position.
12. SOM: MiniSom pinned, `pca_weights_init`, seeded, `random_order=True`. Persist weights, BMUs and the node→label map (JSON) with the labeler ID, and support majority voting across several labelers.
13. A labeling UI that renders per-node thumbnails at sane sizes (paged), with labels saved to disk.

**P3 — Training with leakage-safe evaluation**
14. **Grouped splits by drill hole.** With 4 holes: leave-one-hole-out (4 folds), or 2 train / 1 val / 1 test, rotated. Never random slice-level splits. Also report a random-split result side by side, to quantify the optimism.
15. Baseline CNN per Fig. 8 (plus the head variant as an ablation), inputs scaled to [0, 1], augmentation (noise, flips, brightness, contrast) inside the training pipeline only. Stable label encoding saved with the model.
16. VGG16 (or a modern alternative such as ResNet/EfficientNet), frozen base with `preprocess_input`, then optional fine-tuning; validation strictly on the val fold.
17. Mean Teacher implemented correctly: consistency inside the tape, independent augmentations for student and teacher, consistency over labeled + unlabeled samples, λ and EMA schedules as in the paper, unlabeled pool **excluding test holes**, and evaluation of both student and teacher. Include a unit test that the consistency gradient is non-zero.
18. Seeds, config files (YAML), run logging, and ≥ 3 seeds per configuration.

**P4 — Metrics and reporting**
19. Macro-F1, per-class recall, a confusion matrix, results with and without `non_core`, and bootstrap confidence intervals by hole.
20. Depth-profile comparison against the journals (the quantitative version of Fig. 16): interval-level agreement, contact-position error, and smoothing-window sensitivity.
21. A reproducibility report: environment lockfile (cross-platform), data manifest hashes, configs.

**P5 — Engineering**
22. Convert the notebooks into a package (`preprocess/`, `features/`, `som/`, `train/`, `evaluate/`) with a CLI. Notebooks only for exploration. Use a cross-platform `pyproject.toml` / lockfile.
23. Synthetic smoke-test data (generated striped and coloured tiles), so the full pipeline can be checked in CI before the real boreholes arrive.

**Optional verification steps before P1** (each needs your OK):
- ~~Install a light environment and numerically confirm FE-3, LB-3, LB-6 on synthetic data.~~ **Done: see §12.** The test scripts can become the regression tests for the P2 reimplementation: the corrected Gabor must pass test 1b, the SOM input must contain no index column, and labels must be keyed by `slice_id`.
- Fetch the `3_Clustering.ipynb` LFS object (29.9 MB) to inspect the actual SOM run.
- Contact the corresponding author about the data, the missing pre-processing code, the license, and the Mean Teacher gradient issue.

---

## 12. Numerical verification on synthetic data

*Added 2026-11-18, second pass.* Scripts, raw results and the environment lock are in `audit_tests/` (see `audit_tests/README.md`).

**Method.** The tests execute the **authors' own cell source**, read from the notebook JSON. Where an earlier bug blocks execution (for example LB-2 before LB-3 can be reached), a minimal patch is applied. Every patch asserts that the original text occurs exactly once, and is recorded under `patches_applied` in the corresponding `audit_tests/results/*.json`. All images and descriptors are synthetic and seeded. Environment: Python 3.12.10, numpy 2.5.5, scipy 1.18.1, scikit-image 0.26.0, scikit-learn 1.9.1, pandas 3.1.1, MiniSom 2.3.6. No TensorFlow.

**Evidence levels used below**
- **Confirmed (executed):** the bug was reproduced by running the original code.
- **Confirmed (math):** a property demonstrated numerically on the kernels or arrays themselves.
- **Supported hypothesis:** an effect measured in a controlled synthetic experiment. Its size on real core photographs is unknown.
- **Static only:** not executable here (TensorFlow or real data needed); it rests on code reading and stored outputs.

### 12.1 Gabor filters (tests 1 and 1b)

Synthetic 60×300 RGB gratings (f₀ = 0.12 cycles/px, random phase, Gaussian noise σ = 0.05), 4 orientations × 15 images. Leave-one-out 1-nearest-neighbour classification of orientation on standardized features (chance = 0.25):

| Variant | Operation | Features | Orientation accuracy | Mean corr. between θ-columns of the same frequency |
|---|---|---|---|---|
| A, original (verbatim) | \|kernel\| ∗ image, code bank, min-max | 72 | **0.60** | 0.995–1.000 |
| B, original order, paper bank | \|kernel\| ∗ image, paper bank, min-max | 48 | **0.67** | 0.984–1.000 |
| C, paper Eq. 5 | \|kernel ∗ image\|, paper bank | 48 | **1.00** | −0.33 to 0.52 |
| D, Eq. 5 + min-max | as C, plus the notebook's normalization | 48 | **1.00** | −0.25 to 0.49 |

Kernel level: for every paper frequency, \|kernel\| at θ = 0, π/4, π/2, 3π/4 is **identical** after centre-cropping to a common support (max relative difference 0.0) and symmetric under transpose. It is an isotropic Gaussian, so θ only changes the support size.

A and B scored above chance. That was not predicted by the static audit, so test 1b tested the explanation (H1): the residual signal comes from boundary artefacts of the non-square, wrapped tile, not from orientation selectivity.

| Controlled condition | Chance | A | B | C (Eq. 5) |
|---|---|---|---|---|
| Seam-free (torus-periodic) gratings, 4 orientations, all with \|f\| ≈ 0.100 | 0.25 | **0.30** | **0.275** | 1.00 |
| Mirror pair 45° vs 135° on ordinary tiles | 0.50 | **0.575** | **0.475** | 1.00 |

**Verdict FE-3: confirmed (math + executed).** The original descriptor has **no orientation selectivity**. Whatever orientation-related variance it shows comes from tile-boundary artefacts that depend on the tile's shape and position, which is not a geological signal. Its 36 (code) or 24 (paper bank) filters reduce to 6 distinct Gaussian blurs. The paper's Eq. 5 implementation separates orientation perfectly on the same images.

Other Gabor checks:
- **FE-4: confirmed (executed).** The verbatim code yields **72** Gabor features; the paper bank yields 48.
- **FE-5: confirmed (math).** On the integer pixel grid, cos(2π·0.55x) equals cos(2π·0.45x) (max difference 1.4e-13), so 0.55 cycles/px aliases. At f = 1, θ = 0, the kernel's imaginary part is about 5e-17 of its real part, and Re k = \|k\|: the carrier is constant and the filter degenerates to a blur. This also affects the paper's own bank.
- **FE-6: confirmed (math).** The kernel at f = 0.05 is 69×69 px, larger than the 60-px tile height.

### 12.2 PCA dimensionality and preceding execution errors (test 2)

| ID | Expected | Observed | Status |
|---|---|---|---|
| FE-1 | NameError in the colour cell after `data = np.load(...)` | `NameError: name 'little_images' is not defined` | Confirmed (executed) |
| LB-1 | Non-raw path contains TAB | `'path\to\\little_images.npy'` (the `\t` is a TAB) | Confirmed (executed) |
| LB-2 | `rgb_vector` undefined in a fresh Labeling kernel | `NameError: name 'rgb_vector' is not defined` | Confirmed (executed) |
| **LB-3** | `PCA(128)` on the 72-column Gabor block fails | `ValueError: n_components=128 must be between 0 and min(n_samples, n_features)=72` (200 rows) | **Confirmed (executed)** |
| LB-3 (any N) | Same with 10,000 rows | Same ValueError | Confirmed (executed) |
| LB-4 | Default CSV export → positional slice captures the index | `data.iloc[:, :-512]` has **73** columns incl. `Unnamed: 0` (72 with `index=False`) | Confirmed (executed) |
| Paper design | 48 colour PCs + 48 Gabor = 96 | Fits without error | Confirmed |

The failure is structural: it depends on the number of features, not on sample size. The descriptor table was produced by Feature_Extraction cell [9] verbatim, so column names and order are exactly the authors'.

### 12.3 SOM feature leakage (test 3)

**Part A, direct inspection** (Labeling cells [4], [6] and the SOM part of [8], with patches only for LB-2 and LB-3):

| ID | Expected | Observed | Status |
|---|---|---|---|
| **LB-6** | A positional column enters `df_unified` | Columns other than PCs and Gabor: **`index`** (CSV with `index=False`), or **`index` + `Unnamed: 0`** (pandas default). `index` equals the original row positions 0, 2, 4, … exactly; correlation with row order is 1.0. | **Confirmed (executed)** |
| LB-6 | The leaked column carries full weight | Standard deviation after `StandardScaler` = **1.0** (same weight as every descriptor; 1/202 of total variance) | Confirmed (executed) |
| LB-5 | Raw Gabor is concatenated, not its PCs | 72 `freq_*` columns, 0 `*_gabor` PC columns | Confirmed (executed) |
| LB-7 | SOM input is 201–202 features, not 96 | 201 (`index=False`) / 202 (default CSV) | Confirmed (executed) |
| LB-8 | `init_weights_sst` undefined | `NameError: name 'init_weights_sst' is not defined` | Confirmed (executed) |
| LB-9 | Init in the wrong coordinate space | Code init ≠ the same rows in scaled space. Mean quantization error at init: **17.8** (code) vs 14.3 (same samples, correct space) vs 14.1 (MiniSom `pca_weights_init`) | Confirmed (executed). Impact after 5M iterations is probably small, but untested. |

New minor observation: cell [8] runs PCA with `n_components = n_features` (202), so it needs at least 202 samples. This is irrelevant for real data sets but caused a failure with a tiny test set.

**Part B, impact (supported hypothesis).** 3,000 synthetic descriptor vectors (200 dims, 4 appearance classes drawn **independently of depth**), with or without the depth column, and MiniSom 10×10 (σ = 1.25, lr = 0.01, PCA init, 156k iterations ≈ the paper's iterations per sample). Normalized mutual information (NMI) between BMU and depth decile, expressed as a z-score against 20 shuffled-depth baselines:

| Appearance signal | Leak | Order | NMI(BMU, depth) | Shuffled baseline | z | NMI(BMU, class) |
|---|---|---|---|---|---|---|
| strong | no | sequential | 0.041 | 0.043 | −0.8 | 0.485 |
| strong | no | random | 0.040 | 0.040 | 0.2 | 0.480 |
| strong | **yes** | sequential | 0.061 | 0.041 | **11.1** | 0.485 |
| strong | **yes** | random | 0.054 | 0.042 | **4.9** | 0.480 |
| weak | no | sequential | 0.053 | 0.048 | 2.3 | 0.248 |
| weak | no | random | 0.050 | 0.048 | 0.6 | 0.237 |
| weak | **yes** | sequential | 0.063 | 0.048 | **6.0** | 0.243 |
| weak | **yes** | random | 0.057 | 0.048 | **4.1** | 0.234 |

**Verdict: supported, small effect.** The leaked index produces a statistically clear but small depth imprint on SOM nodes. It is strongest with MiniSom's default sequential (depth-ordered) training, which is what the notebook uses, and it does not degrade appearance clustering in this setting. Sequential order alone produced a weak imprint when the appearance signal was weak (z = 2.3). Caveats: single seed, 1 leaked column out of 201, synthetic features. On real data, where neighbouring slices also look alike, the effect cannot be separated from genuine depth autocorrelation without the data. Either way, the column must be removed: it is a bug, and the fix costs nothing.

### 12.4 Dataset alignment (test 4)

Synthetic core strip (60 px × 12,000 px, blocky "lithology" intervals with class-specific colour and texture) → 300×60 tiles with 50% overlap (79 tiles) and without overlap (40 tiles) → Feature_Extraction cells [6], [8], [9] verbatim → CSV (pandas default) → Labeling cell [4] → label export as in cell [11] → Deep_Learning cells [4] and [6] verbatim (Keras `to_categorical` replaced by a NumPy one-hot).

| ID | Question | Observed | Status |
|---|---|---|---|
| T4a | Does `overlap[::2]` reproduce the non-overlapping tiling? | Yes, arrays identical (40 × 60 × 300 × 3) | Confirmed: the authors' `[::2]` rationale is correct |
| T4b | Inside Labeling, do `little_images[::2]` and `data.iloc[::2]` stay aligned? | **Yes.** Descriptors recomputed from the subsampled images match the subsampled CSV rows exactly (max diff 0.0); a 1-row-shifted control differs by 0.33 | Confirmed: alignment **preserved** within Labeling |
| T4c | Labeling cell [6] with `rgb_vector` from the Feature_Extraction kernel | 79 colour rows vs 40 Gabor rows → `df_unified` has 79 rows; **39 rows contain NaN**; colour PCs of image *i* sit beside the Gabor features of image 2*i*; cell [8] PCA fails (`Input X contains NaN`) | Confirmed (executed): **misalignment** |
| **DL-2** | Labels made for `images[::2]`, loaded by Deep_Learning with the full array | Only **1 of 40** labels is paired with the image it was made for (index 0). The label matches the true class of the paired image in **11/40 (28%)** cases. **39 images are never used.** No error is raised. | **Confirmed (executed): silent label corruption** |
| §7 split | Random 80/20 slice-level split (`random_state=42`), 18,355 tiles | **96.4%** of test tiles have an overlapping (50% overlap) or contiguous (no overlap) training neighbour; **99.9%** have a training tile within ±2 slices (≤ 36 cm) | Confirmed (math); matches 1 − 0.2² |
| DL-3 | `list(set(labels))` class → index mapping | **5 distinct orders** across 6 `PYTHONHASHSEED` values | Confirmed (executed) |

**Verdict.** Slicing and `[::2]` are internally consistent *within* the Labeling notebook. Correspondence breaks at the two hand-offs: Feature_Extraction → Labeling (`rgb_vector`), and Labeling → Deep_Learning (the full array is loaded). Neither raises an error at the hand-off itself, so a run that "works" can silently train on wrong labels. This is why the plan (§11, P1) keys every artefact on an explicit `slice_id` rather than on array position.

### 12.5 What remains unverified

| Finding | Why not tested | Confidence |
|---|---|---|
| Mean Teacher consistency term has zero gradient in the code behind Table 3 (§5) | Needs TensorFlow | High: by construction, NumPy outputs of `predict()` are constants inside `GradientTape` |
| DL-8 / DL-12 (test-as-validation, validation in the MT stream) | Needs TensorFlow; the logic is visible in the code | High (static) |
| Impact of random vs hole-grouped splits on accuracy | Needs real data | The neighbour statistic is confirmed (§12.4); the size of the accuracy inflation is unknown |
| FE-8 (equalization vs min-max) effect on colour classes | Needs real colour data | Hypothesis |
| LB-10 MiniSom default decay changing between versions | Not run across versions | Hypothesis |
| `scipy.misc` import (FE-10) | Tested on SciPy 1.18.1 | **Refuted as a failure:** it still imports, with `DeprecationWarning: scipy.misc is deprecated and will be removed in 2.0.0`. Unused, so remove it. |
