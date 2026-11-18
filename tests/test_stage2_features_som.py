"""Stage 2: corrected Gabor/colour descriptors and SOM labelling.

The Gabor tests are the regression tests derived from the reproduction audit
(tests 1/1b): the descriptor must separate orientation on seam-free gratings,
where the authors' |kernel|-first implementation is at chance level.
"""
import copy

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
from sklearn.preprocessing import StandardScaler

from corelog.features import descriptors as D
from corelog.som import cluster as S

FREQS, THETAS = [0.05, 0.09, 0.17, 0.3, 0.45], [0, 45, 90, 135]


def _wave(kx, ky, rng, h=60, w=300):
    y, x = np.mgrid[0:h, 0:w]
    s = 0.5 + 0.3 * np.sin(2 * np.pi * (kx * x + ky * y) + rng.uniform(0, 6.28)) + rng.normal(0, 0.05, (h, w))
    return (np.repeat(np.clip(s, 0, 1)[..., None], 3, 2) * 255).astype(np.uint8)


def _loo_1nn(X, y):
    X = StandardScaler().fit_transform(X)
    d = ((X[:, None] - X[None]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    return (y[d.argmin(1)] == y).mean()


def test_gabor_fft_matches_ndimage_convolution():
    rng = np.random.default_rng(0)
    g = rng.random((60, 300))
    for f, t in [(0.05, 0), (0.17, 45), (0.45, 135)]:
        k = gabor_kernel(f, theta=np.deg2rad(t))
        ref = np.abs(ndi.convolve(g, k.real, mode="reflect") + 1j * ndi.convolve(g, k.imag, mode="reflect"))
        assert np.allclose(D.gabor_response(g, f, t, "reflect"), ref, atol=1e-9)
        refw = np.abs(ndi.convolve(g, k.real, mode="wrap") + 1j * ndi.convolve(g, k.imag, mode="wrap"))
        assert np.allclose(D.gabor_response(g, f, t, "wrap"), refw, atol=1e-9)


def test_gabor_is_orientation_selective_on_seamfree_gratings():
    """Audit test 1b condition: authors' code scored 0.30 (chance 0.25) here."""
    rng = np.random.default_rng(1)
    waves = {"0": (30 / 300, 0), "90": (0, 6 / 60), "30": (26 / 300, 3 / 60), "150": (26 / 300, -3 / 60)}
    X, y = [], []
    for c, (kx, ky) in enumerate(waves.values()):
        for _ in range(8):
            X.append(D.gabor_features(_wave(kx, ky, rng), FREQS, THETAS)); y.append(c)
    assert _loo_1nn(np.array(X), np.array(y)) >= 0.95


def test_gabor_rejects_frequencies_above_nyquist_and_counts_features():
    with pytest.raises(ValueError):
        D.gabor_features(np.zeros((60, 300, 3), np.uint8), [0.55], [0])
    f = D.gabor_features(np.zeros((60, 300, 3), np.uint8), FREQS, THETAS)
    assert f.shape == (2 * len(FREQS) * len(THETAS),) == (len(D.gabor_names(FREQS, THETAS)),)


def test_color_histogram_modes():
    img = np.zeros((60, 300, 3), np.uint8)
    img[:, :150] = (200, 40, 40)
    img[:, 150:] = (40, 40, 200)
    for mode in ("none", "minmax", "equalize"):
        h = D.color_histogram(img, 8, mode)
        assert h.shape == (512,) and h.sum() == pytest.approx(1.0)
    red = np.full((60, 300, 3), (200, 40, 40), np.uint8)
    assert not np.allclose(D.color_histogram(red, 8, "none"), D.color_histogram(img, 8, "none"))


def test_som_input_has_only_descriptors_and_is_order_invariant():
    rng = np.random.default_rng(2)
    gab, col = rng.random((200, 40)), rng.dirichlet(np.ones(512) * 0.1, 200)
    tr = S.SomInput(8).fit(gab, col)
    X = tr.transform(gab, col)
    assert X.shape == (200, 40 + 8) and tr.n_features == 48     # no index/position column (audit LB-6)
    perm = rng.permutation(200)
    Xp = S.SomInput(8).fit(gab[perm], col[perm]).transform(gab[perm], col[perm])
    assert np.allclose(np.abs(Xp), np.abs(X[perm]), atol=1e-6)  # rows depend on content, not position


def test_vote_majority_and_ties():
    a = pd.DataFrame({"node_row": [0, 0, 1], "node_col": [0, 1, 0], "label": ["x", "x", "y"]})
    b = pd.DataFrame({"node_row": [0, 0, 1], "node_col": [0, 1, 0], "label": ["x", "y", None]})
    v = S.vote([a, b], ["x", "y"]).set_index(["node_row", "node_col"])["label"]
    assert v[(0, 0)] == "x" and v[(0, 1)] == "unlabeled" and v[(1, 0)] == "unlabeled"
    with pytest.raises(ValueError):
        S.vote([pd.DataFrame({"node_row": [0], "node_col": [0], "label": ["typo"]})], ["x"])


def test_features_som_sheets_and_labels_end_to_end(tiled):
    cfg, tiles, _ = tiled
    cfg = copy.deepcopy(cfg)
    F = D.extract(cfg, tiles)
    assert F["gabor"].shape == (len(tiles), 2 * 3 * 4) and F["color"].shape == (len(tiles), 512)
    assert list(F["slice_id"]) == list(tiles["slice_id"])
    res = S.train(cfg)
    assert res["n_tiles"] == len(tiles) and res["n_features"] == 24 + 8
    assert res["vs_log"]["weighted_node_purity"] > 0.7      # synthetic classes are separable
    sheets = S.export_sheets(cfg, tiles)
    assert (sheets / "overview.png").exists() and len(list(sheets.glob("node_*.png"))) == res["nodes_used"]
    nvl = pd.read_csv(sheets.parent / "node_vs_log.csv")
    labeler = nvl.rename(columns={"majority": "label"})[["node_row", "node_col", "label"]]
    lab = S.apply_labels(cfg, [_write(labeler, sheets.parent / "labeling" / "geo1.csv")])
    assert lab["slice_id"].is_unique and set(lab["slice_id"]) == set(tiles["slice_id"])
    agree = lab.merge(pd.read_csv(sheets.parent.parent / "labels" / "labels_log.csv"), on="slice_id")
    agree = agree[(agree["label_x"] != "unlabeled") & (agree["label_y"] != "unlabeled")]
    assert (agree["label_x"] == agree["label_y"]).mean() > 0.7
    # reproducibility: same seed -> same assignments
    a1 = pd.read_csv(sheets.parent / "assignments.csv")
    S.train(cfg)
    assert a1.equals(pd.read_csv(sheets.parent / "assignments.csv"))


def _write(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
