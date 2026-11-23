"""Image descriptors for SOM clustering (paper §3.2.1), with the audit's corrections.

Gabor (paper Eq. 4-5): grayscale image convolved with the COMPLEX Gabor kernel, then
the modulus |I * G| is taken (the authors' code took |G| first, which is an isotropic
blur and loses orientation - audit FE-3, tests 1/1b). Frequencies must be < 0.5
cycles/px (Nyquist; audit FE-5). Boundary handling defaults to 'reflect' (FE-6).
FFT convolution makes the 69x69 low-frequency kernels cheap.

Colour: 3-D RGB histogram (8x8x8 = 512 bins) after per-channel normalisation:
'minmax' (paper text), 'equalize' (authors' code), or 'none'.

Rock masks (features.use_mask = auto | true | false): for rock-only images with a black
background, the background would dominate both descriptors (a black histogram peak, and
strong Gabor edges along the outline of every core piece). With a mask, the colour
histogram and its min-max range use rock pixels only, and for Gabor the background is
filled with the mean rock grey level before filtering and the response statistics are
taken over rock pixels only. Tiles with fewer than MIN_ROCK_PIXELS rock pixels fall back
to the whole tile.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from scipy.signal import fftconvolve
from skimage.exposure import equalize_hist
from skimage.filters import gabor_kernel

from ..config import output_root

FEATURES_NPZ = "features/features.npz"
MIN_ROCK_PIXELS = 50


@lru_cache(maxsize=64)
def _kernel(f: float, theta_deg: float) -> np.ndarray:
    if not 0 < f < 0.5:
        raise ValueError(f"Gabor frequency {f} must be in (0, 0.5) cycles/px (Nyquist)")
    return gabor_kernel(f, theta=np.deg2rad(theta_deg))


def gabor_response(gray: np.ndarray, f: float, theta_deg: float, boundary: str = "reflect") -> np.ndarray:
    """|I * G| with the same 'same-size' semantics as scipy.ndimage.convolve(mode=boundary)."""
    k = _kernel(f, theta_deg)
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    mode = {"reflect": "symmetric", "mirror": "reflect", "wrap": "wrap", "constant": "constant"}[boundary]
    # np.pad repeats the reflection/wrap when the kernel is larger than the image (69 px > 60 px)
    padded = np.pad(gray, ((ph, kh - 1 - ph), (pw, kw - 1 - pw)), mode=mode)
    return np.abs(fftconvolve(padded, k, mode="valid"))


def _usable(mask: np.ndarray | None) -> bool:
    return mask is not None and int(mask.sum()) >= MIN_ROCK_PIXELS


def gabor_features(img: np.ndarray, freqs, thetas_deg, boundary="reflect", minmax=False,
                   mask: np.ndarray | None = None) -> np.ndarray:
    gray = np.asarray(Image.fromarray(img).convert("L"), dtype=np.float64) / 255.0
    sel = mask.astype(bool) if _usable(mask) else None
    if sel is not None:
        gray = np.where(sel, gray, gray[sel].mean())  # no artificial rock/background edge
    out = []
    for f in freqs:
        for t in thetas_deg:
            m = gabor_response(gray, float(f), float(t), boundary)
            v = m[sel] if sel is not None else m.ravel()
            if minmax:
                rng = v.max() - v.min()
                v = (v - v.min()) / rng if rng > 0 else np.zeros_like(v)
            out += [v.mean(), v.std()]
    return np.asarray(out)


def gabor_names(freqs, thetas_deg) -> list[str]:
    return [f"gabor_f{f}_t{t}_{s}" for f in freqs for t in thetas_deg for s in ("mean", "std")]


def color_histogram(img: np.ndarray, bins: int = 8, normalize: str = "minmax",
                    mask: np.ndarray | None = None) -> np.ndarray:
    x = img.astype(np.float64) / 255.0
    px = x[mask.astype(bool)] if _usable(mask) else x.reshape(-1, 3)  # (N, 3)
    if normalize == "minmax":
        lo, hi = px.min(axis=0), px.max(axis=0)
        px = (px - lo) / np.where(hi > lo, hi - lo, 1.0)
    elif normalize == "equalize":
        if _usable(mask):
            px = np.stack([equalize_hist(px[:, c], mask=None) for c in range(3)], -1)
        else:
            x = np.stack([equalize_hist(x[..., c]) for c in range(3)], -1)
            px = x.reshape(-1, 3)
    elif normalize != "none":
        raise ValueError(f"unknown color normalize {normalize!r}")
    h, _ = np.histogramdd(px, bins=bins, range=((0, 1),) * 3)
    return (h / h.sum()).ravel()


def _one(path, fcfg, mask_path=None):
    with Image.open(path) as im:
        img = np.asarray(im.convert("RGB"))
    mask = None
    if mask_path is not None:
        with Image.open(mask_path) as im:
            mask = np.asarray(im.convert("L")) >= 128
    g, c = fcfg["gabor"], fcfg["color"]
    return (gabor_features(img, g["frequencies"], g["thetas_deg"], g.get("boundary", "reflect"),
                           g.get("minmax_normalize", False), mask),
            color_histogram(img, c.get("bins", 8), c.get("normalize", "minmax"), mask))


def _mask_paths(tiles: pd.DataFrame, root, use_mask) -> list:
    has = "mask_path" in tiles and tiles["mask_path"].map(lambda v: isinstance(v, str) and v != "").any()
    if use_mask in (False, "false", "no", "off") or (use_mask == "auto" and not has):
        return [None] * len(tiles)
    if not has:
        raise ValueError("features.use_mask=true but tiles.csv has no mask tiles (set tiling.mask_template)")
    return [root / m if isinstance(m, str) and m else None for m in tiles["mask_path"]]


def extract(cfg: dict, tiles: pd.DataFrame) -> dict:
    """Compute descriptors for all tiles; saved as NPZ keyed by slice_id (row order = tiles order)."""
    root, fc = output_root(cfg), cfg["features"]
    masks = _mask_paths(tiles, root, fc.get("use_mask", "auto"))
    res = Parallel(n_jobs=fc.get("n_jobs", 1))(delayed(_one)(root / p, fc, m)
                                               for p, m in zip(tiles["tile_path"], masks))
    out = {"slice_id": tiles["slice_id"].to_numpy().astype(str),
           "gabor": np.stack([r[0] for r in res]), "color": np.stack([r[1] for r in res]),
           "gabor_names": np.array(gabor_names(fc["gabor"]["frequencies"], fc["gabor"]["thetas_deg"])),
           "masked": np.array([m is not None for m in masks])}
    p = root / FEATURES_NPZ
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **out)
    return out


def load(cfg: dict) -> dict:
    with np.load(output_root(cfg) / FEATURES_NPZ, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
