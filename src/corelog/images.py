"""Image I/O. JPEG/PNG/TIFF via Pillow; Canon raw (.CR2/.CR3) via the optional `rawpy` extra."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

RAW_EXT = {".cr2", ".cr3", ".nef", ".dng", ".arw"}
Image.MAX_IMAGE_PIXELS = 400_000_000  # core-box photos can be very large


def is_raw(path: Path) -> bool:
    return Path(path).suffix.lower() in RAW_EXT


def load_rgb(path: str | Path) -> np.ndarray:
    """Return an 8-bit RGB array (H, W, 3)."""
    path = Path(path)
    if is_raw(path):
        try:
            import rawpy
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError(f"{path.name}: install the raw extra (pip install -e '.[raw]') "
                              "or convert raw photos to TIFF/JPEG first") from e
        with rawpy.imread(str(path)) as raw:
            # camera white balance, no auto-brightening: keeps photos comparable to each other
            return raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8)
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def image_size(path: str | Path) -> tuple[int, int]:
    """(width, height) without decoding the full image where possible."""
    path = Path(path)
    if is_raw(path):
        a = load_rgb(path)
        return a.shape[1], a.shape[0]
    with Image.open(path) as im:
        return im.size


def save_rgb(arr: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
