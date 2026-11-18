from pathlib import Path

import pytest

from corelog.config import REPO_ROOT, load_config
from corelog.synthetic import make_dataset


def smoke_cfg(data_root: Path, out_root: Path, *overrides: str) -> dict:
    return load_config([REPO_ROOT / "configs" / "smoke.yaml"],
                       [f"paths.data_root={data_root}", f"paths.output_root={out_root}", *overrides])


@pytest.fixture(scope="session")
def synth_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("synthetic_dataset")
    make_dataset(root)
    return root


@pytest.fixture(scope="session")
def tiled(synth_root, tmp_path_factory):
    """Synthetic dataset already tiled and log-labelled (shared, read-only for tests)."""
    from corelog.labels import make_log_labels
    from corelog.tiling import make_tiles
    out = tmp_path_factory.mktemp("outputs")
    cfg = smoke_cfg(synth_root, out)
    tiles = make_tiles(cfg)
    labels = make_log_labels(cfg, tiles)
    return cfg, tiles, labels
