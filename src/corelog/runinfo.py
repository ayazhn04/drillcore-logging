"""Reproducibility helpers: seeding, device choice, run folders and environment capture."""
from __future__ import annotations

import datetime as dt
import json
import platform
import random
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from .config import REPO_ROOT, dump, output_root

PKGS = ["corelog", "torch", "torchvision", "numpy", "scipy", "scikit-image", "scikit-learn",
        "pandas", "MiniSom", "PyYAML", "pillow", "rawpy"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:  # pragma: no cover
        pass


def pick_device(name: str = "auto"):
    import torch
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")  # MPS skipped on purpose: not used on the lab machine
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("train.device=cuda but CUDA is not available (see docs/INSTALL.md)")
    return torch.device(name)


def _git(*args) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def environment() -> dict:
    vers = {}
    for p in PKGS:
        try:
            vers[p] = version(p)
        except PackageNotFoundError:
            vers[p] = None
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "packages": vers,
           "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain"))}
    try:
        import torch
        env["cuda_available"] = torch.cuda.is_available()
        env["cuda_version"] = torch.version.cuda
        env["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:  # pragma: no cover
        pass
    return env


def provenance(cfg: dict) -> dict:
    """Is the configured dataset the synthetic test set? Stamped on every run, metric and log."""
    from .config import data_root
    from .synthetic import MARKER
    try:
        synthetic = (data_root(cfg) / MARKER).exists()
    except Exception:  # noqa: BLE001 - data_root unset
        synthetic = False
    if synthetic:
        return {"data_provenance": "SYNTHETIC",
                "warning": "Synthetic test data: NOT a geological classification result."}
    return {"data_provenance": "real dataset (paths.data_root)"}


def new_run_dir(cfg: dict, kind: str, name: str | None = None) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    d = output_root(cfg) / "runs" / (name or f"{stamp}_{kind}_{cfg['model']['name']}")
    d.mkdir(parents=True, exist_ok=False)
    dump(cfg, d / "config.yaml")
    (d / "environment.json").write_text(json.dumps(environment(), indent=2))
    prov = provenance(cfg)
    (d / "provenance.json").write_text(json.dumps(prov, indent=2))
    if prov["data_provenance"] == "SYNTHETIC":
        (d / "SYNTHETIC_DATA_NOT_GEOLOGICAL_RESULTS.txt").write_text(prov["warning"] + "\n")
    return d
