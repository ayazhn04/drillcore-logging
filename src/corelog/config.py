"""YAML configuration: layered files + command-line overrides + portable paths.

Precedence (later wins): configs/base.yaml -> each --config file -> --set key=value.
Paths are never absolute in committed configs:
  * `paths.data_root`   - where the RAs' dataset lives. Set it in a local, git-ignored
                          config (configs/lab.yaml) or with the CORELOG_DATA_ROOT env var.
  * `paths.output_root` - where tiles, features, runs and logs are written
                          (env var CORELOG_OUTPUT_ROOT).
Relative dataset paths are resolved against data_root; generated paths against output_root.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"


class ConfigError(ValueError):
    pass


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_value(s: str) -> Any:
    # YAML parsing gives ints, floats, bools, lists ("[a, b]") and null for free.
    return yaml.safe_load(s)


def apply_override(cfg: dict, assignment: str) -> None:
    if "=" not in assignment:
        raise ConfigError(f"override must be key=value, got {assignment!r}")
    key, value = assignment.split("=", 1)
    node = cfg
    parts = key.strip().split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise ConfigError(f"cannot set {key}: {p} is not a mapping")
    node[parts[-1]] = _parse_value(value)


def load_config(files: Iterable[str | os.PathLike] = (), overrides: Iterable[str] = (),
                base: str | os.PathLike | None = BASE_CONFIG) -> dict:
    cfg: dict = {}
    for f in ([base] if base else []) + list(files):
        with open(f, encoding="utf-8") as fh:
            cfg = deep_merge(cfg, yaml.safe_load(fh) or {})
    for o in overrides:
        apply_override(cfg, o)
    paths = cfg.setdefault("paths", {})
    if os.environ.get("CORELOG_DATA_ROOT"):
        paths["data_root"] = os.environ["CORELOG_DATA_ROOT"]
    if os.environ.get("CORELOG_OUTPUT_ROOT"):
        paths["output_root"] = os.environ["CORELOG_OUTPUT_ROOT"]
    return cfg


def _expand(p: str | os.PathLike) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(p))))


def data_root(cfg: dict) -> Path:
    v = cfg.get("paths", {}).get("data_root")
    if not v:
        raise ConfigError("paths.data_root is not set (use a local config or CORELOG_DATA_ROOT)")
    return _expand(v)


def output_root(cfg: dict) -> Path:
    return _expand(cfg.get("paths", {}).get("output_root", "outputs"))


def data_path(cfg: dict, key: str) -> Path:
    """A dataset file named in paths.<key>, resolved against data_root."""
    p = _expand(cfg["paths"][key])
    return p if p.is_absolute() else data_root(cfg) / p


def out_path(cfg: dict, *parts: str) -> Path:
    p = output_root(cfg).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def classes(cfg: dict) -> list[str]:
    c = cfg.get("labels", {}).get("classes")
    if not c:
        raise ConfigError("labels.classes must list the model classes in a fixed order")
    if len(set(c)) != len(c):
        raise ConfigError("labels.classes contains duplicates")
    return list(c)


def dump(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
