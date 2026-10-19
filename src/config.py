"""Configuration loading utilities.

Every module in this package accepts a plain ``dict`` config produced by
:func:`load_config`, so notebooks, the CLI, and unit tests all share one
source of truth (``config.yaml`` at the repository root).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load ``config.yaml`` into a nested dict.

    Parameters
    ----------
    path:
        Path to a YAML config file. Defaults to ``<repo_root>/config.yaml``.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve one of the ``paths.*`` entries relative to the repo root."""
    rel = cfg["paths"][key]
    p = REPO_ROOT / rel
    p.mkdir(parents=True, exist_ok=True)
    return p


def override(cfg: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Return a deep-copied config with dotted-key overrides applied.

    Example
    -------
    >>> cfg = override(cfg, **{"preprocessing.top_n_variable_genes": 3000})
    """
    new_cfg = copy.deepcopy(cfg)
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        node = new_cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return new_cfg
