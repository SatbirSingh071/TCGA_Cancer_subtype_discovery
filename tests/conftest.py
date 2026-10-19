"""Shared pytest fixtures: a small, fast synthetic cohort for unit tests.

Tests intentionally use a tiny cohort (60 samples, 300 genes) so the full
suite runs in seconds, independent of the full-scale (500 x 4000) demo
cohort used for the actual figures/results in the repository.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config, override
from src.simulate import simulate_cohort


@pytest.fixture(scope="session")
def small_cfg():
    cfg = load_config(REPO_ROOT / "config.yaml")
    cfg = override(
        cfg,
        **{
            "simulate.n_samples": 60,
            "simulate.n_genes": 300,
            "preprocessing.top_n_variable_genes": 100,
            "dimensionality_reduction.pca.n_components": 10,
        },
    )
    return cfg


@pytest.fixture(scope="session")
def small_cohort(small_cfg):
    expr, clinical = simulate_cohort(small_cfg, seed=123)
    return expr, clinical
