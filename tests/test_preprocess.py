import numpy as np
import pandas as pd

from src.preprocess import (
    detect_outlier_samples,
    filter_low_expression_genes,
    log_transform,
    qc_report,
    select_variable_genes,
    zscore_normalize,
)


def test_qc_report_matches_shape(small_cohort):
    expr, _ = small_cohort
    report = qc_report(expr)
    assert report["n_genes"] == expr.shape[0]
    assert report["n_samples"] == expr.shape[1]
    assert 0 <= report["pct_missing"] < 5


def test_filter_low_expression_genes_reduces_or_keeps(small_cohort):
    expr, _ = small_cohort
    filtered = filter_low_expression_genes(expr.fillna(0), min_samples_expressing=0.5)
    assert filtered.shape[0] <= expr.shape[0]
    assert (filtered > 0).sum(axis=1).min() >= 0.5 * expr.shape[1]


def test_log_transform_nonnegative_and_monotonic():
    expr = pd.DataFrame({"s1": [0.0, 10.0, 1000.0], "s2": [5.0, 50.0, 500.0]}, index=["g1", "g2", "g3"])
    log_expr = log_transform(expr, "log2")
    assert (log_expr >= 0).all().all()
    assert log_expr.loc["g3", "s1"] > log_expr.loc["g2", "s1"] > log_expr.loc["g1", "s1"]


def test_variable_gene_selection_respects_top_n(small_cohort, small_cfg):
    expr, _ = small_cohort
    log_expr = log_transform(expr.fillna(0), "log2")
    top_n = 50
    variable_expr, variance = select_variable_genes(log_expr, top_n=top_n, method="mad")
    assert variable_expr.shape[0] == top_n
    assert variance.shape[0] == log_expr.shape[0]


def test_zscore_normalize_mean_zero_sd_one(small_cohort):
    expr, _ = small_cohort
    log_expr = log_transform(expr.fillna(0), "log2")
    scaled = zscore_normalize(log_expr)
    np.testing.assert_allclose(scaled.mean(axis=1).values, 0, atol=1e-6)
    np.testing.assert_allclose(scaled.std(axis=1).values, 1, atol=1e-6)


def test_outlier_detection_returns_valid_subset(small_cohort):
    expr, _ = small_cohort
    log_expr = log_transform(expr.fillna(0), "log2")
    outliers, distances = detect_outlier_samples(log_expr, n_pcs=5, z_threshold=3.5)
    assert set(outliers).issubset(set(expr.columns))
    assert len(distances) == expr.shape[1]
