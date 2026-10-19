import numpy as np

from src import clustering as cl
from src.preprocess import log_transform, select_variable_genes, zscore_normalize


def _small_scaled_expr(small_cohort):
    expr, _ = small_cohort
    log_expr = log_transform(expr.fillna(0), "log2")
    variable_expr, _ = select_variable_genes(log_expr, top_n=80, method="mad")
    return zscore_normalize(variable_expr)


def test_pca_shapes(small_cohort):
    scaled_expr = _small_scaled_expr(small_cohort)
    result = cl.run_pca(scaled_expr, n_components=10)
    assert result["scores"].shape == (scaled_expr.shape[1], 10)
    assert result["loadings"].shape == (scaled_expr.shape[0], 10)
    assert len(result["explained_variance_ratio"]) == 10
    assert np.all(np.diff(result["explained_variance_ratio"]) <= 1e-9)  # non-increasing


def test_kmeans_sweep_covers_k_range(small_cohort):
    scaled_expr = _small_scaled_expr(small_cohort)
    sweep = cl.kmeans_sweep(scaled_expr.T.values, k_range=(2, 5))
    assert list(sweep["k"]) == [2, 3, 4, 5]
    assert sweep["inertia"].is_monotonic_decreasing


def test_hierarchical_clusters_count(small_cohort):
    scaled_expr = _small_scaled_expr(small_cohort)
    Z = cl.hierarchical_linkage(scaled_expr.T.values, method="ward")
    labels = cl.hierarchical_clusters(Z, n_clusters=4)
    assert len(set(labels)) == 4
    assert len(labels) == scaled_expr.shape[1]


def test_hopkins_statistic_in_valid_range(small_cohort):
    scaled_expr = _small_scaled_expr(small_cohort)
    stat = cl.hopkins_statistic(scaled_expr.T.values)
    assert 0.0 <= stat <= 1.0


def test_evaluate_clustering_handles_noise_label():
    X = np.random.default_rng(0).normal(size=(50, 5))
    labels = np.array([0] * 20 + [1] * 20 + [-1] * 10)
    metrics = cl.evaluate_clustering(X, labels)
    assert metrics["n_clusters"] == 2
    assert metrics["n_noise"] == 10
