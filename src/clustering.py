"""Dimensionality reduction, clustering, and cluster-validity metrics.

Covers every method requested for the "Dimensionality Reduction" and
"Clustering" sections of the study: PCA, t-SNE (multi-perplexity), UMAP
(multi-neighbor/min_dist grid), K-Means (k=2..10 with elbow + silhouette),
agglomerative hierarchical clustering (Ward/complete/average linkage),
DBSCAN, HDBSCAN, Spectral clustering, and the standard trio of unsupervised
validity indices (silhouette, Davies-Bouldin, Calinski-Harabasz), plus a
Hopkins-statistic cluster-tendency check.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

try:
    import umap
except ImportError:  # pragma: no cover
    umap = None

try:
    import hdbscan as hdbscan_lib
except ImportError:  # pragma: no cover
    hdbscan_lib = None

from .logging_config import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Cluster tendency
# --------------------------------------------------------------------------- #
def hopkins_statistic(X: np.ndarray, sample_ratio: float = 0.1, seed: int = 42) -> float:
    """Hopkins statistic: tests whether ``X`` has meaningful cluster structure.

    Values near 0.5 indicate uniformly random (no cluster tendency) data;
    values approaching 1.0 indicate highly clusterable data. Computed by
    comparing nearest-neighbor distances of real samples to those of
    synthetic uniform-random points in the same feature space.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    m = max(int(sample_ratio * n), 1)

    nbrs = NearestNeighbors(n_neighbors=2).fit(X)
    idx = rng.choice(n, size=m, replace=False)
    u_distances, _ = nbrs.kneighbors(X[idx], n_neighbors=2)
    u_distances = u_distances[:, 1]  # nearest *other* real point

    mins, maxs = X.min(axis=0), X.max(axis=0)
    random_points = rng.uniform(mins, maxs, size=(m, d))
    w_distances, _ = nbrs.kneighbors(random_points, n_neighbors=1)
    w_distances = w_distances[:, 0]

    return float(w_distances.sum() / (w_distances.sum() + u_distances.sum()))


# --------------------------------------------------------------------------- #
# PCA
# --------------------------------------------------------------------------- #
def run_pca(scaled_expr: pd.DataFrame, n_components: int = 50, seed: int = 42) -> dict[str, Any]:
    """PCA on samples x genes (transposed input); returns scores, loadings, variance."""
    X = scaled_expr.T.values  # samples x genes
    n_components = min(n_components, min(X.shape) - 1)
    pca = PCA(n_components=n_components, random_state=seed)
    scores = pca.fit_transform(X)

    scores_df = pd.DataFrame(
        scores, index=scaled_expr.columns, columns=[f"PC{i+1}" for i in range(n_components)]
    )
    loadings_df = pd.DataFrame(
        pca.components_.T, index=scaled_expr.index, columns=[f"PC{i+1}" for i in range(n_components)]
    )
    return {
        "model": pca,
        "scores": scores_df,
        "loadings": loadings_df,
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }


def top_pc_loadings(loadings_df: pd.DataFrame, pc: str = "PC1", top_n: int = 25) -> pd.DataFrame:
    """Rank genes by absolute loading on a given PC (top drivers of that axis)."""
    s = loadings_df[pc].sort_values(key=np.abs, ascending=False).head(top_n)
    return s.to_frame(name=f"{pc}_loading")


# --------------------------------------------------------------------------- #
# t-SNE / UMAP
# --------------------------------------------------------------------------- #
def run_tsne(scaled_expr: pd.DataFrame, perplexity: int = 30, seed: int = 42, pca_components: int = 30) -> pd.DataFrame:
    X = scaled_expr.T.values
    if pca_components and X.shape[1] > pca_components:
        X = PCA(n_components=pca_components, random_state=seed).fit_transform(X)
    tsne = TSNE(
        n_components=2, perplexity=perplexity, learning_rate="auto",
        init="pca", random_state=seed, max_iter=1000,
    )
    embedding = tsne.fit_transform(X)
    return pd.DataFrame(embedding, index=scaled_expr.columns, columns=["tSNE1", "tSNE2"])


def run_umap(
    scaled_expr: pd.DataFrame, n_neighbors: int = 15, min_dist: float = 0.1, seed: int = 42, pca_components: int = 30
) -> pd.DataFrame:
    if umap is None:
        raise ImportError("umap-learn is required: pip install umap-learn")
    X = scaled_expr.T.values
    if pca_components and X.shape[1] > pca_components:
        X = PCA(n_components=pca_components, random_state=seed).fit_transform(X)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=seed)
    embedding = reducer.fit_transform(X)
    return pd.DataFrame(embedding, index=scaled_expr.columns, columns=["UMAP1", "UMAP2"])


# --------------------------------------------------------------------------- #
# K-Means sweep
# --------------------------------------------------------------------------- #
def kmeans_sweep(X: np.ndarray, k_range: tuple[int, int] = (2, 10), seed: int = 42) -> pd.DataFrame:
    """Fit K-Means for every k in the range; return inertia + validity metrics."""
    rows = []
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(X)
        row = {
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, labels) if k > 1 else np.nan,
            "davies_bouldin": davies_bouldin_score(X, labels) if k > 1 else np.nan,
            "calinski_harabasz": calinski_harabasz_score(X, labels) if k > 1 else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def fit_kmeans(X: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)


# --------------------------------------------------------------------------- #
# Hierarchical clustering
# --------------------------------------------------------------------------- #
def hierarchical_linkage(X: np.ndarray, method: str = "ward", metric: str = "euclidean") -> np.ndarray:
    if method == "ward" and metric != "euclidean":
        raise ValueError("Ward linkage requires the Euclidean metric.")
    condensed = pdist(X, metric=metric)
    return linkage(condensed, method=method)


def hierarchical_clusters(Z: np.ndarray, n_clusters: int) -> np.ndarray:
    return fcluster(Z, t=n_clusters, criterion="maxclust")


def compare_linkages(X: np.ndarray, n_clusters: int, methods: list[str] = ("ward", "complete", "average")) -> pd.DataFrame:
    rows = []
    for method in methods:
        Z = hierarchical_linkage(X, method=method)
        labels = hierarchical_clusters(Z, n_clusters)
        rows.append({
            "linkage": method,
            "silhouette": silhouette_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels),
            "cophenetic_correlation": _cophenetic_correlation(X, Z),
        })
    return pd.DataFrame(rows)


def _cophenetic_correlation(X: np.ndarray, Z: np.ndarray) -> float:
    from scipy.cluster.hierarchy import cophenet
    c, _ = cophenet(Z, pdist(X))
    return float(c)


# --------------------------------------------------------------------------- #
# Density-based / spectral (bonus methods)
# --------------------------------------------------------------------------- #
def fit_dbscan(X: np.ndarray, eps: float = 3.0, min_samples: int = 5) -> np.ndarray:
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)


def fit_hdbscan(X: np.ndarray, min_cluster_size: int = 15, min_samples: int = 5) -> np.ndarray:
    if hdbscan_lib is None:
        raise ImportError("hdbscan is required: pip install hdbscan")
    return hdbscan_lib.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(X)


def fit_spectral(X: np.ndarray, n_clusters: int = 4, seed: int = 42) -> np.ndarray:
    return SpectralClustering(
        n_clusters=n_clusters, affinity="nearest_neighbors", random_state=seed, assign_labels="kmeans"
    ).fit_predict(X)


# --------------------------------------------------------------------------- #
# Unified evaluation across all methods
# --------------------------------------------------------------------------- #
def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Silhouette / Davies-Bouldin / Calinski-Harabasz, ignoring noise label -1."""
    mask = labels != -1
    labels_eval = labels[mask]
    X_eval = X[mask]
    n_clusters = len(set(labels_eval))
    if n_clusters < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan, "n_clusters": n_clusters, "n_noise": int((~mask).sum())}
    return {
        "silhouette": silhouette_score(X_eval, labels_eval),
        "davies_bouldin": davies_bouldin_score(X_eval, labels_eval),
        "calinski_harabasz": calinski_harabasz_score(X_eval, labels_eval),
        "n_clusters": n_clusters,
        "n_noise": int((~mask).sum()),
    }


# --------------------------------------------------------------------------- #
# Explainable AI: ranked biomarker table
# --------------------------------------------------------------------------- #
def biomarker_ranking(
    scaled_expr: pd.DataFrame,
    cluster_labels: pd.Series,
    loadings_df: pd.DataFrame,
    de_results: dict[Any, pd.DataFrame],
    seed: int = 42,
) -> pd.DataFrame:
    """Rank each gene's contribution to subtype separation via three
    independent, complementary signals:

    1. **PCA loading magnitude** on PC1/PC2 -- genes driving the axes of
       greatest overall transcriptomic variation.
    2. **Differential-expression effect size** (|log2FC| from the
       one-vs-rest DE test) -- genes that best distinguish each specific
       cluster.
    3. **Random Forest feature importance** (Gini importance) from a
       classifier trained to predict cluster membership from expression --
       a model-based proxy that captures non-linear / interaction effects
       the first two linear signals can miss.

    All three signals are min-max normalized to [0, 1] and averaged into a
    single ``rank_score`` per (gene, cluster) pair, giving a ranked
    biomarker table suitable for a heatmap or a "top drivers" table.
    """
    from sklearn.ensemble import RandomForestClassifier

    X = scaled_expr.T.values
    y = cluster_labels.loc[scaled_expr.columns].values
    rf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
    rf.fit(X, y)
    rf_importance = pd.Series(rf.feature_importances_, index=scaled_expr.index, name="rf_importance")

    pc_loading_score = loadings_df[["PC1", "PC2"]].abs().max(axis=1)
    pc_loading_score = pc_loading_score.reindex(scaled_expr.index).fillna(0)

    def _norm(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0

    rows = []
    for cluster_id, de_df in de_results.items():
        de_df = de_df.set_index("gene")
        abs_fc = de_df["log2FC"].abs().reindex(scaled_expr.index).fillna(0)
        rank_score = (_norm(abs_fc) + _norm(pc_loading_score) + _norm(rf_importance)) / 3
        table = pd.DataFrame(
            {
                "gene": scaled_expr.index,
                "cluster": cluster_id,
                "log2FC": de_df["log2FC"].reindex(scaled_expr.index).values,
                "padj": de_df["padj"].reindex(scaled_expr.index).values,
                "pc_loading_score": pc_loading_score.values,
                "rf_importance": rf_importance.values,
                "rank_score": rank_score.values,
            }
        )
        rows.append(table)
    return pd.concat(rows, ignore_index=True).sort_values(["cluster", "rank_score"], ascending=[True, False])


def compare_all_methods(
    X: np.ndarray, cfg_clustering: dict[str, Any], seed: int = 42, X_density: np.ndarray | None = None
) -> pd.DataFrame:
    """Run every clustering algorithm requested in config and tabulate validity metrics.

    ``X_density`` (typically a lower-dimensional PCA embedding) is used for the
    density-/graph-based methods (DBSCAN, HDBSCAN, Spectral), which are far
    more sensitive to the curse of dimensionality than KMeans or hierarchical
    clustering; if omitted, ``X`` is reused for every method.
    """
    results = []
    X_density = X if X_density is None else X_density

    k = cfg_clustering["chosen_k"]
    labels_km = fit_kmeans(X, k, seed)
    results.append({"method": "KMeans", **evaluate_clustering(X, labels_km)})

    for linkage_method in cfg_clustering["hierarchical"]["linkages"]:
        Z = hierarchical_linkage(X, method=linkage_method)
        labels_hc = hierarchical_clusters(Z, cfg_clustering["hierarchical"]["n_clusters"])
        results.append({"method": f"Hierarchical-{linkage_method}", **evaluate_clustering(X, labels_hc)})

    labels_db = fit_dbscan(X_density, **cfg_clustering["dbscan"])
    results.append({"method": "DBSCAN (PCA space)", **evaluate_clustering(X_density, labels_db)})

    if hdbscan_lib is not None:
        labels_hdb = fit_hdbscan(X_density, **cfg_clustering["hdbscan"])
        results.append({"method": "HDBSCAN (PCA space)", **evaluate_clustering(X_density, labels_hdb)})

    labels_sc = fit_spectral(X_density, n_clusters=cfg_clustering["spectral"]["n_clusters"], seed=seed)
    results.append({"method": "Spectral (PCA space)", **evaluate_clustering(X_density, labels_sc)})

    return pd.DataFrame(results)
