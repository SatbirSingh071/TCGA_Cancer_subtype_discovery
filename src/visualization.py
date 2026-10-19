"""Publication-quality plotting functions used throughout the pipeline.

Every function takes a ``save_path`` (or builds one from ``figures_dir``)
and returns the created ``matplotlib`` ``Figure`` so notebooks can display
it inline while the same call also persists a PNG for the README / reports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram

sns.set_theme(style="whitegrid", context="talk", font_scale=0.75)
PALETTE = sns.color_palette("Set2", 10)
SUBTYPE_PALETTE = {
    "Luminal A": "#4C72B0", "Luminal B": "#55A868",
    "Basal-like": "#C44E52", "HER2-enriched": "#8172B2",
}

FIGSIZE_WIDE = (10, 6)
FIGSIZE_SQUARE = (7, 7)
DPI = 150


def _save(fig: plt.Figure, save_path: Path | None) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")


# --------------------------------------------------------------------------- #
# QC / EDA
# --------------------------------------------------------------------------- #
def plot_library_size_distribution(library_size: pd.Series, save_path: Path | None = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    sns.histplot(library_size, bins=40, kde=True, ax=axes[0], color=PALETTE[0])
    axes[0].set_title("Library size distribution")
    axes[0].set_xlabel("Total expression (linear scale)")
    sns.boxplot(x=library_size, ax=axes[1], color=PALETTE[0])
    axes[1].set_title("Library size (boxplot)")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_missingness(missing_per_sample: pd.Series, missing_per_gene: pd.Series, save_path: Path | None = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    sns.histplot(missing_per_sample, bins=30, ax=axes[0], color=PALETTE[1])
    axes[0].set_title("Missing values per sample")
    sns.histplot(missing_per_gene, bins=30, ax=axes[1], color=PALETTE[1])
    axes[1].set_title("Missing values per gene")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_gene_variance_distribution(variance: pd.Series, top_n: int = 2000, save_path: Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    sorted_var = variance.sort_values(ascending=False).reset_index(drop=True)
    ax.plot(sorted_var.values, color=PALETTE[2], lw=1.5)
    ax.axvline(top_n, color="crimson", ls="--", label=f"top {top_n} cutoff")
    ax.set_xlabel("Gene rank")
    ax.set_ylabel("Variance (MAD)")
    ax.set_title("Gene variance distribution (ranked)")
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_sample_correlation_heatmap(
    expr: pd.DataFrame, annotation: pd.Series | None = None, save_path: Path | None = None, sample_frac: float = 1.0
) -> plt.Figure:
    """Heatmap of pairwise sample-sample Spearman correlation."""
    if sample_frac < 1.0:
        expr = expr.sample(frac=sample_frac, axis=1, random_state=0)
    corr = expr.corr(method="spearman")
    row_colors = None
    if annotation is not None:
        ann = annotation.loc[corr.index]
        lut = dict(zip(ann.unique(), sns.color_palette("Set2", ann.nunique())))
        row_colors = ann.map(lut)
    g = sns.clustermap(
        corr, cmap="vlag", center=corr.values.mean(), row_colors=row_colors, col_colors=row_colors,
        figsize=(9, 9), xticklabels=False, yticklabels=False,
    )
    g.fig.suptitle("Sample-sample correlation (Spearman)", y=1.02)
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        g.savefig(save_path, dpi=DPI, bbox_inches="tight")
    return g.fig


def plot_feature_distributions(log_expr: pd.DataFrame, genes: list[str], save_path: Path | None = None) -> plt.Figure:
    n = len(genes)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, gene in zip(axes, genes):
        sns.histplot(log_expr.loc[gene], bins=30, ax=ax, color=PALETTE[3], kde=True)
        ax.set_title(gene)
        ax.set_xlabel("log2(TPM + 1)")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# --------------------------------------------------------------------------- #
# PCA
# --------------------------------------------------------------------------- #
def plot_scree(explained_variance_ratio: np.ndarray, n_components: int = 30, save_path: Path | None = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    evr = explained_variance_ratio[:n_components]
    axes[0].bar(range(1, len(evr) + 1), evr * 100, color=PALETTE[0])
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("% variance explained")
    axes[0].set_title("Scree plot")

    cum = np.cumsum(explained_variance_ratio) * 100
    axes[1].plot(range(1, len(cum) + 1), cum, marker="o", ms=3, color=PALETTE[1])
    axes[1].axhline(80, color="grey", ls="--", lw=1)
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Cumulative % variance explained")
    axes[1].set_title("Cumulative explained variance")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_pc_scatter(
    scores: pd.DataFrame, x: str = "PC1", y: str = "PC2", hue: pd.Series | None = None,
    title: str | None = None, save_path: Path | None = None, palette: dict | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    plot_df = scores[[x, y]].copy()
    if hue is not None:
        plot_df["group"] = hue.loc[plot_df.index].values
        sns.scatterplot(data=plot_df, x=x, y=y, hue="group", ax=ax, s=45, alpha=0.85, palette=palette, edgecolor="white", linewidth=0.3)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title=hue.name or "group")
    else:
        sns.scatterplot(data=plot_df, x=x, y=y, ax=ax, s=45, alpha=0.85, color=PALETTE[0], edgecolor="white", linewidth=0.3)
    ax.set_title(title or f"{x} vs {y}")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_embedding_grid(
    embeddings: dict[str, pd.DataFrame], hue: pd.Series, ncols: int = 3, title: str = "",
    save_path: Path | None = None, palette: dict | None = None,
) -> plt.Figure:
    """Grid of 2-D embeddings (e.g. multiple t-SNE perplexities or UMAP params)."""
    n = len(embeddings)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.2 * nrows), squeeze=False)
    for ax, (name, emb) in zip(axes.flat, embeddings.items()):
        cols = emb.columns[:2]
        plot_df = emb.copy()
        plot_df["group"] = hue.loc[plot_df.index].values
        sns.scatterplot(data=plot_df, x=cols[0], y=cols[1], hue="group", ax=ax, s=25, alpha=0.85, palette=palette, legend=False)
        ax.set_title(name, fontsize=11)
    for ax in axes.flat[n:]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels() if n else ([], [])
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# --------------------------------------------------------------------------- #
# Clustering diagnostics
# --------------------------------------------------------------------------- #
def plot_elbow_silhouette(sweep_df: pd.DataFrame, save_path: Path | None = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    axes[0].plot(sweep_df["k"], sweep_df["inertia"], marker="o", color=PALETTE[0])
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia (within-cluster SS)")
    axes[0].set_title("Elbow plot")

    axes[1].plot(sweep_df["k"], sweep_df["silhouette"], marker="o", color=PALETTE[2])
    best_k = sweep_df.loc[sweep_df["silhouette"].idxmax(), "k"]
    axes[1].axvline(best_k, color="crimson", ls="--", label=f"best k={int(best_k)}")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Mean silhouette score")
    axes[1].set_title("Silhouette vs k")
    axes[1].legend()
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_silhouette_samples_plot(X: np.ndarray, labels: np.ndarray, save_path: Path | None = None) -> plt.Figure:
    from sklearn.metrics import silhouette_samples, silhouette_score

    sil_values = silhouette_samples(X, labels)
    sil_avg = silhouette_score(X, labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    y_lower = 10
    clusters = sorted(set(labels))
    colors = sns.color_palette("Set2", len(clusters))
    for i, c in enumerate(clusters):
        vals = np.sort(sil_values[labels == c])
        y_upper = y_lower + len(vals)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, color=colors[i], alpha=0.8)
        ax.text(-0.05, y_lower + 0.5 * len(vals), str(c))
        y_lower = y_upper + 10
    ax.axvline(sil_avg, color="red", ls="--", label=f"mean = {sil_avg:.3f}")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster")
    ax.set_title("Per-sample silhouette plot")
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_dendrogram(Z: np.ndarray, labels: list[str] | None = None, color_threshold: float | None = None, save_path: Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 5))
    dendrogram(Z, labels=labels, ax=ax, color_threshold=color_threshold, no_labels=True)
    ax.set_title("Hierarchical clustering dendrogram")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_cluster_heatmap(
    scaled_expr: pd.DataFrame, cluster_labels: pd.Series, genes: list[str] | None = None,
    save_path: Path | None = None, palette: dict | None = None,
):
    """Heatmap of samples (columns) x marker genes (rows), grouped/annotated by cluster."""
    if genes is not None:
        data = scaled_expr.loc[scaled_expr.index.intersection(genes)]
    else:
        data = scaled_expr
    order = cluster_labels.sort_values().index
    data = data[order]
    ann = cluster_labels.loc[order]
    if palette is None:
        lut = dict(zip(sorted(ann.unique()), sns.color_palette("Set2", ann.nunique())))
    else:
        lut = palette
    col_colors = ann.map(lut)
    g = sns.clustermap(
        data, cmap="vlag", center=0, col_colors=col_colors, col_cluster=False, row_cluster=True,
        xticklabels=False, figsize=(12, max(6, 0.18 * len(data))), vmin=-3, vmax=3,
    )
    g.fig.suptitle("Cluster marker-gene heatmap", y=1.02)
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        g.savefig(save_path, dpi=DPI, bbox_inches="tight")
    return g.fig


# --------------------------------------------------------------------------- #
# Differential expression / enrichment
# --------------------------------------------------------------------------- #
def plot_volcano(de_df: pd.DataFrame, title: str = "Volcano plot", fc_thresh: float = 1.0, fdr_thresh: float = 0.05, save_path: Path | None = None, top_label_n: int = 15) -> plt.Figure:
    """``de_df`` must have columns: gene, log2FC, padj."""
    fig, ax = plt.subplots(figsize=(8, 7))
    df = de_df.copy()
    df["neg_log10_padj"] = -np.log10(df["padj"].clip(lower=1e-300))
    sig_up = (df["log2FC"] >= fc_thresh) & (df["padj"] < fdr_thresh)
    sig_down = (df["log2FC"] <= -fc_thresh) & (df["padj"] < fdr_thresh)
    not_sig = ~(sig_up | sig_down)

    ax.scatter(df.loc[not_sig, "log2FC"], df.loc[not_sig, "neg_log10_padj"], s=12, color="lightgrey", alpha=0.6, label="Not significant")
    ax.scatter(df.loc[sig_up, "log2FC"], df.loc[sig_up, "neg_log10_padj"], s=16, color="#C44E52", alpha=0.8, label="Up-regulated")
    ax.scatter(df.loc[sig_down, "log2FC"], df.loc[sig_down, "neg_log10_padj"], s=16, color="#4C72B0", alpha=0.8, label="Down-regulated")

    ax.axvline(fc_thresh, color="grey", ls="--", lw=1)
    ax.axvline(-fc_thresh, color="grey", ls="--", lw=1)
    ax.axhline(-np.log10(fdr_thresh), color="grey", ls="--", lw=1)

    top_genes = df.loc[sig_up | sig_down].reindex(df["neg_log10_padj"].sort_values(ascending=False).index).head(top_label_n)
    try:
        from adjustText import adjust_text
        texts = [ax.text(r["log2FC"], r["neg_log10_padj"], r["gene"], fontsize=8) for _, r in top_genes.iterrows()]
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))
    except ImportError:
        for _, r in top_genes.iterrows():
            ax.text(r["log2FC"], r["neg_log10_padj"], r["gene"], fontsize=8)

    ax.set_xlabel("log2 fold change")
    ax.set_ylabel("-log10(adjusted p-value)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_enrichment_dotplot(enrich_df: pd.DataFrame, title: str = "Enrichment", top_n: int = 15, save_path: Path | None = None) -> plt.Figure:
    """``enrich_df`` columns: Term, Adjusted P-value, Combined Score, Overlap (or gene count)."""
    df = enrich_df.sort_values("Adjusted P-value").head(top_n).copy()
    df["neg_log10_padj"] = -np.log10(df["Adjusted P-value"].clip(lower=1e-300))
    df = df.sort_values("neg_log10_padj")

    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(df))))
    sizes = (df["gene_count"] if "gene_count" in df.columns else df.get("Combined Score", pd.Series(20, index=df.index)))
    sizes = 30 + 200 * (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-9)
    sc = ax.scatter(df["neg_log10_padj"], df["Term"], s=sizes, c=df["neg_log10_padj"], cmap="viridis", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("-log10(adjusted p-value)")
    ax.set_title(title)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("-log10(FDR)")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_enrichment_barplot(enrich_df: pd.DataFrame, title: str = "Enrichment", top_n: int = 15, save_path: Path | None = None) -> plt.Figure:
    df = enrich_df.sort_values("Adjusted P-value").head(top_n).copy()
    df["neg_log10_padj"] = -np.log10(df["Adjusted P-value"].clip(lower=1e-300))
    df = df.sort_values("neg_log10_padj")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(df))))
    ax.barh(df["Term"], df["neg_log10_padj"], color=PALETTE[4])
    ax.set_xlabel("-log10(adjusted p-value)")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# --------------------------------------------------------------------------- #
# Survival analysis
# --------------------------------------------------------------------------- #
def plot_km_curves(km_fitters: dict[str, Any], title: str = "Kaplan-Meier survival by cluster", save_path: Path | None = None, palette: dict | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, kmf) in enumerate(km_fitters.items()):
        color = (palette or {}).get(name, PALETTE[i % len(PALETTE)])
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color, label=f"{name} (n={kmf.event_observed.shape[0]})")
    ax.set_xlabel("Days since diagnosis")
    ax.set_ylabel("Survival probability")
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# --------------------------------------------------------------------------- #
# Explainable AI / biomarkers
# --------------------------------------------------------------------------- #
def plot_biomarker_heatmap(scaled_expr: pd.DataFrame, biomarker_table: pd.DataFrame, cluster_labels: pd.Series, top_n_per_cluster: int = 8, save_path: Path | None = None, palette: dict | None = None):
    top_genes = (
        biomarker_table.sort_values(["cluster", "rank_score"], ascending=[True, False])
        .groupby("cluster").head(top_n_per_cluster)["gene"].tolist()
    )
    return plot_cluster_heatmap(scaled_expr, cluster_labels, genes=top_genes, save_path=save_path, palette=palette)
