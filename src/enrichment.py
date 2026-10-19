"""Differential expression (one-cluster-vs-rest) and functional enrichment.

Differential expression uses a Welch's t-test per gene (appropriate for
unequal-variance, roughly-normal log-expression data) with
Benjamini-Hochberg FDR correction -- the same overall approach as
``limma``/``DESeq2`` contrasts, simplified for an already log2-normalized,
variance-stabilized matrix.

Functional enrichment queries the public Enrichr API (via ``gseapy``) for
GO Biological Process, KEGG, and Reactome gene sets. Because
:mod:`src.simulate` uses real HGNC gene symbols for every marker gene, these
queries return real, interpretable annotations even when run against the
synthetic demo cohort.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .logging_config import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Differential expression
# --------------------------------------------------------------------------- #
def differential_expression_one_vs_rest(log_expr: pd.DataFrame, cluster_labels: pd.Series, cluster_id: Any) -> pd.DataFrame:
    """Welch's t-test of each gene: samples in ``cluster_id`` vs. all others.

    ``log_expr`` is genes x samples, log2-scale (not necessarily variance-scaled --
    using the full log-expression matrix, not just the top-variable subset,
    gives DE testing access to genes that were excluded from clustering but
    may still be markers of the discovered groups).
    """
    cluster_labels = cluster_labels.loc[log_expr.columns]
    in_group = cluster_labels == cluster_id
    a = log_expr.loc[:, in_group.values]
    b = log_expr.loc[:, ~in_group.values]

    t_stat, p_val = stats.ttest_ind(a.values, b.values, axis=1, equal_var=False, nan_policy="omit")
    mean_a, mean_b = a.mean(axis=1), b.mean(axis=1)
    log2fc = (mean_a - mean_b).values

    padj = multipletests(np.nan_to_num(p_val, nan=1.0), method="fdr_bh")[1]

    df = pd.DataFrame(
        {
            "gene": log_expr.index,
            "mean_in_cluster": mean_a.values,
            "mean_other": mean_b.values,
            "log2FC": log2fc,
            "t_stat": t_stat,
            "pvalue": p_val,
            "padj": padj,
        }
    ).sort_values("padj")
    return df


def run_de_for_all_clusters(log_expr: pd.DataFrame, cluster_labels: pd.Series) -> dict[Any, pd.DataFrame]:
    results = {}
    for cluster_id in sorted(cluster_labels.unique()):
        log.info("Running differential expression: cluster %s vs rest...", cluster_id)
        results[cluster_id] = differential_expression_one_vs_rest(log_expr, cluster_labels, cluster_id)
    return results


def top_de_genes(de_df: pd.DataFrame, direction: str = "up", n: int = 150, fdr_thresh: float = 0.05, fc_thresh: float = 1.0) -> list[str]:
    sig = de_df[de_df["padj"] < fdr_thresh]
    if direction == "up":
        sig = sig[sig["log2FC"] >= fc_thresh].sort_values("log2FC", ascending=False)
    elif direction == "down":
        sig = sig[sig["log2FC"] <= -fc_thresh].sort_values("log2FC")
    else:
        sig = sig.reindex(sig["log2FC"].abs().sort_values(ascending=False).index)
    return sig["gene"].head(n).tolist()


# --------------------------------------------------------------------------- #
# Functional enrichment (GO / KEGG / Reactome via Enrichr)
# --------------------------------------------------------------------------- #
def run_enrichment(
    gene_list: list[str],
    gene_set_libraries: list[str] | None = None,
    organism: str = "human",
) -> dict[str, pd.DataFrame]:
    """Query Enrichr (via gseapy) for GO / KEGG / Reactome enrichment.

    Returns a dict keyed by library name, each a DataFrame with the
    standard Enrichr columns plus a normalized ``gene_count`` column.
    Falls back to an empty, well-formed DataFrame per library (with a
    logged warning) if the network call fails, so the rest of the pipeline
    can proceed offline / in CI without a hard dependency on Enrichr uptime.
    """
    import gseapy as gp

    gene_set_libraries = gene_set_libraries or [
        "GO_Biological_Process_2023", "KEGG_2021_Human", "Reactome_2022",
    ]
    results = {}
    for lib in gene_set_libraries:
        try:
            enr = gp.enrichr(gene_list=gene_list, gene_sets=lib, organism=organism, outdir=None)
            df = enr.results.copy()
            df["gene_count"] = df["Overlap"].apply(lambda s: int(s.split("/")[0]))
            results[lib] = df.sort_values("Adjusted P-value")
        except Exception as exc:  # pragma: no cover - network dependent
            log.warning("Enrichment query failed for library %s: %s", lib, exc)
            results[lib] = pd.DataFrame(
                columns=["Term", "Overlap", "P-value", "Adjusted P-value", "Genes", "gene_count"]
            )
    return results


def run_enrichment_for_all_clusters(
    de_results: dict[Any, pd.DataFrame], gene_set_libraries: list[str] | None = None,
    top_n: int = 150, direction: str = "up",
) -> dict[Any, dict[str, pd.DataFrame]]:
    out = {}
    for cluster_id, de_df in de_results.items():
        genes = top_de_genes(de_df, direction=direction, n=top_n)
        log.info("Enrichment for cluster %s (%d marker genes)...", cluster_id, len(genes))
        out[cluster_id] = run_enrichment(genes, gene_set_libraries) if genes else {}
    return out
