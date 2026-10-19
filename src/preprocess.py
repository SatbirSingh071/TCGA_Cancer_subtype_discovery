"""Quality control and preprocessing of the raw gene expression matrix.

Pipeline implemented here (see ``notebooks/02_preprocessing.ipynb`` for the
narrated, figure-producing walkthrough):

1. Load raw genes x samples matrix + clinical metadata.
2. Quality control: library size per sample, genes detected per sample,
   missingness report.
3. Missing value assessment + imputation (gene-wise median).
4. Feature filtering: drop genes that are not expressed in a minimum
   fraction of samples (removes uninformative, mostly-zero genes).
5. Log2(x + 1) transformation to stabilize variance across the wide
   dynamic range of RNA-seq counts/TPM.
6. Outlier sample detection via robust Mahalanobis distance in PCA space.
7. Variance-based feature selection (top-N most variable genes by MAD or
   variance), the standard input reduction step before clustering.
8. Z-score normalization (gene-wise) so that every gene contributes on a
   comparable scale to downstream distance-based methods.
9. Batch-effect assessment: quantifies how much of the leading principal
   components' variance is explained by a nuisance batch variable
   (sequencing plate), and offers a simple mean-centering correction.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA

from .config import load_config, resolve_path
from .logging_config import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw_data(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw expression matrix + clinical table for the configured cancer type.

    Prefers a real GDC download if present; otherwise falls back to the
    simulated cohort (generating it on first use).
    """
    from .config import REPO_ROOT

    cancer_type = cfg["project"]["cancer_type"]
    raw_dir = REPO_ROOT / "data" / "raw"
    metadata_dir = REPO_ROOT / "data" / "metadata"

    real_expr_path = raw_dir / f"{cancer_type}_expression_matrix.tsv"
    real_clinical_path = metadata_dir / f"{cancer_type}_clinical.csv"
    sim_expr_path = raw_dir / f"{cancer_type}_simulated_expression_matrix.tsv"
    sim_clinical_path = metadata_dir / f"{cancer_type}_simulated_clinical.csv"

    if real_expr_path.exists() and real_clinical_path.exists():
        log.info("Loading real GDC-downloaded data from %s", real_expr_path)
        expr = pd.read_csv(real_expr_path, sep="\t", index_col=0)
        gene_names = expr.pop("gene_name") if "gene_name" in expr.columns else None
        if gene_names is not None:
            expr.index = gene_names.values
        clinical = pd.read_csv(real_clinical_path, index_col="case_submitter_id")
    else:
        if not sim_expr_path.exists():
            log.info("No cached data found; generating synthetic cohort (see src/simulate.py).")
            from .simulate import run_simulation

            run_simulation(cfg)
        log.info("Loading simulated data from %s", sim_expr_path)
        expr = pd.read_csv(sim_expr_path, sep="\t", index_col=0)
        clinical = pd.read_csv(sim_clinical_path, index_col="case_submitter_id")

    expr = expr.loc[:, expr.columns.intersection(clinical.index)]
    clinical = clinical.loc[expr.columns]
    return expr, clinical


# --------------------------------------------------------------------------- #
# Quality control
# --------------------------------------------------------------------------- #
def qc_report(expr: pd.DataFrame) -> dict[str, Any]:
    """Compute per-sample QC metrics and a missingness summary.

    ``expr`` is genes (rows) x samples (columns), linear scale.
    """
    library_size = expr.sum(axis=0, skipna=True)
    genes_detected = (expr > 0).sum(axis=0)
    missing_per_sample = expr.isna().sum(axis=0)
    missing_per_gene = expr.isna().sum(axis=1)

    report = {
        "n_genes": int(expr.shape[0]),
        "n_samples": int(expr.shape[1]),
        "total_missing_values": int(expr.isna().sum().sum()),
        "pct_missing": float(expr.isna().sum().sum() / expr.size * 100),
        "library_size": library_size,
        "genes_detected": genes_detected,
        "missing_per_sample": missing_per_sample,
        "missing_per_gene": missing_per_gene,
        "library_size_summary": library_size.describe().to_dict(),
        "genes_detected_summary": genes_detected.describe().to_dict(),
    }
    log.info(
        "QC: %d genes x %d samples | %.4f%% missing | median library size %.0f | median genes detected %d",
        report["n_genes"], report["n_samples"], report["pct_missing"],
        report["library_size_summary"]["50%"], int(report["genes_detected_summary"]["50%"]),
    )
    return report


def impute_missing(expr: pd.DataFrame) -> pd.DataFrame:
    """Gene-wise median imputation for the small fraction of missing calls."""
    n_missing = expr.isna().sum().sum()
    if n_missing == 0:
        return expr
    log.info("Imputing %d missing values with per-gene median.", n_missing)
    return expr.apply(lambda row: row.fillna(row.median()), axis=1)


# --------------------------------------------------------------------------- #
# Filtering / transformation
# --------------------------------------------------------------------------- #
def filter_low_expression_genes(expr: pd.DataFrame, min_samples_expressing: float = 0.2) -> pd.DataFrame:
    """Drop genes expressed (value > 0) in fewer than the given fraction of samples."""
    frac_expressing = (expr > 0).sum(axis=1) / expr.shape[1]
    keep = frac_expressing >= min_samples_expressing
    log.info(
        "Gene filtering: keeping %d / %d genes expressed in >= %.0f%% of samples.",
        keep.sum(), expr.shape[0], min_samples_expressing * 100,
    )
    return expr.loc[keep]


def log_transform(expr: pd.DataFrame, method: str = "log2") -> pd.DataFrame:
    if method == "log2":
        return np.log2(expr + 1)
    elif method == "log10":
        return np.log10(expr + 1)
    elif method == "ln":
        return np.log1p(expr)
    raise ValueError(f"Unknown log transform method: {method}")


# --------------------------------------------------------------------------- #
# Outlier detection
# --------------------------------------------------------------------------- #
def detect_outlier_samples(
    log_expr: pd.DataFrame, n_pcs: int = 10, z_threshold: float = 3.5
) -> tuple[list[str], pd.Series]:
    """Flag outlier samples via robust Mahalanobis distance in PCA space.

    A sample is flagged if its robust Mahalanobis distance (Minimum
    Covariance Determinant estimator) in the top-``n_pcs`` PCA space
    exceeds ``z_threshold`` robust standard deviations from the median
    distance -- a standard multivariate outlier screen for expression data.
    """
    X = log_expr.T.values  # samples x genes
    X = X - X.mean(axis=0)
    n_pcs = min(n_pcs, min(X.shape) - 1)
    pcs = PCA(n_components=n_pcs, random_state=0).fit_transform(X)

    mcd = MinCovDet(random_state=0).fit(pcs)
    distances = np.sqrt(mcd.mahalanobis(pcs))
    med, mad = np.median(distances), stats.median_abs_deviation(distances, scale="normal")
    robust_z = (distances - med) / (mad if mad > 0 else 1.0)

    flagged = log_expr.columns[robust_z > z_threshold].tolist()
    dist_series = pd.Series(distances, index=log_expr.columns, name="mahalanobis_distance")
    log.info("Outlier detection: flagged %d / %d samples (robust z > %.1f).", len(flagged), X.shape[0], z_threshold)
    return flagged, dist_series


# --------------------------------------------------------------------------- #
# Variance-based feature selection + normalization
# --------------------------------------------------------------------------- #
def compute_gene_variance(log_expr: pd.DataFrame, method: str = "mad") -> pd.Series:
    if method == "mad":
        return log_expr.apply(lambda row: stats.median_abs_deviation(row, scale="normal"), axis=1)
    elif method == "var":
        return log_expr.var(axis=1)
    raise ValueError(f"Unknown variance method: {method}")


def select_variable_genes(log_expr: pd.DataFrame, top_n: int = 2000, method: str = "mad") -> tuple[pd.DataFrame, pd.Series]:
    variance = compute_gene_variance(log_expr, method=method)
    top_genes = variance.sort_values(ascending=False).head(top_n).index
    log.info("Variance-based feature selection: kept top %d / %d genes (method=%s).", len(top_genes), log_expr.shape[0], method)
    return log_expr.loc[top_genes], variance


def zscore_normalize(log_expr: pd.DataFrame) -> pd.DataFrame:
    """Gene-wise z-score standardization (mean 0, sd 1 across samples)."""
    mean = log_expr.mean(axis=1)
    sd = log_expr.std(axis=1).replace(0, 1.0)
    return log_expr.sub(mean, axis=0).div(sd, axis=0)


# --------------------------------------------------------------------------- #
# Batch effect assessment
# --------------------------------------------------------------------------- #
def assess_batch_effect(scaled_expr: pd.DataFrame, batch_labels: pd.Series, n_pcs: int = 5) -> pd.DataFrame:
    """Quantify the fraction of PCA variance associated with a batch variable.

    For each of the top ``n_pcs`` principal components, fits a one-way
    ANOVA of PC score ~ batch and reports the R^2 (eta-squared), i.e. how
    much of that PC's variance is explained by batch membership. This is
    the same diagnostic logic used by tools like ``pvca`` in bulk RNA-seq
    batch-effect QC.
    """
    X = scaled_expr.T.values
    pcs = PCA(n_components=n_pcs, random_state=0).fit_transform(X)
    batch_labels = batch_labels.loc[scaled_expr.columns]

    rows = []
    for i in range(n_pcs):
        groups = [pcs[batch_labels.values == lvl, i] for lvl in batch_labels.unique()]
        f_stat, p_val = stats.f_oneway(*groups)
        ss_between = sum(len(g) * (g.mean() - pcs[:, i].mean()) ** 2 for g in groups)
        ss_total = ((pcs[:, i] - pcs[:, i].mean()) ** 2).sum()
        eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
        rows.append({"PC": f"PC{i+1}", "eta_squared_batch": eta_sq, "f_stat": f_stat, "p_value": p_val})
    return pd.DataFrame(rows)


def correct_batch_mean_centering(log_expr: pd.DataFrame, batch_labels: pd.Series) -> pd.DataFrame:
    """Simple per-batch mean-centering correction (a lightweight ComBat alternative).

    For a portfolio-scale project this linear correction is sufficient to
    demonstrate the concept; production pipelines should prefer ComBat
    (``pycombat``) or ``limma::removeBatchEffect`` for real multi-center data.
    """
    batch_labels = batch_labels.loc[log_expr.columns]
    corrected = log_expr.copy()
    grand_mean = log_expr.mean(axis=1)
    for lvl in batch_labels.unique():
        cols = batch_labels[batch_labels == lvl].index
        batch_mean = log_expr[cols].mean(axis=1)
        corrected[cols] = log_expr[cols].sub(batch_mean, axis=0).add(grand_mean, axis=0)
    return corrected


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_preprocessing(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    pp_cfg = cfg["preprocessing"]
    processed_dir = resolve_path(cfg, "processed_dir")
    reports_dir = resolve_path(cfg, "reports_dir")

    expr, clinical = load_raw_data(cfg)
    qc_before = qc_report(expr)

    expr = impute_missing(expr)
    expr = filter_low_expression_genes(expr, pp_cfg["min_samples_expressing"])
    log_expr = log_transform(expr, pp_cfg["log_transform"])

    outliers, distances = detect_outlier_samples(log_expr, z_threshold=pp_cfg["outlier_zscore_threshold"])

    variable_expr, gene_variance = select_variable_genes(
        log_expr, top_n=pp_cfg["top_n_variable_genes"], method=pp_cfg["variance_method"]
    )
    scaled_expr = zscore_normalize(variable_expr) if pp_cfg["scale"] else variable_expr

    batch_col = "sequencing_plate" if "sequencing_plate" in clinical.columns else None
    batch_report = None
    if batch_col is not None:
        batch_report = assess_batch_effect(scaled_expr, clinical[batch_col])
        batch_report.to_csv(reports_dir / "batch_effect_pca_variance.csv", index=False)

    log_expr.to_csv(processed_dir / "log_expression_full.tsv", sep="\t")
    variable_expr.to_csv(processed_dir / "log_expression_top_variable.tsv", sep="\t")
    scaled_expr.to_csv(processed_dir / "scaled_expression_top_variable.tsv", sep="\t")
    gene_variance.sort_values(ascending=False).to_csv(reports_dir / "gene_variance_ranking.csv", header=["variance"])
    clinical.to_csv(processed_dir / "clinical_processed.csv")

    summary = {
        "n_genes_raw": qc_before["n_genes"],
        "n_samples_raw": qc_before["n_samples"],
        "pct_missing_raw": qc_before["pct_missing"],
        "n_genes_after_filtering": int(log_expr.shape[0]),
        "n_variable_genes_selected": int(variable_expr.shape[0]),
        "n_outlier_samples_flagged": len(outliers),
        "outlier_sample_ids": outliers,
    }
    with open(reports_dir / "preprocessing_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    log.info("Preprocessing complete. Summary: %s", summary)
    return {
        "expr_raw": expr,
        "log_expr_full": log_expr,
        "variable_expr": variable_expr,
        "scaled_expr": scaled_expr,
        "clinical": clinical,
        "qc_before": qc_before,
        "outliers": outliers,
        "distances": distances,
        "gene_variance": gene_variance,
        "batch_report": batch_report,
        "summary": summary,
    }


if __name__ == "__main__":
    run_preprocessing()
