"""Biologically-motivated synthetic TCGA-BRCA dataset generator.

Why this module exists
-----------------------
A full-scale real GDC pull of ~1,100 TCGA-BRCA RNA-seq samples is a
multi-hundred-megabyte, multi-hour workload (see :mod:`src.download`, which
*is* a real, working GDC API client and is exercised on a small live subset
as part of this project's test suite). For a reproducible, instantly
re-runnable portfolio artifact, this module generates a synthetic cohort
that reproduces the well-established molecular biology of breast cancer
intrinsic subtypes (PAM50: Basal-like, Luminal A, Luminal B, HER2-enriched)
so that every downstream step -- QC, batch-effect diagnostics, dimensionality
reduction, clustering, differential expression, functional enrichment, and
survival analysis -- operates on data with genuine, known ground truth. This
lets the write-up validate *whether the unsupervised pipeline recovers a
signal that is known, by construction, to be there* -- exactly the same
logic used to benchmark clustering methods on TCGA data in the literature
(e.g. recovering PAM50 calls from unsupervised clustering of expression).

All gene symbols used are real human HGNC gene symbols (marker genes are the
actual PAM50 / clinically-used breast cancer markers; background "filler"
genes are sampled from a real KEGG_2021_Human gene symbol pool fetched from
Enrichr), so downstream GO/KEGG/Reactome enrichment queries in
:mod:`src.enrichment` return real, matchable annotations.

The simulator injects realistic nuisance structure researchers must handle
in practice: a sequencing-batch/plate effect, a handful of low-library-size
and globally-scaled outlier samples, and missing values -- all of which are
detected and addressed in :mod:`src.preprocess`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import REPO_ROOT, load_config, resolve_path
from .logging_config import get_logger

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Real PAM50-style marker genes (actual HGNC symbols) driving each subtype.
# Signs indicate the direction of the subtype-specific effect on log2 expression.
# Biological references: Parker et al. 2009 (PAM50), Perou et al. 2000 Nature.
# --------------------------------------------------------------------------- #
SUBTYPE_MARKERS: dict[str, dict[str, float]] = {
    "Basal-like": {
        "KRT5": 2.8, "KRT14": 2.6, "KRT17": 2.4, "EGFR": 2.0, "FOXC1": 2.2,
        "MIA": 1.8, "SFRP1": 1.6, "MYC": 1.2, "CDKN2A": 1.4, "TP53": 0.8,
        "MKI67": 1.6, "CCNB1": 1.4, "AURKA": 1.3, "BIRC5": 1.5,
        "ESR1": -3.0, "PGR": -2.6, "GATA3": -2.4, "FOXA1": -2.2, "XBP1": -1.6,
        "BCL2": -1.8, "ERBB2": -0.6,
    },
    "Luminal A": {
        "ESR1": 2.8, "PGR": 2.6, "GATA3": 2.2, "FOXA1": 2.0, "XBP1": 1.6,
        "BCL2": 1.8, "MLPH": 1.6, "NAT1": 1.4, "SLC39A6": 1.2,
        "MKI67": -1.4, "CCNB1": -1.2, "AURKA": -1.1, "BIRC5": -1.3, "MYBL2": -1.2,
        "KRT5": -1.6, "KRT14": -1.4, "EGFR": -1.2, "ERBB2": -0.8,
    },
    "Luminal B": {
        "ESR1": 1.6, "PGR": 0.6, "GATA3": 1.2, "FOXA1": 1.0,
        "MKI67": 2.2, "CCNB1": 2.0, "AURKA": 1.8, "BIRC5": 2.0, "MYBL2": 2.1,
        "CCNE1": 1.6, "MYC": 1.4,
        "KRT5": -1.2, "KRT14": -1.0,
    },
    "HER2-enriched": {
        "ERBB2": 3.2, "GRB7": 2.8, "STARD3": 2.4, "PGAP3": 2.0, "MIEN1": 1.8,
        "MYC": 1.2, "MKI67": 1.4, "CCNB1": 1.2,
        "ESR1": -2.0, "PGR": -2.2, "GATA3": -1.4, "FOXA1": -1.2,
        "KRT5": -0.6,
    },
}

SUBTYPE_PREVALENCE = {
    "Luminal A": 0.40,
    "Luminal B": 0.20,
    "Basal-like": 0.20,
    "HER2-enriched": 0.20,
}

# Relative hazard multipliers used to simulate subtype-dependent survival
# (Basal-like and HER2-enriched carry worse prognosis than Luminal A, in
# line with the well-documented TCGA/METABRIC survival literature).
SUBTYPE_HAZARD_RATIO = {
    "Luminal A": 0.55,
    "Luminal B": 0.90,
    "HER2-enriched": 1.35,
    "Basal-like": 1.75,
}

GENE_POOL_PATH = REPO_ROOT / "data" / "metadata" / "real_gene_symbol_pool.txt"


def _load_gene_pool() -> list[str]:
    if GENE_POOL_PATH.exists():
        genes = GENE_POOL_PATH.read_text(encoding="utf-8").splitlines()
        return [g.strip() for g in genes if g.strip()]
    log.warning("Real gene symbol pool not found at %s; falling back to placeholder names.", GENE_POOL_PATH)
    return [f"GENE{i}" for i in range(20000)]


def simulate_cohort(cfg: dict[str, Any] | None = None, seed: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a synthetic TCGA-BRCA-like expression matrix + clinical table.

    Returns
    -------
    expr : DataFrame, shape (n_genes, n_samples)
        Row index = gene symbol. Values are pseudo-TPM (non-negative, linear scale).
    clinical : DataFrame, shape (n_samples, ...)
        One row per sample, indexed by the same sample IDs as ``expr`` columns.
    """
    cfg = cfg or load_config()
    sim_cfg = cfg["simulate"]
    rng = np.random.default_rng(seed if seed is not None else sim_cfg["seed"])

    n_samples = sim_cfg["n_samples"]
    n_genes = sim_cfg["n_genes"]
    noise_sd = sim_cfg["noise_sd"]
    subtypes = list(SUBTYPE_PREVALENCE.keys())

    # --- gene universe: guarantee all marker genes are present, pad with real background genes ---
    marker_genes = sorted({g for markers in SUBTYPE_MARKERS.values() for g in markers})
    pool = [g for g in _load_gene_pool() if g not in marker_genes]
    rng.shuffle(pool)
    n_background = max(n_genes - len(marker_genes), 0)
    genes = marker_genes + pool[:n_background]
    genes = genes[:n_genes] if len(genes) > n_genes else genes
    n_genes = len(genes)
    gene_index = {g: i for i, g in enumerate(genes)}

    # --- sample -> subtype assignment ---
    probs = np.array([SUBTYPE_PREVALENCE[s] for s in subtypes])
    probs = probs / probs.sum()
    sample_subtype = rng.choice(subtypes, size=n_samples, p=probs)
    sample_ids = [f"SIM-{cfg['project']['cancer_type'].split('-')[-1]}-{i:04d}" for i in range(n_samples)]

    # --- baseline log2 expression per gene (housekeeping-like variation across genes) ---
    baseline = rng.normal(loc=6.0, scale=1.8, size=n_genes)
    baseline = np.clip(baseline, 0.5, None)

    # --- latent co-expression structure -------------------------------------------
    # Real transcriptional subtypes are not defined by a handful of curated marker
    # genes alone -- dozens of curated markers (PAM50) are representative flags for
    # much larger coordinated transcriptional programs (proliferation, ER signaling,
    # HER2 amplicon neighbors, basal cytoskeletal programs) spanning hundreds of
    # co-regulated genes. We model that explicitly: each subtype gets its own latent
    # "module" of ~5% of genes that co-vary with subtype membership (in addition to
    # the curated marker genes injected below), plus several subtype-independent
    # programs representing real but nuisance biological variation (e.g. immune
    # infiltration, stromal content) that a good pipeline must see past.
    n_subtype_programs = len(subtypes)
    n_noise_programs = 4
    n_programs = n_subtype_programs + n_noise_programs
    module_sparsity = 0.05
    gene_loadings = rng.normal(0, 1, size=(n_genes, n_programs)) * (rng.random((n_genes, n_programs)) < module_sparsity)

    program_activity = rng.normal(0, 1.0, size=(n_samples, n_programs))
    subtype_program_effect = 2.2
    for i, subtype in enumerate(subtypes):
        mask = sample_subtype == subtype
        program_activity[mask, i] += subtype_program_effect

    correlated_component = program_activity @ gene_loadings.T  # (n_samples, n_genes)

    log_expr = np.tile(baseline, (n_samples, 1)) + correlated_component
    log_expr += rng.normal(0, noise_sd, size=(n_samples, n_genes))

    # --- inject curated PAM50-style marker gene effects (strong, gene-specific signal) ---
    for subtype, markers in SUBTYPE_MARKERS.items():
        mask = sample_subtype == subtype
        for gene, effect in markers.items():
            gi = gene_index[gene]
            # per-sample jitter around the canonical effect size for realism
            jitter = rng.normal(effect, 0.35, size=mask.sum())
            log_expr[mask, gi] += jitter

    # --- batch / sequencing-plate effect (nuisance structure for QC section) ---
    plates = rng.choice(["Plate_A", "Plate_B", "Plate_C"], size=n_samples, p=[0.5, 0.3, 0.2])
    plate_effect_genes = rng.choice(n_genes, size=max(int(0.03 * n_genes), 10), replace=False)
    for plate, shift in zip(["Plate_A", "Plate_B", "Plate_C"], [0.0, 0.6, -0.5]):
        mask = plates == plate
        log_expr[np.ix_(mask, plate_effect_genes)] += shift

    # --- residual library-size effect + a few low-depth / globally-scaled outlier samples ---
    # (kept small: TPM-like units are already library-size normalized, so only a
    # small residual technical shift remains -- unlike the much larger batch/plate
    # effect below, which is deliberately gene-subset-specific rather than global)
    lib_size_factor = rng.normal(0, 0.08, size=n_samples)
    log_expr += lib_size_factor[:, None]

    n_outliers = max(int(0.02 * n_samples), 2)
    outlier_idx = rng.choice(n_samples, size=n_outliers, replace=False)
    log_expr[outlier_idx] += rng.choice([-3.5, 3.5], size=n_outliers)[:, None]

    # --- convert to pseudo-TPM (non-negative, linear scale) ---
    tpm = np.clip(2 ** log_expr - 1, 0, None)

    # --- sprinkle a small amount of missingness (dropout / masked low-quality calls) ---
    missing_mask = rng.random(tpm.shape) < 0.001
    tpm[missing_mask] = np.nan

    expr = pd.DataFrame(tpm.T, index=genes, columns=sample_ids)
    expr.index.name = "gene_name"

    # ----------------------------------------------------------------- #
    # Clinical metadata, correlated with subtype
    # ----------------------------------------------------------------- #
    age = np.clip(rng.normal(58, 12, size=n_samples), 26, 90).round(1)
    gender = rng.choice(["female", "male"], size=n_samples, p=[0.99, 0.01])

    stage_levels = ["Stage I", "Stage IA", "Stage IIA", "Stage IIB", "Stage IIIA", "Stage IIIC", "Stage IV"]
    stage_probs_by_subtype = {
        "Luminal A": [0.18, 0.20, 0.28, 0.16, 0.10, 0.05, 0.03],
        "Luminal B": [0.12, 0.15, 0.28, 0.20, 0.13, 0.08, 0.04],
        "HER2-enriched": [0.08, 0.10, 0.24, 0.24, 0.18, 0.11, 0.05],
        "Basal-like": [0.06, 0.09, 0.22, 0.25, 0.20, 0.13, 0.05],
    }
    stage = np.array(
        [rng.choice(stage_levels, p=stage_probs_by_subtype[s]) for s in sample_subtype]
    )

    # Weibull proportional-hazards survival simulation
    shape_k = 1.3
    scale_lambda = 2200.0  # days
    hazard_ratio = np.array([SUBTYPE_HAZARD_RATIO[s] for s in sample_subtype])
    u = rng.random(n_samples)
    event_time = scale_lambda * (-np.log(u) / hazard_ratio) ** (1 / shape_k)

    admin_censoring_time = rng.normal(2600, 500, size=n_samples).clip(400, None)
    days_to_event = np.minimum(event_time, admin_censoring_time).round(0)
    event_observed = (event_time <= admin_censoring_time).astype(int)
    vital_status = np.where(event_observed == 1, "Dead", "Alive")

    tumor_grade = np.array(
        [rng.choice(["G1", "G2", "G3"], p=[0.5, 0.35, 0.15]) if s == "Luminal A"
         else rng.choice(["G1", "G2", "G3"], p=[0.1, 0.35, 0.55])
         for s in sample_subtype]
    )

    clinical = pd.DataFrame(
        {
            "case_submitter_id": sample_ids,
            "true_subtype": sample_subtype,
            "sequencing_plate": plates,
            "age_at_index": age,
            "gender": gender,
            "ajcc_pathologic_stage": stage,
            "tumor_grade": tumor_grade,
            "vital_status": vital_status,
            "days_to_event": days_to_event,
            "event_observed": event_observed,
        }
    ).set_index("case_submitter_id")

    log.info(
        "Simulated cohort: %d samples x %d genes | subtypes: %s",
        n_samples, n_genes, clinical["true_subtype"].value_counts().to_dict(),
    )
    return expr, clinical


def run_simulation(cfg: dict[str, Any] | None = None) -> tuple[Path, Path]:
    """Simulate a cohort and write it to ``data/raw`` / ``data/metadata``."""
    cfg = cfg or load_config()
    raw_dir = resolve_path(cfg, "processed_dir").parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = resolve_path(cfg, "metadata_dir")

    expr, clinical = simulate_cohort(cfg)

    cancer_type = cfg["project"]["cancer_type"]
    expr_path = raw_dir / f"{cancer_type}_simulated_expression_matrix.tsv"
    clinical_path = metadata_dir / f"{cancer_type}_simulated_clinical.csv"

    expr.to_csv(expr_path, sep="\t")
    clinical.to_csv(clinical_path)

    log.info("Wrote simulated expression matrix: %s", expr_path)
    log.info("Wrote simulated clinical metadata: %s", clinical_path)
    return expr_path, clinical_path


if __name__ == "__main__":
    run_simulation()
