"""End-to-end orchestration script: preprocessing through survival analysis.

Runs the complete study in one pass and writes every figure/table used in
the README and notebooks. This is the same logic the notebooks walk through
interactively; the script exists so the whole analysis can be reproduced
non-interactively with a single command:

    python scripts/run_pipeline.py

Runtime on the default synthetic cohort (500 samples x 2,000 variable genes):
approximately 2-4 minutes on a laptop CPU.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src import clustering as cl
from src import enrichment as en
from src import survival as sv
from src import visualization as viz
from src.config import load_config, resolve_path
from src.logging_config import get_logger, setup_logging
from src.preprocess import run_preprocessing

log = get_logger(__name__)

SUBTYPE_PALETTE = viz.SUBTYPE_PALETTE


def main(cfg: dict | None = None) -> None:
    setup_logging()
    cfg = cfg or load_config()
    t0 = time.time()

    figures_dir = resolve_path(cfg, "figures_dir")
    results_dir = resolve_path(cfg, "results_dir")
    reports_dir = resolve_path(cfg, "reports_dir")
    for sub in ["eda", "dimensionality_reduction", "clustering", "biological", "survival"]:
        (figures_dir / sub).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Preprocessing
    # ------------------------------------------------------------------ #
    log.info("=" * 80)
    log.info("STAGE 1/6: Preprocessing")
    pp = run_preprocessing(cfg)
    log_expr_full = pp["log_expr_full"]
    variable_expr = pp["variable_expr"]
    scaled_expr = pp["scaled_expr"]
    clinical = pp["clinical"]
    has_ground_truth = "true_subtype" in clinical.columns
    truth = clinical["true_subtype"] if has_ground_truth else None

    viz.plot_library_size_distribution(pp["qc_before"]["library_size"], figures_dir / "eda" / "library_size_distribution.png")
    viz.plot_missingness(pp["qc_before"]["missing_per_sample"], pp["qc_before"]["missing_per_gene"], figures_dir / "eda" / "missingness.png")
    viz.plot_gene_variance_distribution(pp["gene_variance"], cfg["preprocessing"]["top_n_variable_genes"], figures_dir / "eda" / "gene_variance_distribution.png")
    viz.plot_sample_correlation_heatmap(variable_expr, annotation=truth, save_path=figures_dir / "eda" / "sample_correlation_heatmap.png")
    marker_example_genes = ["ESR1", "ERBB2", "MKI67", "KRT5"]
    marker_example_genes = [g for g in marker_example_genes if g in log_expr_full.index]
    if marker_example_genes:
        viz.plot_feature_distributions(log_expr_full, marker_example_genes, figures_dir / "eda" / "marker_gene_distributions.png")

    hopkins = cl.hopkins_statistic(scaled_expr.T.values)
    log.info("Hopkins statistic (cluster tendency): %.3f (0.5 = random, 1.0 = highly clusterable)", hopkins)

    # ------------------------------------------------------------------ #
    # 2. Dimensionality reduction
    # ------------------------------------------------------------------ #
    log.info("=" * 80)
    log.info("STAGE 2/6: Dimensionality reduction (PCA, t-SNE, UMAP)")
    dr_cfg = cfg["dimensionality_reduction"]
    pca_result = cl.run_pca(scaled_expr, n_components=dr_cfg["pca"]["n_components"])
    scores, loadings, evr = pca_result["scores"], pca_result["loadings"], pca_result["explained_variance_ratio"]

    viz.plot_scree(evr, save_path=figures_dir / "dimensionality_reduction" / "pca_scree.png")
    hue = truth if has_ground_truth else None
    viz.plot_pc_scatter(scores, "PC1", "PC2", hue=hue, title="PCA: PC1 vs PC2", save_path=figures_dir / "dimensionality_reduction" / "pca_pc1_pc2.png", palette=SUBTYPE_PALETTE)
    viz.plot_pc_scatter(scores, "PC1", "PC3", hue=hue, title="PCA: PC1 vs PC3", save_path=figures_dir / "dimensionality_reduction" / "pca_pc1_pc3.png", palette=SUBTYPE_PALETTE)

    for pc in ["PC1", "PC2"]:
        top_loadings = cl.top_pc_loadings(loadings, pc, top_n=25)
        top_loadings.to_csv(results_dir / f"pca_top_loadings_{pc}.csv")

    tsne_embeddings = {}
    for perplexity in dr_cfg["tsne"]["perplexities"]:
        log.info("t-SNE perplexity=%d ...", perplexity)
        emb = cl.run_tsne(scaled_expr, perplexity=perplexity)
        tsne_embeddings[f"perplexity={perplexity}"] = emb
        emb.to_csv(results_dir / f"tsne_embedding_perplexity{perplexity}.csv")
    if has_ground_truth:
        viz.plot_embedding_grid(tsne_embeddings, truth, title="t-SNE across perplexities", save_path=figures_dir / "dimensionality_reduction" / "tsne_perplexity_comparison.png", palette=SUBTYPE_PALETTE)

    umap_embeddings = {}
    for n_neighbors in dr_cfg["umap"]["n_neighbors_grid"]:
        log.info("UMAP n_neighbors=%d ...", n_neighbors)
        emb = cl.run_umap(scaled_expr, n_neighbors=n_neighbors, min_dist=dr_cfg["umap"]["default_min_dist"])
        umap_embeddings[f"n_neighbors={n_neighbors}"] = emb
    if has_ground_truth:
        viz.plot_embedding_grid(umap_embeddings, truth, title="UMAP across n_neighbors", save_path=figures_dir / "dimensionality_reduction" / "umap_n_neighbors_comparison.png", palette=SUBTYPE_PALETTE)

    min_dist_embeddings = {}
    for min_dist in dr_cfg["umap"]["min_dist_grid"]:
        emb = cl.run_umap(scaled_expr, n_neighbors=dr_cfg["umap"]["default_n_neighbors"], min_dist=min_dist)
        min_dist_embeddings[f"min_dist={min_dist}"] = emb
    if has_ground_truth:
        viz.plot_embedding_grid(min_dist_embeddings, truth, title="UMAP across min_dist", save_path=figures_dir / "dimensionality_reduction" / "umap_min_dist_comparison.png", palette=SUBTYPE_PALETTE)

    default_key = f"n_neighbors={dr_cfg['umap']['default_n_neighbors']}"
    if default_key in umap_embeddings:
        default_umap = umap_embeddings[default_key]
    else:
        default_umap = cl.run_umap(
            scaled_expr, n_neighbors=dr_cfg["umap"]["default_n_neighbors"], min_dist=dr_cfg["umap"]["default_min_dist"]
        )
    default_umap.to_csv(results_dir / "umap_embedding_default.csv")
    viz.plot_pc_scatter(default_umap, "UMAP1", "UMAP2", hue=hue, title="UMAP (default parameters)", save_path=figures_dir / "dimensionality_reduction" / "umap_default.png", palette=SUBTYPE_PALETTE)

    # ------------------------------------------------------------------ #
    # 3. Clustering
    # ------------------------------------------------------------------ #
    log.info("=" * 80)
    log.info("STAGE 3/6: Clustering")
    X = scaled_expr.T.values
    clus_cfg = cfg["clustering"]

    sweep = cl.kmeans_sweep(X, tuple(clus_cfg["kmeans"]["k_range"]))
    sweep.to_csv(results_dir / "kmeans_sweep.csv", index=False)
    viz.plot_elbow_silhouette(sweep, save_path=figures_dir / "clustering" / "kmeans_elbow_silhouette.png")

    chosen_k = clus_cfg["chosen_k"]
    km_labels = cl.fit_kmeans(X, chosen_k)
    km_labels_series = pd.Series(km_labels, index=scaled_expr.columns, name="kmeans_cluster")
    km_labels_series.to_csv(results_dir / "cluster_assignments_kmeans.csv")
    viz.plot_silhouette_samples_plot(X, km_labels, save_path=figures_dir / "clustering" / "kmeans_silhouette_samples.png")

    cluster_label_map = {i: f"Cluster {i}" for i in sorted(set(km_labels))}
    km_named = km_labels_series.map(cluster_label_map)
    viz.plot_pc_scatter(scores, "PC1", "PC2", hue=km_named, title=f"K-Means (k={chosen_k}) on PCA plane", save_path=figures_dir / "clustering" / "kmeans_on_pca.png")

    linkage_comparison = cl.compare_linkages(X, clus_cfg["hierarchical"]["n_clusters"], clus_cfg["hierarchical"]["linkages"])
    linkage_comparison.to_csv(results_dir / "hierarchical_linkage_comparison.csv", index=False)

    Z_ward = cl.hierarchical_linkage(X, method="ward")
    viz.plot_dendrogram(Z_ward, save_path=figures_dir / "clustering" / "dendrogram_ward.png")
    hc_labels = cl.hierarchical_clusters(Z_ward, clus_cfg["hierarchical"]["n_clusters"])
    hc_labels_series = pd.Series(hc_labels, index=scaled_expr.columns, name="hierarchical_cluster")
    hc_labels_series.to_csv(results_dir / "cluster_assignments_hierarchical.csv")

    marker_genes_for_heatmap = [g for g in [
        "ESR1", "PGR", "GATA3", "FOXA1", "ERBB2", "GRB7", "KRT5", "KRT14", "EGFR", "MKI67", "CCNB1", "BIRC5",
    ] if g in scaled_expr.index]
    if marker_genes_for_heatmap:
        viz.plot_cluster_heatmap(scaled_expr, km_labels_series, genes=marker_genes_for_heatmap, save_path=figures_dir / "clustering" / "cluster_marker_heatmap.png")

    X_density = pca_result["scores"].iloc[:, :30].values
    method_comparison = cl.compare_all_methods(X, clus_cfg, X_density=X_density)
    method_comparison.to_csv(results_dir / "clustering_method_comparison.csv", index=False)

    if has_ground_truth:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        ari = adjusted_rand_score(truth.loc[scaled_expr.columns], km_labels)
        nmi = normalized_mutual_info_score(truth.loc[scaled_expr.columns], km_labels)
        with open(reports_dir / "ground_truth_recovery.json", "w") as fh:
            json.dump({"adjusted_rand_index": ari, "normalized_mutual_info": nmi}, fh, indent=2)
        log.info("Ground-truth subtype recovery: ARI=%.3f, NMI=%.3f", ari, nmi)

    # ------------------------------------------------------------------ #
    # 4. Biological interpretation: DE + enrichment
    # ------------------------------------------------------------------ #
    log.info("=" * 80)
    log.info("STAGE 4/6: Differential expression + functional enrichment")
    de_results = en.run_de_for_all_clusters(log_expr_full, km_labels_series)
    for cluster_id, de_df in de_results.items():
        de_df.to_csv(results_dir / f"de_cluster_{cluster_id}_vs_rest.csv", index=False)
        viz.plot_volcano(
            de_df, title=f"Cluster {cluster_id} vs. rest",
            fdr_thresh=cfg["differential_expression"]["fdr_threshold"],
            fc_thresh=cfg["differential_expression"]["log2fc_threshold"],
            save_path=figures_dir / "biological" / f"volcano_cluster_{cluster_id}.png",
        )

    enrich_cfg = cfg["enrichment"]
    enrichment_results = en.run_enrichment_for_all_clusters(
        de_results, gene_set_libraries=enrich_cfg["gene_set_libraries"], top_n=enrich_cfg["top_n_genes_per_cluster"],
    )
    for cluster_id, lib_results in enrichment_results.items():
        for lib, df in lib_results.items():
            safe_lib = lib.replace(" ", "_")
            df.to_csv(results_dir / f"enrichment_cluster_{cluster_id}_{safe_lib}.csv", index=False)
            if len(df):
                viz.plot_enrichment_dotplot(df, title=f"Cluster {cluster_id}: {lib}", save_path=figures_dir / "biological" / f"enrichment_dotplot_cluster{cluster_id}_{safe_lib}.png")
                viz.plot_enrichment_barplot(df, title=f"Cluster {cluster_id}: {lib}", save_path=figures_dir / "biological" / f"enrichment_barplot_cluster{cluster_id}_{safe_lib}.png")

    # ------------------------------------------------------------------ #
    # 5. Clinical + survival analysis
    # ------------------------------------------------------------------ #
    log.info("=" * 80)
    log.info("STAGE 5/6: Clinical comparison + survival analysis")
    surv_cfg = cfg["survival"]
    clinical_summary = sv.summarize_clinical_by_cluster(clinical, km_labels_series)
    clinical_summary.to_csv(results_dir / "clinical_summary_by_cluster.csv")

    association_tests = sv.clinical_association_tests(clinical, km_labels_series)
    association_tests.to_csv(results_dir / "clinical_association_tests.csv", index=False)

    km_fitters = sv.fit_km_by_cluster(clinical, km_labels_series, surv_cfg["time_col"], surv_cfg["event_col"])
    viz.plot_km_curves(km_fitters, title=f"Kaplan-Meier survival by K-Means cluster (k={chosen_k})", save_path=figures_dir / "survival" / "km_curves_by_cluster.png")

    global_lr = sv.global_logrank_test(clinical, km_labels_series, surv_cfg["time_col"], surv_cfg["event_col"])
    pairwise_lr = sv.pairwise_logrank_tests(clinical, km_labels_series, surv_cfg["time_col"], surv_cfg["event_col"])
    pairwise_lr.to_csv(results_dir / "survival_pairwise_logrank.csv", index=False)
    with open(reports_dir / "survival_global_logrank.json", "w") as fh:
        json.dump({"test_statistic": float(global_lr.test_statistic), "p_value": float(global_lr.p_value)}, fh, indent=2)

    try:
        cph = sv.fit_cox_model(clinical, km_labels_series, surv_cfg["time_col"], surv_cfg["event_col"], adjust_for=["age_at_index"])
        cph.summary.to_csv(results_dir / "cox_model_summary.csv")
    except Exception as exc:
        log.warning("Cox model fitting failed (non-fatal): %s", exc)

    if has_ground_truth:
        km_truth_fitters = sv.fit_km_by_cluster(clinical, truth, surv_cfg["time_col"], surv_cfg["event_col"])
        viz.plot_km_curves(km_truth_fitters, title="Kaplan-Meier survival by ground-truth PAM50-like subtype", save_path=figures_dir / "survival" / "km_curves_by_true_subtype.png", palette=SUBTYPE_PALETTE)

    # ------------------------------------------------------------------ #
    # 6. Explainable AI: ranked biomarker table
    # ------------------------------------------------------------------ #
    log.info("=" * 80)
    log.info("STAGE 6/6: Explainable AI - ranked biomarker table")
    biomarker_table = cl.biomarker_ranking(scaled_expr, km_labels_series, loadings, de_results)
    biomarker_table.to_csv(results_dir / "ranked_biomarker_table.csv", index=False)
    viz.plot_biomarker_heatmap(scaled_expr, biomarker_table, km_labels_series, top_n_per_cluster=8, save_path=figures_dir / "biological" / "top_biomarkers_heatmap.png")

    elapsed = time.time() - t0
    log.info("=" * 80)
    log.info("Pipeline complete in %.1f seconds. Figures -> %s | Results -> %s", elapsed, figures_dir, results_dir)


if __name__ == "__main__":
    main()
