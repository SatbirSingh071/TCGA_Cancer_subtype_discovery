"""Command-line interface for the TCGA cancer subtype discovery pipeline.

Examples
--------
Run the entire pipeline end-to-end on the simulated demo cohort::

    python -m src.cli run-all --config config.yaml

Run a single stage::

    python -m src.cli download --config config.yaml
    python -m src.cli simulate --config config.yaml
    python -m src.cli preprocess --config config.yaml
    python -m src.cli cluster --config config.yaml --k 4
"""
from __future__ import annotations

import click

from .config import load_config, override, resolve_path
from .logging_config import get_logger, setup_logging

log = get_logger(__name__)


@click.group()
@click.option("--config", "config_path", default=None, help="Path to config.yaml (defaults to repo root).")
@click.option("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
@click.pass_context
def cli(ctx: click.Context, config_path: str | None, log_level: str) -> None:
    """TCGA Cancer Subtype Discovery -- unsupervised ML pipeline CLI."""
    setup_logging(level=log_level)
    ctx.ensure_object(dict)
    ctx.obj["cfg"] = load_config(config_path)


@cli.command()
@click.pass_context
def download(ctx: click.Context) -> None:
    """Download real RNA-seq + clinical data from the GDC API."""
    from .download import run_download

    run_download(ctx.obj["cfg"])


@cli.command()
@click.pass_context
def simulate(ctx: click.Context) -> None:
    """Generate the biologically-motivated synthetic demo cohort."""
    from .simulate import run_simulation

    run_simulation(ctx.obj["cfg"])


@cli.command()
@click.option("--top-n-genes", type=int, default=None, help="Override top_n_variable_genes.")
@click.pass_context
def preprocess(ctx: click.Context, top_n_genes: int | None) -> None:
    """QC, filtering, log-transform, outlier detection, variance selection, scaling."""
    from .preprocess import run_preprocessing

    cfg = ctx.obj["cfg"]
    if top_n_genes:
        cfg = override(cfg, **{"preprocessing.top_n_variable_genes": top_n_genes})
    run_preprocessing(cfg)


@cli.command()
@click.option("--k", type=int, default=None, help="Number of clusters for the chosen method (overrides config).")
@click.pass_context
def cluster(ctx: click.Context, k: int | None) -> None:
    """Run PCA/t-SNE/UMAP + KMeans/Hierarchical/DBSCAN/HDBSCAN/Spectral and write results/."""
    import pandas as pd

    from . import clustering as cl
    from .preprocess import run_preprocessing

    cfg = ctx.obj["cfg"]
    if k:
        cfg = override(cfg, **{"clustering.chosen_k": k})
    results_dir = resolve_path(cfg, "results_dir")

    pp = run_preprocessing(cfg)
    scaled_expr = pp["scaled_expr"]
    X = scaled_expr.T.values

    pca_result = cl.run_pca(scaled_expr, n_components=cfg["dimensionality_reduction"]["pca"]["n_components"])
    pca_result["scores"].to_csv(results_dir / "pca_scores.csv")

    sweep = cl.kmeans_sweep(X, tuple(cfg["clustering"]["kmeans"]["k_range"]))
    sweep.to_csv(results_dir / "kmeans_sweep.csv", index=False)

    chosen_k = cfg["clustering"]["chosen_k"]
    labels = cl.fit_kmeans(X, chosen_k)
    labels_series = pd.Series(labels, index=scaled_expr.columns, name="cluster")
    labels_series.to_csv(results_dir / "cluster_assignments.csv")

    comparison = cl.compare_all_methods(X, cfg["clustering"])
    comparison.to_csv(results_dir / "clustering_method_comparison.csv", index=False)

    log.info("Clustering complete. k=%d chosen. Method comparison:\n%s", chosen_k, comparison.to_string())


@cli.command(name="run-all")
@click.pass_context
def run_all(ctx: click.Context) -> None:
    """Run the full pipeline: preprocess -> cluster -> DE/enrichment -> survival."""
    cfg = ctx.obj["cfg"]
    from scripts.run_pipeline import main as run_pipeline_main

    run_pipeline_main(cfg)


def main() -> None:
    cli(obj={})


if __name__ == "__main__":
    main()
