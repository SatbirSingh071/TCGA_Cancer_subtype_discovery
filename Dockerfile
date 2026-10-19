# =============================================================================
# tcga-cancer-subtype-discovery
#
# Build:  docker build -t tcga-subtype-discovery .
# Run:    docker run --rm \
#           -v "$(pwd)/figures:/app/figures" \
#           -v "$(pwd)/results:/app/results" \
#           -v "$(pwd)/reports:/app/reports" \
#           tcga-subtype-discovery
#
# Runs the full pipeline (preprocessing -> dimensionality reduction ->
# clustering -> differential expression/enrichment -> survival analysis) on
# the synthetic demo cohort and writes every figure/table to the mounted
# host directories.
# =============================================================================
FROM python:3.11-slim

LABEL org.opencontainers.image.title="tcga-cancer-subtype-discovery"
LABEL org.opencontainers.image.description="Unsupervised discovery of molecular cancer subtypes from TCGA RNA-seq data"
LABEL org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies for scientific Python wheels (scipy/umap/hdbscan) that
# occasionally need to compile on platforms without prebuilt wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/raw data/processed data/metadata figures results reports logs

# Non-root user for defense-in-depth.
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

VOLUME ["/app/figures", "/app/results", "/app/reports", "/app/data"]

ENTRYPOINT ["python", "scripts/run_pipeline.py"]
