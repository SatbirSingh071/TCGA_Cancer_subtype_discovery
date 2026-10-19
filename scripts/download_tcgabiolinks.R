#!/usr/bin/env Rscript
# =============================================================================
# download_tcgabiolinks.R
#
# Equivalent full-scale data acquisition using TCGAbiolinks, the standard
# Bioconductor package used across the TCGA/GDC research community. Prefer
# this route for a production-scale pull of an entire project (e.g. all
# ~1,100 TCGA-BRCA cases); use src/download.py for lightweight, dependency-free
# Python access or CI smoke tests.
# =============================================================================

if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
if (!requireNamespace("TCGAbiolinks", quietly = TRUE))
    BiocManager::install("TCGAbiolinks")

library(TCGAbiolinks)
library(SummarizedExperiment)

project   <- "TCGA-BRCA"
out_dir   <- file.path("data", "raw")
meta_dir  <- file.path("data", "metadata")
dir.create(out_dir,  recursive = TRUE, showWarnings = FALSE)
dir.create(meta_dir, recursive = TRUE, showWarnings = FALSE)

message("Querying GDC for ", project, " RNA-seq (STAR - Counts) ...")
query <- GDCquery(
    project       = project,
    data.category = "Transcriptome Profiling",
    data.type     = "Gene Expression Quantification",
    workflow.type = "STAR - Counts",
    sample.type   = "Primary Tumor"
)
GDCdownload(query, directory = out_dir, method = "api")
data <- GDCprepare(query, directory = out_dir)

expr_tpm <- assay(data, "tpm_unstrand")
write.table(
    expr_tpm,
    file = file.path(out_dir, paste0(project, "_expression_matrix_full.tsv")),
    sep = "\t", quote = FALSE, col.names = NA
)

message("Querying matched clinical data ...")
clinical <- GDCquery_clinic(project = project, type = "clinical")
write.csv(
    clinical,
    file = file.path(meta_dir, paste0(project, "_clinical_full.csv")),
    row.names = FALSE
)

message("Done. Expression matrix and clinical table written to data/raw and data/metadata.")
