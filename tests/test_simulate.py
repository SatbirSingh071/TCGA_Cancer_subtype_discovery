import numpy as np

from src.simulate import SUBTYPE_MARKERS


def test_cohort_shape(small_cfg, small_cohort):
    expr, clinical = small_cohort
    assert expr.shape == (small_cfg["simulate"]["n_genes"], small_cfg["simulate"]["n_samples"])
    assert list(expr.columns) == list(clinical.index)


def test_clinical_has_required_columns(small_cohort):
    _, clinical = small_cohort
    for col in ["true_subtype", "age_at_index", "gender", "days_to_event", "event_observed"]:
        assert col in clinical.columns


def test_marker_genes_present_in_expression_matrix(small_cohort):
    expr, _ = small_cohort
    marker_genes = {g for markers in SUBTYPE_MARKERS.values() for g in markers}
    assert marker_genes.issubset(set(expr.index))


def test_marker_gene_direction_matches_subtype(small_cohort):
    """ESR1 should be higher, on average, in Luminal A samples than Basal-like."""
    expr, clinical = small_cohort
    log_expr = np.log2(expr.fillna(0) + 1)
    luminal_a_mean = log_expr.loc["ESR1", clinical["true_subtype"] == "Luminal A"].mean()
    basal_mean = log_expr.loc["ESR1", clinical["true_subtype"] == "Basal-like"].mean()
    assert luminal_a_mean > basal_mean


def test_survival_hazard_ordering(small_cohort):
    """Basal-like should show a higher observed event rate than Luminal A
    on average, reflecting the worse-prognosis hazard ratio used in the
    simulator (this is a statistical tendency, not a hard guarantee for a
    single small random draw, so we check on a fixed seed)."""
    _, clinical = small_cohort
    event_rate = clinical.groupby("true_subtype")["event_observed"].mean()
    assert event_rate["Basal-like"] >= event_rate["Luminal A"]
