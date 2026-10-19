"""Clinical comparison and survival analysis across discovered clusters.

Implements:
* Cross-tabulation / statistical comparison of clusters vs. clinical
  variables (age via ANOVA, gender/stage via chi-square).
* Kaplan-Meier estimation per cluster + pairwise and global log-rank tests.
* A multivariate Cox proportional-hazards model adjusting for age and stage,
  to check whether the cluster effect on survival persists after adjusting
  for known confounders.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from scipy import stats

from .logging_config import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Clinical comparison
# --------------------------------------------------------------------------- #
def summarize_clinical_by_cluster(clinical: pd.DataFrame, cluster_labels: pd.Series) -> pd.DataFrame:
    df = clinical.copy()
    df["cluster"] = cluster_labels.loc[df.index]
    numeric_cols = [c for c in ["age_at_index"] if c in df.columns]
    categorical_cols = [c for c in ["gender", "ajcc_pathologic_stage", "tumor_grade", "vital_status"] if c in df.columns]

    summary_rows = []
    for cluster_id, group in df.groupby("cluster"):
        row: dict[str, Any] = {"cluster": cluster_id, "n": len(group)}
        for col in numeric_cols:
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_sd"] = group[col].std()
        for col in categorical_cols:
            top = group[col].value_counts(normalize=True)
            if len(top):
                row[f"{col}_mode"] = top.index[0]
                row[f"{col}_mode_pct"] = round(top.iloc[0] * 100, 1)
        summary_rows.append(row)
    return pd.DataFrame(summary_rows).set_index("cluster")


def clinical_association_tests(clinical: pd.DataFrame, cluster_labels: pd.Series) -> pd.DataFrame:
    """ANOVA for continuous variables, chi-square for categorical variables."""
    df = clinical.copy()
    df["cluster"] = cluster_labels.loc[df.index]
    rows = []

    if "age_at_index" in df.columns:
        groups = [g["age_at_index"].dropna().values for _, g in df.groupby("cluster")]
        f_stat, p_val = stats.f_oneway(*groups)
        rows.append({"variable": "age_at_index", "test": "one-way ANOVA", "statistic": f_stat, "p_value": p_val})

    for col in ["gender", "ajcc_pathologic_stage", "tumor_grade", "vital_status"]:
        if col in df.columns:
            contingency = pd.crosstab(df["cluster"], df[col])
            if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
                rows.append({"variable": col, "test": "chi-square", "statistic": chi2, "p_value": p_val})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Kaplan-Meier + log-rank
# --------------------------------------------------------------------------- #
def fit_km_by_cluster(clinical: pd.DataFrame, cluster_labels: pd.Series, time_col: str, event_col: str) -> dict[Any, KaplanMeierFitter]:
    fitters = {}
    df = clinical.copy()
    df["cluster"] = cluster_labels.loc[df.index]
    for cluster_id, group in df.groupby("cluster"):
        kmf = KaplanMeierFitter(label=str(cluster_id))
        kmf.fit(group[time_col], event_observed=group[event_col])
        fitters[cluster_id] = kmf
    return fitters


def global_logrank_test(clinical: pd.DataFrame, cluster_labels: pd.Series, time_col: str, event_col: str):
    df = clinical.copy()
    df["cluster"] = cluster_labels.loc[df.index]
    result = multivariate_logrank_test(df[time_col], df["cluster"], df[event_col])
    return result


def pairwise_logrank_tests(clinical: pd.DataFrame, cluster_labels: pd.Series, time_col: str, event_col: str) -> pd.DataFrame:
    df = clinical.copy()
    df["cluster"] = cluster_labels.loc[df.index]
    rows = []
    for c1, c2 in combinations(sorted(df["cluster"].unique()), 2):
        g1 = df[df["cluster"] == c1]
        g2 = df[df["cluster"] == c2]
        result = logrank_test(g1[time_col], g2[time_col], g1[event_col], g2[event_col])
        rows.append({"cluster_a": c1, "cluster_b": c2, "test_statistic": result.test_statistic, "p_value": result.p_value})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Cox proportional hazards (multivariate adjustment)
# --------------------------------------------------------------------------- #
def fit_cox_model(clinical: pd.DataFrame, cluster_labels: pd.Series, time_col: str, event_col: str, adjust_for: list[str] | None = None) -> CoxPHFitter:
    df = clinical.copy()
    df["cluster"] = cluster_labels.loc[df.index].astype(str)
    cols = [time_col, event_col, "cluster"] + (adjust_for or [])
    model_df = df[cols].dropna().copy()
    model_df = pd.get_dummies(model_df, columns=["cluster"] + [c for c in (adjust_for or []) if model_df[c].dtype == object], drop_first=True)
    cph = CoxPHFitter()
    cph.fit(model_df, duration_col=time_col, event_col=event_col)
    return cph
