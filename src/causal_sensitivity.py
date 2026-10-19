"""Observational sensitivity analysis for high equipment utilization.

The analysis uses propensity-score weighting to compare High- and Low-
utilization equipment after balancing a small set of observed confounders. It
is deliberately labelled a sensitivity analysis: unmeasured condition, damage,
seller behaviour, and local demand prevent a definitive causal claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_CONFOUNDERS = ("Age", "SaleYear")
CATEGORICAL_CONFOUNDERS = ("InventoryGroupCategory", "RegionCode")
TREATMENT_COLUMN = "UtilizationTier"
OUTCOME_COLUMN = "TargetValue"


class CausalContractError(ValueError):
    """Raised when the observational comparison cannot be performed."""


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights))


def _weighted_variance(values: np.ndarray, weights: np.ndarray) -> float:
    mean = _weighted_mean(values, weights)
    return float(np.average((values - mean) ** 2, weights=weights))


def _effective_sample_size(weights: np.ndarray) -> float:
    denominator = float(np.sum(weights**2))
    return float(np.sum(weights) ** 2 / denominator) if denominator else 0.0


def _standardized_mean_difference(
    values: np.ndarray, treatment: np.ndarray, weights: np.ndarray
) -> float:
    treated = treatment == 1
    control = treatment == 0
    treated_mean = _weighted_mean(values[treated], weights[treated])
    control_mean = _weighted_mean(values[control], weights[control])
    pooled = np.sqrt(
        (
            _weighted_variance(values[treated], weights[treated])
            + _weighted_variance(values[control], weights[control])
        )
        / 2
    )
    return float((treated_mean - control_mean) / pooled) if pooled > 0 else 0.0


def run_causal_sensitivity(
    data: pd.DataFrame,
    *,
    max_rows: int = 50_000,
    random_state: int = 42,
    propensity_trim: tuple[float, float] = (0.05, 0.95),
) -> dict[str, Any]:
    """Estimate an IPTW-adjusted High-vs-Low utilization price association."""

    required = {
        TREATMENT_COLUMN,
        OUTCOME_COLUMN,
        *NUMERIC_CONFOUNDERS,
        *CATEGORICAL_CONFOUNDERS,
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise CausalContractError(
            "Causal sensitivity analysis requires column(s): " + ", ".join(missing)
        )
    lower, upper = propensity_trim
    if not 0 < lower < upper < 1:
        raise CausalContractError("Propensity trim must satisfy 0 < lower < upper < 1.")

    frame = data.loc[
        data[TREATMENT_COLUMN].isin(["High", "Low"]),
        [
            TREATMENT_COLUMN,
            OUTCOME_COLUMN,
            *NUMERIC_CONFOUNDERS,
            *CATEGORICAL_CONFOUNDERS,
        ],
    ].copy()
    frame[OUTCOME_COLUMN] = pd.to_numeric(frame[OUTCOME_COLUMN], errors="coerce")
    frame = frame.loc[frame[OUTCOME_COLUMN].gt(0)].copy()
    if len(frame) < 200 or frame[TREATMENT_COLUMN].nunique() != 2:
        raise CausalContractError(
            "Need at least 200 valid High/Low rows with both groups represented."
        )
    if len(frame) > max_rows:
        sampled_groups = []
        total_rows = len(frame)
        for _, group in frame.groupby(TREATMENT_COLUMN):
            group_size = max(1, round(max_rows * len(group) / total_rows))
            sampled_groups.append(
                group.sample(n=min(group_size, len(group)), random_state=random_state)
            )
        frame = pd.concat(sampled_groups, ignore_index=True)

    treatment = frame[TREATMENT_COLUMN].eq("High").astype(int).to_numpy()
    features = frame.loc[:, [*NUMERIC_CONFOUNDERS, *CATEGORICAL_CONFOUNDERS]]
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, list(NUMERIC_CONFOUNDERS)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_CONFOUNDERS)),
        ]
    )
    propensity_model = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(max_iter=1_000, random_state=random_state),
            ),
        ]
    )
    propensity_model.fit(features, treatment)
    propensity = propensity_model.predict_proba(features)[:, 1]

    overlap = (propensity >= lower) & (propensity <= upper)
    frame = frame.loc[overlap].reset_index(drop=True)
    treatment = treatment[overlap]
    propensity = propensity[overlap]
    if len(frame) < 100 or len(np.unique(treatment)) != 2:
        raise CausalContractError(
            "Propensity trimming removed too many rows or one treatment group."
        )

    treated_rate = float(treatment.mean())
    weights = np.where(
        treatment == 1,
        treated_rate / propensity,
        (1 - treated_rate) / (1 - propensity),
    )
    lower_weight, upper_weight = np.quantile(weights, [0.01, 0.99])
    weights = np.clip(weights, lower_weight, upper_weight)

    log_price = np.log1p(frame[OUTCOME_COLUMN].to_numpy(dtype=float))
    treated = treatment == 1
    control = treatment == 0
    unweighted_difference = float(log_price[treated].mean() - log_price[control].mean())
    weighted_treated_mean = _weighted_mean(log_price[treated], weights[treated])
    weighted_control_mean = _weighted_mean(log_price[control], weights[control])
    weighted_difference = weighted_treated_mean - weighted_control_mean

    treated_variance = _weighted_variance(log_price[treated], weights[treated])
    control_variance = _weighted_variance(log_price[control], weights[control])
    treated_effective_n = _effective_sample_size(weights[treated])
    control_effective_n = _effective_sample_size(weights[control])
    standard_error = float(
        np.sqrt(
            treated_variance / treated_effective_n
            + control_variance / control_effective_n
        )
    )
    confidence_low = weighted_difference - 1.96 * standard_error
    confidence_high = weighted_difference + 1.96 * standard_error

    raw_weights = np.ones(len(frame), dtype=float)
    balance = {}
    for column in NUMERIC_CONFOUNDERS:
        values = pd.to_numeric(frame[column], errors="coerce")
        values = values.fillna(values.median()).to_numpy(dtype=float)
        balance[column] = {
            "standardized_mean_difference_before": _standardized_mean_difference(
                values, treatment, raw_weights
            ),
            "standardized_mean_difference_after": _standardized_mean_difference(
                values, treatment, weights
            ),
        }

    return {
        "question": (
            "After balancing observed age, sale year, category, and region, "
            "how is High utilization associated with selling price versus Low utilization?"
        ),
        "method": "Stabilized inverse-probability weighting with propensity trimming",
        "input_rows": int(len(data)),
        "eligible_high_low_rows_before_sampling": int(
            data[TREATMENT_COLUMN].isin(["High", "Low"]).sum()
        ),
        "analysis_rows_after_sampling_and_overlap_trim": int(len(frame)),
        "high_utilization_rows": int(treated.sum()),
        "low_utilization_rows": int(control.sum()),
        "propensity_trim": [lower, upper],
        "unadjusted_log_price_difference": unweighted_difference,
        "weighted_log_price_difference": weighted_difference,
        "weighted_approximate_percent_difference": float(
            (np.exp(weighted_difference) - 1) * 100
        ),
        "weighted_95_percent_interval_percent": [
            float((np.exp(confidence_low) - 1) * 100),
            float((np.exp(confidence_high) - 1) * 100),
        ],
        "effective_sample_size": {
            "high": treated_effective_n,
            "low": control_effective_n,
        },
        "numeric_balance_diagnostics": balance,
        "assumptions": [
            "All important confounders are measured and included.",
            "High and Low groups have enough propensity-score overlap.",
            "Each auction row is treated as independent for this learning analysis.",
        ],
        "limitations": [
            "Mechanical condition, damage, seller behaviour, and local demand are unmeasured.",
            "The utilisation label may encode information not captured by included confounders.",
            "This observational sensitivity analysis does not prove a causal effect.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/clean_train.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/causal_sensitivity.json")
    )
    parser.add_argument("--max-rows", type=int, default=50_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"Cleaned data not found at '{args.input}'. Run python -m src.clean_data first."
        )
    data = pd.read_csv(args.input, low_memory=False)
    report = run_causal_sensitivity(data, max_rows=args.max_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Causal sensitivity report: {args.output}")
    print("Reminder: observational adjustment does not prove causation.")


if __name__ == "__main__":
    main()
