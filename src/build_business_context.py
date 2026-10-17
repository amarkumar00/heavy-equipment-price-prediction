"""Build decision-support context from training and out-of-fold evidence.

The generated JSON contains only aggregate statistics. It is kept local because
the underlying competition dataset is access controlled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_MODEL_DATA = Path("data/processed/model_data.csv")
DEFAULT_OOF = Path("reports/lightgbm_oof_predictions.csv")
DEFAULT_OUTPUT = Path("models/business_context.json")
CATEGORY_COLUMN = "InventoryGroupCategory"

FORM_OPTION_COLUMNS = (
    "ProductConfigID",
    "Spec_FullDescriptor",
    "Spec_BaseClass",
    "UtilizationTier",
    "AssetScaleFactor",
    "DataOriginCode",
    "VendorPartnerID",
    "Spec_SubClass",
    "FunctionalClassification",
    "RegionCode",
    "InventoryGroupCategory",
    "CabinType",
    "Forks",
    "DrivetrainType",
)


def _python_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_context(model_data: pd.DataFrame, oof: pd.DataFrame) -> dict[str, Any]:
    required_model_columns = {
        "TransactionDate",
        "TargetValue",
        CATEGORY_COLUMN,
    }
    missing_model = sorted(required_model_columns.difference(model_data.columns))
    if missing_model:
        raise ValueError(
            "Model data is missing required column(s): " + ", ".join(missing_model)
        )
    required_oof_columns = {"TargetValue", "prediction"}
    missing_oof = sorted(required_oof_columns.difference(oof.columns))
    if missing_oof:
        raise ValueError(
            "OOF predictions are missing required column(s): " + ", ".join(missing_oof)
        )

    target = pd.to_numeric(model_data["TargetValue"], errors="coerce")
    categories = (
        model_data.assign(_target=target)
        .groupby(CATEGORY_COLUMN, dropna=False)["_target"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    category_statistics = {
        "__MISSING__" if pd.isna(row[CATEGORY_COLUMN]) else str(row[CATEGORY_COLUMN]): {
            "count": int(row["count"]),
            "average_selling_price": float(row["mean"]),
            "median_selling_price": float(row["median"]),
        }
        for _, row in categories.iterrows()
    }

    actual_log = np.log1p(pd.to_numeric(oof["TargetValue"], errors="raise"))
    predicted_log = np.log1p(
        np.clip(pd.to_numeric(oof["prediction"], errors="raise"), 0.0, None)
    )
    absolute_log_error = np.abs(actual_log - predicted_log)
    quantile = 0.90
    log_error_radius = float(np.quantile(absolute_log_error, quantile, method="higher"))
    empirical_coverage = float((absolute_log_error <= log_error_radius).mean())

    dates = pd.to_datetime(model_data["TransactionDate"], errors="coerce")
    form_options: dict[str, list[Any]] = {}
    defaults: dict[str, Any] = {}
    for column in FORM_OPTION_COLUMNS:
        if column not in model_data:
            continue
        counts = model_data[column].dropna().astype("string").value_counts()
        form_options[column] = [str(value) for value in counts.head(60).index]
        defaults[column] = str(counts.index[0]) if len(counts) else None

    manufacture_source = (
        model_data["ManufactureYearCorrected"]
        if "ManufactureYearCorrected" in model_data
        else pd.Series(np.nan, index=model_data.index)
    )
    hours_source = (
        model_data["OperationalHoursMeter"]
        if "OperationalHoursMeter" in model_data
        else pd.Series(np.nan, index=model_data.index)
    )
    manufacture_year = pd.to_numeric(manufacture_source, errors="coerce")
    operational_hours = pd.to_numeric(hours_source, errors="coerce")
    return {
        "context_version": 1,
        "training_rows": len(model_data),
        "training_date_min": dates.min().date().isoformat(),
        "training_date_max": dates.max().date().isoformat(),
        "global_price_statistics": {
            "average_selling_price": float(target.mean()),
            "median_selling_price": float(target.median()),
            "minimum_selling_price": float(target.min()),
            "maximum_selling_price": float(target.max()),
        },
        "category_column": CATEGORY_COLUMN,
        "category_statistics": category_statistics,
        "uncertainty": {
            "method": "90th percentile absolute log error on time-aware OOF predictions",
            "quantile": quantile,
            "log_error_radius": log_error_radius,
            "empirical_coverage": empirical_coverage,
            "oof_rows": len(oof),
            "note": (
                "This is an empirical model-error band, not a guaranteed "
                "confidence interval. Market shifts and unusual equipment can "
                "produce errors outside the range."
            ),
        },
        "form_options": form_options,
        "form_defaults": {
            **defaults,
            "ManufactureYear": _python_value(manufacture_year.median()),
            "OperationalHoursMeter": _python_value(operational_hours.median()),
        },
        "use_cases": [
            "Seller reserve-price planning",
            "Buyer budget estimation",
            "Auction inventory valuation",
            "Manual-review prioritization",
        ],
        "decision_boundary": (
            "Decision support only. This model does not make automatic purchase, "
            "sale, lending, or appraisal decisions. Human inspection is required."
        ),
    }


def run(model_data_path: Path, oof_path: Path, output_path: Path) -> None:
    for path, label in [
        (model_data_path, "model data"),
        (oof_path, "LightGBM OOF predictions"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required {label} not found at '{path}'. Run milestones 5-9 first."
            )
    model_data = pd.read_csv(model_data_path, low_memory=False)
    oof = pd.read_csv(oof_path, low_memory=False)
    context = build_context(model_data, oof)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(context, indent=2), encoding="utf-8")
    print(
        f"Business context built from {len(model_data):,} training rows and "
        f"{len(oof):,} OOF predictions: {output_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-data", type=Path, default=DEFAULT_MODEL_DATA)
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.model_data, args.oof, args.output)


if __name__ == "__main__":
    main()
