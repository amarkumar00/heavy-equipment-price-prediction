"""Clean the authorised heavy-equipment competition training data.

The module is deliberately small and explicit so a student can trace every data
quality decision. It never downloads data and never writes raw rows outside the
configured processed-data path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("data/raw/train.csv")
DEFAULT_OUTPUT = Path("data/processed/clean_train.csv")
DEFAULT_REPORT = Path("reports/data_quality_report.json")

REQUIRED_COLUMNS = {
    "TransactionID",
    "TransactionDate",
    "TargetValue",
    "ManufactureYear",
    "OperationalHoursMeter",
}


class DataContractError(ValueError):
    """Raised when an input file does not satisfy the documented data contract."""


def _json_value(value: Any) -> Any:
    """Convert NumPy/Pandas scalar values into JSON-safe Python values."""

    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def validate_columns(data: pd.DataFrame) -> None:
    """Fail early when the competition schema is not the expected one."""

    missing = sorted(REQUIRED_COLUMNS.difference(data.columns))
    if missing:
        raise DataContractError(
            "The training CSV is missing required column(s): "
            f"{', '.join(missing)}. Check data/README.md and confirm that the "
            "authorised competition train.csv was downloaded."
        )


def clean_training_data(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return cleaned rows and a machine-readable data-quality report."""

    validate_columns(data)
    clean = data.copy()
    input_rows = len(clean)
    input_columns = len(clean.columns)

    exact_duplicate_rows = int(clean.duplicated().sum())
    duplicate_transaction_ids = int(clean["TransactionID"].duplicated().sum())
    clean = clean.drop_duplicates().copy()

    raw_dates = clean["TransactionDate"].copy()
    clean["TransactionDate"] = pd.to_datetime(raw_dates, errors="coerce")
    invalid_dates = int(raw_dates.notna().sum() - clean["TransactionDate"].notna().sum())

    clean["TargetValue"] = pd.to_numeric(clean["TargetValue"], errors="coerce")
    invalid_target_mask = clean["TargetValue"].isna() | (clean["TargetValue"] <= 0)
    invalid_target_rows = int(invalid_target_mask.sum())
    clean = clean.loc[~invalid_target_mask].copy()

    manufacture_year = pd.to_numeric(clean["ManufactureYear"], errors="coerce")
    transaction_year = clean["TransactionDate"].dt.year
    invalid_year_mask = (
        manufacture_year.isin([1000, 1001])
        | manufacture_year.lt(1940)
        | manufacture_year.gt(transaction_year + 1)
    )
    invalid_manufacture_years = int(invalid_year_mask.fillna(False).sum())
    manufacture_year = manufacture_year.mask(invalid_year_mask)
    clean["ManufactureYearCorrected"] = manufacture_year
    clean["ManufactureYearMissing"] = manufacture_year.isna().astype("int8")

    age = transaction_year - manufacture_year
    clean["Age"] = age.where(age.between(0, 100))

    hours = pd.to_numeric(clean["OperationalHoursMeter"], errors="coerce")
    negative_hours = int(hours.lt(0).fillna(False).sum())
    hours = hours.mask(hours < 0)
    clean["OperationalHoursMeter"] = hours
    clean["OperationalHoursMissing"] = hours.isna().astype("int8")
    clean["OperationalHoursZero"] = hours.eq(0).astype("int8")
    positive_hours = hours.where(hours > 0)
    clean["LogOperationalHours"] = np.log1p(positive_hours)
    clean["HoursPerYear"] = (positive_hours / clean["Age"].clip(lower=1)).replace(
        [np.inf, -np.inf], np.nan
    )

    clean["SaleYear"] = clean["TransactionDate"].dt.year
    clean["SaleMonth"] = clean["TransactionDate"].dt.month
    clean["SaleQuarter"] = clean["TransactionDate"].dt.quarter
    clean["SaleDayOfWeek"] = clean["TransactionDate"].dt.dayofweek
    clean["LogTargetValue"] = np.log1p(clean["TargetValue"])
    clean["TotalMissingCount"] = clean.isna().sum(axis=1)

    clean = clean.sort_values(
        ["TransactionDate", "TransactionID"], na_position="last"
    ).reset_index(drop=True)

    missing_counts = {
        str(column): int(count)
        for column, count in clean.isna().sum().sort_values(ascending=False).items()
        if count > 0
    }
    target_summary = {
        key: _json_value(value)
        for key, value in clean["TargetValue"]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .to_dict()
        .items()
    }

    report: dict[str, Any] = {
        "input": {"rows": input_rows, "columns": input_columns},
        "output": {"rows": len(clean), "columns": len(clean.columns)},
        "removed": {
            "exact_duplicate_rows": exact_duplicate_rows,
            "invalid_or_non_positive_target_rows": invalid_target_rows,
        },
        "quality_observations": {
            "duplicate_transaction_ids_before_cleaning": duplicate_transaction_ids,
            "unparseable_transaction_dates": invalid_dates,
            "invalid_manufacture_years_set_to_missing": invalid_manufacture_years,
            "negative_operational_hours_set_to_missing": negative_hours,
            "missing_operational_hours_after_cleaning": int(hours.isna().sum()),
            "zero_operational_hours_retained_with_indicator": int(hours.eq(0).sum()),
        },
        "missing_value_counts_after_cleaning": missing_counts,
        "target_value_summary": target_summary,
        "decisions": [
            "Exact duplicate rows are removed; duplicate transaction IDs are reported.",
            "Non-positive or missing target values are excluded from supervised learning.",
            "Manufacture years outside 1940 to one year after sale are set to missing.",
            "Negative operating hours are set to missing; zero is retained and flagged.",
            "Associations discovered later must not be described as causal effects.",
        ],
    }
    return clean, report


def run(input_path: Path, output_path: Path, report_path: Path) -> None:
    """Read, clean, and save the dataset and quality report."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Training data not found at '{input_path}'. Join the competition, "
            "download the authorised files, place train.csv in data/raw/, and "
            "see data/README.md for the expected layout."
        )

    raw = pd.read_csv(input_path, low_memory=False)
    clean, report = clean_training_data(raw)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Cleaned {len(raw):,} rows into {len(clean):,} rows: {output_path}")
    print(f"Data-quality report: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.input, args.output, args.report)


if __name__ == "__main__":
    main()
