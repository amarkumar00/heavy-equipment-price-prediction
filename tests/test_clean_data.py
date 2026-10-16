from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.clean_data import DataContractError, clean_training_data, run


def sample_raw_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": ["T1", "T2", "T3", "T4", "T4"],
            "TransactionDate": [
                "2024-01-01",
                "2024-02-01",
                "bad-date",
                "2024-04-01",
                "2024-04-01",
            ],
            "TargetValue": [10000, 20000, -5, 40000, 40000],
            "ManufactureYear": [2020, 1000, 2010, 2026, 2026],
            "OperationalHoursMeter": [100, 0, -10, None, None],
            "AssetID": ["A1", "A2", "A3", "A4", "A4"],
            "Spec_BaseClass": ["Loader", "Crane", "Loader", "Dozer", "Dozer"],
            "RegionCode": ["N", "S", "N", "W", "W"],
            "UtilizationTier": ["Low", "High", "Medium", "Low", "Low"],
        }
    )


def test_cleaning_records_quality_decisions() -> None:
    clean, report = clean_training_data(sample_raw_data())

    assert len(clean) == 3
    assert report["removed"]["exact_duplicate_rows"] == 1
    assert report["removed"]["invalid_or_non_positive_target_rows"] == 1
    assert report["quality_observations"]["invalid_manufacture_years_set_to_missing"] == 2
    assert clean["LogTargetValue"].notna().all()
    assert clean.loc[clean["TransactionID"] == "T2", "OperationalHoursZero"].item() == 1


def test_missing_required_columns_raise_clear_error() -> None:
    with pytest.raises(DataContractError, match="TargetValue"):
        clean_training_data(pd.DataFrame({"TransactionID": ["T1"]}))


def test_run_writes_csv_and_json(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.csv"
    clean_path = tmp_path / "clean.csv"
    report_path = tmp_path / "report.json"
    sample_raw_data().to_csv(raw_path, index=False)

    run(raw_path, clean_path, report_path)

    assert clean_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["output"]["rows"] == 3


def test_missing_file_error_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        run(tmp_path / "missing.csv", tmp_path / "out.csv", tmp_path / "report.json")
