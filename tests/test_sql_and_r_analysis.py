from __future__ import annotations

import shutil
import sqlite3
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from src.build_database import build_database
from src.clean_data import clean_training_data
from src.run_sql_analysis import collect_summary, create_analysis_views


REPO_ROOT = Path(__file__).resolve().parents[1]


def analysis_data() -> pd.DataFrame:
    rows = 16
    raw = pd.DataFrame(
        {
            "TransactionID": [f"T{index:02d}" for index in range(rows)],
            "AssetID": [f"A{index:02d}" for index in range(rows)],
            "TransactionDate": pd.date_range("2023-01-01", periods=rows, freq="MS"),
            "TargetValue": [12000 + 1750 * index for index in range(rows)],
            "ManufactureYear": [2010 + index % 10 for index in range(rows)],
            "OperationalHoursMeter": [500 + 650 * index for index in range(rows)],
            "Spec_BaseClass": ["Loader", "Crane", "Dozer", "Excavator"] * 4,
            "RegionCode": ["North", "South"] * 8,
            "UtilizationTier": ["Low", "Medium", "High", "Medium"] * 4,
            "DataOriginCode": ["Auction"] * rows,
            "InventoryGroupCategory": ["Earthmoving", "Lifting"] * 8,
        }
    )
    clean, _ = clean_training_data(raw)
    return clean


def test_sqlite_database_and_six_analyses(tmp_path: Path) -> None:
    database_path = tmp_path / "equipvalue.db"
    build_database(analysis_data(), database_path, REPO_ROOT / "sql/create_tables.sql")

    with sqlite3.connect(database_path) as connection:
        create_analysis_views(connection, REPO_ROOT / "sql/analysis_queries.sql")
        summary = collect_summary(connection)
        transaction_count = connection.execute(
            "SELECT COUNT(*) FROM equipment_transactions"
        ).fetchone()[0]

    assert transaction_count == 16
    assert summary["analysis"].nunique() == 6
    assert {"analysis", "dimension", "metric", "transaction_count"}.issubset(
        summary.columns
    )


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript is not installed")
def test_r_analysis_produces_expected_reports(tmp_path: Path) -> None:
    input_path = tmp_path / "clean.csv"
    output_dir = tmp_path / "reports"
    analysis_data().to_csv(input_path, index=False)

    result = subprocess.run(
        [
            "Rscript",
            str(REPO_ROOT / "r_analysis/equipment_price_analysis.R"),
            f"--input={input_path}",
            f"--output-dir={output_dir}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    expected = {
        "r_statistical_summary.csv",
        "r_group_summary.csv",
        "r_hypothesis_tests.csv",
        "price_by_category.png",
        "age_vs_price.png",
        "regression_diagnostics.png",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir()})

    hypothesis_tests = pd.read_csv(output_dir / "r_hypothesis_tests.csv")
    assert {
        "test",
        "null_hypothesis",
        "p_value",
        "p_value_display",
        "effect_size_name",
        "effect_size",
        "interpretation",
    }.issubset(hypothesis_tests.columns)
    assert {
        "one_way_anova_log_price_by_category",
        "welch_t_test_high_vs_low_utilization_log_price",
    }.issubset(set(hypothesis_tests["test"]))
