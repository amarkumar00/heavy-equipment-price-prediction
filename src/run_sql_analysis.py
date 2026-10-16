"""Create SQL analysis views and export one combined business summary."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd


DEFAULT_DATABASE = Path("data/equipvalue.db")
DEFAULT_QUERIES = Path("sql/analysis_queries.sql")
DEFAULT_OUTPUT = Path("reports/sql_business_summary.csv")

ANALYSES = {
    "average_price_by_age_band": "age_band_average_price",
    "median_price_by_category": "category_median_price",
    "average_price_by_region": "region_average_price",
    "average_price_by_utilization": "utilization_average_price",
    "average_price_by_operational_hours": "operational_hours_average_price",
    "transaction_volume_by_category": "category_transaction_volume",
}


def create_analysis_views(connection: sqlite3.Connection, sql_path: Path) -> None:
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL analysis file not found at '{sql_path}'.")
    sql = sql_path.read_text(encoding="utf-8")
    view_definitions = sql.split("-- Human-readable examples.", maxsplit=1)[0]
    connection.executescript(view_definitions)


def collect_summary(connection: sqlite3.Connection) -> pd.DataFrame:
    reports: list[pd.DataFrame] = []
    for analysis_name, view_name in ANALYSES.items():
        result = pd.read_sql_query(f"SELECT * FROM {view_name}", connection)
        result.insert(0, "analysis", analysis_name)
        reports.append(result)
    return pd.concat(reports, ignore_index=True)


def run(database_path: Path, sql_path: Path, output_path: Path) -> None:
    if not database_path.exists():
        raise FileNotFoundError(
            f"SQLite database not found at '{database_path}'. Run "
            "'python -m src.build_database' first."
        )
    with sqlite3.connect(database_path) as connection:
        create_analysis_views(connection, sql_path)
        summary = collect_summary(connection)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"Exported {len(summary):,} SQL result rows: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.database, args.queries, args.output)


if __name__ == "__main__":
    main()
