"""Build the normalized SQLite database from the cleaned competition data."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT = Path("data/processed/clean_train.csv")
DEFAULT_DATABASE = Path("data/equipvalue.db")
DEFAULT_SCHEMA = Path("sql/create_tables.sql")


ATTRIBUTE_COLUMNS = {
    "ManufactureYearCorrected": "manufacture_year",
    "Spec_FullDescriptor": "spec_full_descriptor",
    "Spec_BaseClass": "spec_base_class",
    "Spec_SubClass": "spec_sub_class",
    "Spec_ReleaseSeries": "spec_release_series",
    "Spec_VariantModifier": "spec_variant_modifier",
    "AssetScaleFactor": "asset_scale_factor",
    "FunctionalClassification": "functional_classification",
    "InventoryGroupCategory": "inventory_group_category",
    "InventoryGroupDescription": "inventory_group_description",
    "CabinType": "cabin_type",
    "Forks": "forks",
    "DrivetrainType": "drivetrain_type",
    "ProductConfigID": "product_config_id",
    "VendorPartnerID": "vendor_partner_id",
}

TRANSACTION_COLUMNS = {
    "TransactionID": "transaction_id",
    "TransactionDate": "transaction_date",
    "TargetValue": "target_value",
    "OperationalHoursMeter": "operational_hours_meter",
    "RegionCode": "region_code",
    "DataOriginCode": "data_origin_code",
    "UtilizationTier": "utilization_tier",
    "Age": "age",
    "LogOperationalHours": "log_operational_hours",
    "HoursPerYear": "hours_per_year",
    "OperationalHoursMissing": "operational_hours_missing",
    "OperationalHoursZero": "operational_hours_zero",
}


def _database_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "item"):
        return value.item()
    return value


def _column(data: pd.DataFrame, name: str) -> pd.Series:
    if name in data.columns:
        return data[name]
    return pd.Series([None] * len(data), index=data.index, dtype="object")


def prepare_tables(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create normalized attribute and transaction tables."""

    required = {"TransactionID", "TargetValue"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(
            "Cleaned data is missing required column(s): " + ", ".join(missing)
        )

    transaction_ids = data["TransactionID"].astype("string")
    if "AssetID" in data.columns:
        raw_asset_ids = data["AssetID"].astype("string")
        asset_ids = raw_asset_ids.where(
            raw_asset_ids.notna() & raw_asset_ids.ne(""),
            "transaction-" + transaction_ids,
        )
    else:
        asset_ids = "transaction-" + transaction_ids

    attributes = pd.DataFrame({"asset_id": asset_ids})
    for source, destination in ATTRIBUTE_COLUMNS.items():
        attributes[destination] = _column(data, source)
    attributes = attributes.drop_duplicates(subset=["asset_id"], keep="first")

    transactions = pd.DataFrame({"asset_id": asset_ids})
    for source, destination in TRANSACTION_COLUMNS.items():
        transactions[destination] = _column(data, source)

    if "transaction_date" in transactions:
        parsed_dates = pd.to_datetime(transactions["transaction_date"], errors="coerce")
        transactions["transaction_date"] = parsed_dates.dt.strftime("%Y-%m-%d")

    return attributes, transactions


def build_database(data: pd.DataFrame, database_path: Path, schema_path: Path) -> None:
    """Create tables, insert rows, and validate the resulting database."""

    if not schema_path.exists():
        raise FileNotFoundError(f"SQL schema not found at '{schema_path}'.")

    attributes, transactions = prepare_tables(data)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(database_path) as connection:
        connection.executescript(schema_path.read_text(encoding="utf-8"))

        attribute_columns = list(attributes.columns)
        attribute_sql = (
            f"INSERT INTO equipment_attributes ({', '.join(attribute_columns)}) "
            f"VALUES ({', '.join('?' for _ in attribute_columns)})"
        )
        connection.executemany(
            attribute_sql,
            (
                tuple(_database_safe(value) for value in row)
                for row in attributes.itertuples(index=False, name=None)
            ),
        )

        transaction_columns = list(transactions.columns)
        transaction_sql = (
            f"INSERT INTO equipment_transactions ({', '.join(transaction_columns)}) "
            f"VALUES ({', '.join('?' for _ in transaction_columns)})"
        )
        connection.executemany(
            transaction_sql,
            (
                tuple(_database_safe(value) for value in row)
                for row in transactions.itertuples(index=False, name=None)
            ),
        )
        connection.commit()

        saved_transactions = connection.execute(
            "SELECT COUNT(*) FROM equipment_transactions"
        ).fetchone()[0]
        if saved_transactions != len(transactions):
            raise RuntimeError(
                f"Expected {len(transactions)} transactions, saved {saved_transactions}."
            )


def run(input_path: Path, database_path: Path, schema_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Cleaned data not found at '{input_path}'. Run "
            "'python -m src.clean_data' first."
        )
    data = pd.read_csv(input_path, low_memory=False)
    build_database(data, database_path, schema_path)
    print(f"SQLite database created from {len(data):,} rows: {database_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.input, args.database, args.schema)


if __name__ == "__main__":
    main()
