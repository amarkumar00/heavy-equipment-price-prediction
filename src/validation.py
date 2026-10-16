"""Time-aware expanding-window validation and repeated-asset diagnostics."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.features import DATE_COLUMN


@dataclass(frozen=True)
class DateFold:
    number: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def expanding_date_folds(data: pd.DataFrame, n_splits: int = 3) -> Iterator[DateFold]:
    """Split unique dates so the same date never appears on both sides."""

    if DATE_COLUMN not in data:
        raise ValueError(f"'{DATE_COLUMN}' is required for time-aware validation.")
    dates = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
    if dates.isna().any():
        raise ValueError("TransactionDate contains missing or invalid values.")

    unique_dates = np.sort(dates.unique())
    if len(unique_dates) < n_splits + 1:
        raise ValueError(
            f"Need at least {n_splits + 1} unique dates for {n_splits} folds; "
            f"found {len(unique_dates)}."
        )

    splitter = TimeSeriesSplit(n_splits=n_splits)
    date_values = dates.to_numpy(dtype="datetime64[ns]")
    for number, (train_date_positions, validation_date_positions) in enumerate(
        splitter.split(unique_dates), start=1
    ):
        train_dates = unique_dates[train_date_positions]
        validation_dates = unique_dates[validation_date_positions]
        train_indices = np.flatnonzero(np.isin(date_values, train_dates))
        validation_indices = np.flatnonzero(np.isin(date_values, validation_dates))
        yield DateFold(
            number=number,
            train_indices=train_indices,
            validation_indices=validation_indices,
            train_start=pd.Timestamp(train_dates[0]),
            train_end=pd.Timestamp(train_dates[-1]),
            validation_start=pd.Timestamp(validation_dates[0]),
            validation_end=pd.Timestamp(validation_dates[-1]),
        )


def asset_overlap_details(
    training_data: pd.DataFrame, validation_data: pd.DataFrame
) -> tuple[pd.Series, dict[str, float | int]]:
    """Flag validation rows whose AssetID already appeared in training."""

    if "AssetID" not in training_data or "AssetID" not in validation_data:
        raise ValueError(
            "AssetID is required for the repeated-equipment leakage check."
        )
    training_assets = set(training_data["AssetID"].dropna().astype("string"))
    validation_assets = validation_data["AssetID"].astype("string")
    seen_mask = validation_assets.isin(training_assets)
    overlapping_assets = set(validation_assets[seen_mask].dropna())
    report: dict[str, float | int] = {
        "training_unique_assets": int(training_data["AssetID"].nunique(dropna=True)),
        "validation_unique_assets": int(
            validation_data["AssetID"].nunique(dropna=True)
        ),
        "overlapping_unique_assets": len(overlapping_assets),
        "validation_rows_with_seen_asset": int(seen_mask.sum()),
        "validation_rows": len(validation_data),
        "seen_asset_row_rate": float(seen_mask.mean()) if len(seen_mask) else 0.0,
    }
    return seen_mask, report
