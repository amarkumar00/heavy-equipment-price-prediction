from __future__ import annotations

import pandas as pd

from src.validation import asset_overlap_details, expanding_date_folds


def test_expanding_folds_keep_dates_strictly_in_the_future() -> None:
    data = pd.DataFrame(
        {
            "TransactionDate": pd.date_range("2024-01-01", periods=12, freq="D"),
            "AssetID": [f"A{index}" for index in range(12)],
        }
    )
    folds = list(expanding_date_folds(data, n_splits=3))

    assert len(folds) == 3
    previous_training_size = 0
    for fold in folds:
        assert fold.train_end < fold.validation_start
        assert len(fold.train_indices) > previous_training_size
        previous_training_size = len(fold.train_indices)


def test_same_date_never_appears_on_both_sides() -> None:
    data = pd.DataFrame(
        {
            "TransactionDate": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
                "2024-01-04",
                "2024-01-04",
            ],
            "AssetID": list("ABCDEFGH"),
        }
    )
    for fold in expanding_date_folds(data, n_splits=3):
        train_dates = set(data.iloc[fold.train_indices]["TransactionDate"])
        validation_dates = set(data.iloc[fold.validation_indices]["TransactionDate"])
        assert train_dates.isdisjoint(validation_dates)


def test_asset_overlap_is_reported_per_validation_row() -> None:
    training = pd.DataFrame({"AssetID": ["A1", "A2", "A2"]})
    validation = pd.DataFrame({"AssetID": ["A2", "A3", "A4"]})

    seen, report = asset_overlap_details(training, validation)

    assert seen.tolist() == [True, False, False]
    assert report["overlapping_unique_assets"] == 1
    assert report["validation_rows_with_seen_asset"] == 1
