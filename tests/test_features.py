from __future__ import annotations

import pandas as pd
import pytest

from src.features import (
    FeatureContractError,
    FeatureEngineer,
    add_row_features,
    prepare_model_data,
)


def sample_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": ["T1", "T2", "T3"],
            "AssetID": ["A1", "A1", "A2"],
            "TransactionDate": ["2022-01-01", "2022-04-15", "2023-01-02"],
            "TargetValue": [10000, 12000, 15000],
            "ManufactureYear": [2020, 2020, 2018],
            "OperationalHoursMeter": [100, 300, None],
            "UtilizationTier": ["Low", "High", None],
            "AssetScaleFactor": ["Small", "Large", "Medium"],
            "Spec_FullDescriptor": ["Loader series", "Loader series", "Crane"],
            "ProductConfigID": [10, 10, 20],
            "Spec_BaseClass": ["Loader", "Loader", "Crane"],
            "Spec_SubClass": ["Wheel", "Wheel", "Mobile"],
            "FunctionalClassification": ["Earth", "Earth", "Lift"],
            "RegionCode": ["N", "N", "S"],
            "DataOriginCode": [1, 1, 2],
            "VendorPartnerID": [11, 11, 12],
            "InventoryGroupCategory": ["A", "A", "B"],
            "CabinType": ["Open", "Open", "Closed"],
            "Forks": ["No", "No", "Yes"],
            "DrivetrainType": ["4WD", "4WD", "2WD"],
        }
    )


def test_row_features_are_understandable_and_do_not_mutate_input() -> None:
    raw = sample_rows()
    original = raw.copy(deep=True)
    featured = add_row_features(raw)

    assert raw.equals(original)
    assert featured.loc[0, "SaleQuarter"] == 1
    assert featured.loc[0, "SaleIsWeekend"] == 1
    assert featured.loc[0, "Age"] == 2
    assert featured.loc[0, "UtilizationOrdinal"] == 0
    assert featured.loc[1, "EquipmentScaleOrdinal"] == 5
    assert featured.loc[0, "ReleaseSeriesPresent"] == 1


def test_frequency_features_are_fitted_only_on_training_rows() -> None:
    training = sample_rows().iloc[:2].copy()
    validation = sample_rows().iloc[2:].copy()
    engineer = FeatureEngineer().fit(training)

    training_features = engineer.transform(training)
    validation_features = engineer.transform(validation)

    assert training_features["Frequency_AssetID"].tolist() == [2.0, 2.0]
    assert validation_features["Frequency_AssetID"].item() == 0.0
    assert validation_features["Frequency_ProductConfigID"].item() == 0.0
    assert "TargetValue" not in training_features.columns


def test_prepare_model_data_retains_audit_columns() -> None:
    prepared = prepare_model_data(sample_rows())
    for column in ["TransactionID", "AssetID", "TransactionDate", "TargetValue"]:
        assert column in prepared.columns


def test_invalid_date_fails_before_modelling() -> None:
    bad = sample_rows()
    bad.loc[0, "TransactionDate"] = "not-a-date"
    with pytest.raises(FeatureContractError, match="invalid TransactionDate"):
        add_row_features(bad)
