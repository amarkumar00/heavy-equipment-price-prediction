from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.batch_predict import (
    benchmark_batch_prediction,
    predict_csv,
    predict_dataframe_in_chunks,
)
from src.build_business_context import build_context
from src.features import FeatureEngineer, build_preprocessor
from src.predictor import PredictionInputError, PricePredictor


def synthetic_training_data(rows: int = 48) -> pd.DataFrame:
    index = np.arange(rows)
    return pd.DataFrame(
        {
            "TransactionID": [f"T{value}" for value in index],
            "AssetID": [f"A{value % 20}" for value in index],
            "TransactionDate": pd.date_range("2010-01-01", periods=rows, freq="7D"),
            "TargetValue": 12_000 + index * 550 + (index % 4) * 1_500,
            "ManufactureYear": 1995 + (index % 14),
            "OperationalHoursMeter": 500 + index * 90,
            "UtilizationTier": np.where(index % 2, "High", "Low"),
            "AssetScaleFactor": np.where(index % 3, "Medium", "Small"),
            "Spec_FullDescriptor": np.where(index % 2, "Loader series", "Dozer"),
            "ProductConfigID": 100 + (index % 5),
            "Spec_BaseClass": np.where(index % 2, "Loader", "Dozer"),
            "Spec_SubClass": np.where(index % 2, "G", "B"),
            "FunctionalClassification": np.where(index % 2, "Earth", "Road"),
            "RegionCode": np.where(index % 2, "Texas", "Florida"),
            "InventoryGroupCategory": np.where(index % 2, "TEX", "MG"),
            "CabinType": np.where(index % 2, "EROPS", "OROPS"),
            "Forks": np.where(index % 2, "Yes", "None or Unspecified"),
            "DrivetrainType": np.where(index % 2, "Standard", "Powershift"),
            "DataOriginCode": np.where(index % 2, "ADI149", "CEA153"),
            "VendorPartnerID": np.where(index % 2, "UJF", "GTX"),
        }
    )


@pytest.fixture()
def predictor(tmp_path: Path) -> PricePredictor:
    data = synthetic_training_data()
    engineer = FeatureEngineer()
    features = engineer.fit_transform(data)
    preprocessor = build_preprocessor(min_category_frequency=1)
    matrix = preprocessor.fit_transform(features)
    model = lgb.LGBMRegressor(
        n_estimators=30,
        learning_rate=0.1,
        num_leaves=15,
        min_child_samples=2,
        verbosity=-1,
        random_state=42,
    ).fit(matrix, np.log1p(data["TargetValue"]))
    predictions = np.expm1(model.predict(matrix))
    oof = pd.DataFrame({"TargetValue": data["TargetValue"], "prediction": predictions})
    context = build_context(data, oof)

    joblib.dump(engineer, tmp_path / "classical_feature_engineer.joblib")
    joblib.dump(preprocessor, tmp_path / "classical_preprocessor.joblib")
    model.booster_.save_model(str(tmp_path / "lightgbm_model.txt"))
    (tmp_path / "classical_model_metadata.json").write_text(
        json.dumps(
            {
                "training_date_min": "2010-01-01",
                "training_date_max": "2010-12-31",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "business_context.json").write_text(
        json.dumps(context), encoding="utf-8"
    )
    return PricePredictor.load(tmp_path)


def test_prediction_is_non_negative_explainable_and_target_safe(
    predictor: PricePredictor,
) -> None:
    rows = synthetic_training_data(3)
    first = predictor.predict(rows)
    changed_target = rows.copy()
    changed_target["TargetValue"] = [1, 999_999, 42]
    second = predictor.predict(changed_target)

    assert (first["estimated_selling_price"] >= 0).all()
    assert (first["range_lower"] <= first["estimated_selling_price"]).all()
    assert (first["estimated_selling_price"] <= first["range_upper"]).all()
    assert first["major_influencing_features"].str.len().gt(0).all()
    np.testing.assert_allclose(
        first["estimated_selling_price"], second["estimated_selling_price"]
    )


def test_unknown_categories_warn_but_can_be_scored(
    predictor: PricePredictor,
) -> None:
    row = synthetic_training_data(1)
    row["RegionCode"] = "Moon Base"
    row["ProductConfigID"] = "never-seen"
    row["InventoryGroupCategory"] = "ALIEN"
    result = predictor.predict(row).iloc[0]

    assert result.estimated_selling_price >= 0
    assert "unfamiliar RegionCode" in result.unfamiliar_input_warning
    assert "unfamiliar ProductConfigID" in result.unfamiliar_input_warning
    assert "unfamiliar InventoryGroupCategory" in result.unfamiliar_input_warning


def test_invalid_input_fails_with_understandable_error(
    predictor: PricePredictor,
) -> None:
    row = synthetic_training_data(1)
    row["TransactionDate"] = "bad-date"
    with pytest.raises(PredictionInputError, match="TransactionDate"):
        predictor.predict(row)


def test_chunked_batch_preserves_row_count_and_separates_invalid_rows(
    predictor: PricePredictor,
) -> None:
    data = synthetic_training_data(7)
    data["TransactionDate"] = data["TransactionDate"].astype("string")
    data.loc[1, "TransactionDate"] = "bad-date"
    data.loc[5, "OperationalHoursMeter"] = -1
    predictions, invalid = predict_dataframe_in_chunks(data, predictor, chunk_size=2)

    assert len(predictions) == 5
    assert len(invalid) == 2
    assert len(predictions) + len(invalid) == len(data)
    assert "TargetValue" not in predictions.columns
    assert invalid["validation_error"].str.len().gt(0).all()


def test_csv_chunk_row_numbers_continue_across_chunks(
    predictor: PricePredictor, tmp_path: Path
) -> None:
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    invalid_path = tmp_path / "invalid.csv"
    synthetic_training_data(5).to_csv(input_path, index=False)

    summary = predict_csv(
        input_path,
        output_path,
        invalid_path,
        predictor,
        chunk_size=2,
    )
    output = pd.read_csv(output_path)
    assert summary["predicted_rows"] == 5
    assert output["source_row_number"].tolist() == [2, 3, 4, 5, 6]


def test_all_invalid_csv_still_writes_useful_output_headers(
    predictor: PricePredictor, tmp_path: Path
) -> None:
    input_path = tmp_path / "invalid_input.csv"
    output_path = tmp_path / "empty_predictions.csv"
    invalid_path = tmp_path / "invalid_rows.csv"
    data = synthetic_training_data(2)
    data["TransactionDate"] = ["bad", "also-bad"]
    data.to_csv(input_path, index=False)

    summary = predict_csv(
        input_path, output_path, invalid_path, predictor, chunk_size=1
    )
    empty_predictions = pd.read_csv(output_path)
    invalid = pd.read_csv(invalid_path)
    assert summary["predicted_rows"] == 0
    assert "estimated_selling_price" in empty_predictions.columns
    assert len(invalid) == 2


def test_benchmark_records_runtime_memory_and_claim_boundary(
    predictor: PricePredictor,
) -> None:
    report = benchmark_batch_prediction(
        synthetic_training_data(8), predictor, sizes=[5, 10], chunk_size=4
    )

    assert [row["requested_rows"] for row in report["results"]] == [5, 10]
    assert all(row["runtime_seconds"] >= 0 for row in report["results"])
    assert all(row["python_peak_memory_mb"] >= 0 for row in report["results"])
    assert "not production-scale" in report["claim_boundary"]
