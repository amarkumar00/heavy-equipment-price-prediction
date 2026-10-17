"""Reusable LightGBM prediction and business-recommendation service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse

from src.features import CATEGORICAL_FEATURES, FeatureEngineer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "models"
HUMAN_REVIEW_DISCLAIMER = (
    "Human inspection is still required before any purchase, sale, reserve-price, "
    "lending, or appraisal decision."
)


class PredictionInputError(ValueError):
    """Raised when one or more rows cannot be safely scored."""


@dataclass(frozen=True)
class ArtifactPaths:
    model: Path
    feature_engineer: Path
    preprocessor: Path
    metadata: Path
    business_context: Path

    @classmethod
    def from_directory(cls, directory: Path) -> ArtifactPaths:
        return cls(
            model=directory / "lightgbm_model.txt",
            feature_engineer=directory / "classical_feature_engineer.joblib",
            preprocessor=directory / "classical_preprocessor.joblib",
            metadata=directory / "classical_model_metadata.json",
            business_context=directory / "business_context.json",
        )

    def missing(self) -> list[Path]:
        return [path for path in self.__dict__.values() if not path.exists()]


def validate_prediction_inputs(data: pd.DataFrame) -> pd.DataFrame:
    """Return row-level validity and understandable error messages."""

    result = pd.DataFrame(index=data.index)
    errors: list[list[str]] = [[] for _ in range(len(data))]

    if "TransactionDate" not in data:
        for row_errors in errors:
            row_errors.append("TransactionDate is required")
        dates = pd.Series(pd.NaT, index=data.index)
    else:
        dates = pd.to_datetime(data["TransactionDate"], errors="coerce", format="mixed")
        for position in np.flatnonzero(dates.isna().to_numpy()):
            errors[position].append("TransactionDate is missing or invalid")

    if "ManufactureYear" in data:
        years = pd.to_numeric(data["ManufactureYear"], errors="coerce")
    elif "ManufactureYearCorrected" in data:
        years = pd.to_numeric(data["ManufactureYearCorrected"], errors="coerce")
    else:
        years = pd.Series(np.nan, index=data.index)

    non_missing_years = years.notna()
    invalid_years = non_missing_years & (
        years.lt(1940) | years.gt(dates.dt.year.add(1))
    )
    for position in np.flatnonzero(invalid_years.fillna(False).to_numpy()):
        errors[position].append(
            "ManufactureYear must be between 1940 and one year after sale"
        )

    if "OperationalHoursMeter" in data:
        raw_hours = data["OperationalHoursMeter"]
        hours = pd.to_numeric(raw_hours, errors="coerce")
        invalid_numeric_hours = raw_hours.notna() & hours.isna()
        negative_hours = hours.lt(0).fillna(False)
        for position in np.flatnonzero(invalid_numeric_hours.to_numpy()):
            errors[position].append("OperationalHoursMeter must be numeric or blank")
        for position in np.flatnonzero(negative_hours.to_numpy()):
            errors[position].append("OperationalHoursMeter cannot be negative")

    result["is_valid"] = [not row_errors for row_errors in errors]
    result["validation_error"] = ["; ".join(row_errors) for row_errors in errors]
    return result


def _normalise_key(value: Any) -> str:
    return "__MISSING__" if pd.isna(value) else str(value)


def _human_feature_name(feature_name: str) -> str:
    if feature_name.startswith("numeric__"):
        return feature_name.removeprefix("numeric__")
    name = feature_name.removeprefix("categorical__")
    for column in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
        prefix = f"{column}_"
        if name.startswith(prefix):
            return f"{column}={name.removeprefix(prefix)}"
    return name


class PricePredictor:
    """Load trained artifacts and produce decision-support predictions."""

    def __init__(
        self,
        *,
        model: lgb.Booster,
        feature_engineer: FeatureEngineer,
        preprocessor: Any,
        metadata: dict[str, Any],
        business_context: dict[str, Any],
    ) -> None:
        self.model = model
        self.feature_engineer = feature_engineer
        self.preprocessor = preprocessor
        self.metadata = metadata
        self.business_context = business_context
        self.feature_names = np.asarray(preprocessor.get_feature_names_out())
        categorical_pipeline = preprocessor.named_transformers_["categorical"]
        encoder = categorical_pipeline.named_steps["one_hot"]
        self.known_categorical_values = {
            column: {str(value) for value in values}
            for column, values in zip(
                CATEGORICAL_FEATURES, encoder.categories_, strict=True
            )
        }

    @classmethod
    def load(cls, artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> PricePredictor:
        paths = ArtifactPaths.from_directory(Path(artifact_dir))
        missing = paths.missing()
        if missing:
            missing_names = ", ".join(path.name for path in missing)
            raise FileNotFoundError(
                "Prediction artifacts are missing: "
                f"{missing_names}. Run classical training and "
                "'python -m src.build_business_context' first."
            )
        return cls(
            model=lgb.Booster(model_file=str(paths.model)),
            feature_engineer=joblib.load(paths.feature_engineer),
            preprocessor=joblib.load(paths.preprocessor),
            metadata=json.loads(paths.metadata.read_text(encoding="utf-8")),
            business_context=json.loads(
                paths.business_context.read_text(encoding="utf-8")
            ),
        )

    def _unfamiliar_warning(self, row: pd.Series) -> str:
        warnings: list[str] = []
        important_frequency_fields = (
            "ProductConfigID",
            "Spec_FullDescriptor",
            "Spec_BaseClass",
            "FunctionalClassification",
            "RegionCode",
        )
        for column in important_frequency_fields:
            key = _normalise_key(row.get(column, pd.NA))
            if key == "__MISSING__":
                warnings.append(f"{column} is missing")
            elif key not in self.feature_engineer.frequency_maps.get(column, {}):
                warnings.append(f"unfamiliar {column}: {key}")

        for column in (
            "InventoryGroupCategory",
            "CabinType",
            "DataOriginCode",
            "VendorPartnerID",
        ):
            key = _normalise_key(row.get(column, pd.NA))
            known = self.known_categorical_values.get(column, set())
            if key == "__MISSING__":
                warnings.append(f"{column} is missing")
            elif key not in known:
                warnings.append(f"unfamiliar {column}: {key}")

        training_min = pd.Timestamp(self.metadata["training_date_min"])
        training_max = pd.Timestamp(self.metadata["training_date_max"])
        date = pd.to_datetime(row.get("TransactionDate"), errors="coerce")
        if pd.notna(date) and not training_min <= date <= training_max:
            warnings.append(
                f"sale date is outside training period "
                f"{training_min.date()} to {training_max.date()}"
            )
        return (
            "; ".join(warnings) if warnings else "No major unfamiliar inputs detected"
        )

    def _category_average(self, value: Any) -> tuple[float, str]:
        category = _normalise_key(value)
        statistics = self.business_context["category_statistics"].get(category)
        if statistics:
            return float(statistics["average_selling_price"]), (
                f"Historical average for category {category}"
            )
        global_average = float(
            self.business_context["global_price_statistics"]["average_selling_price"]
        )
        return (
            global_average,
            "Overall historical average used; category was unfamiliar",
        )

    def predict(self, data: pd.DataFrame, *, validate: bool = True) -> pd.DataFrame:
        if data.empty:
            return pd.DataFrame(
                columns=[
                    "estimated_selling_price",
                    "range_lower",
                    "range_upper",
                    "major_influencing_features",
                    "similar_category_historical_average",
                    "category_average_note",
                    "unfamiliar_input_warning",
                    "uncertainty_note",
                    "human_inspection_required",
                ],
                index=data.index,
            )
        if validate:
            validation = validate_prediction_inputs(data)
            invalid = validation.loc[~validation["is_valid"]]
            if not invalid.empty:
                messages = "; ".join(
                    f"row {index}: {row.validation_error}"
                    for index, row in invalid.iterrows()
                )
                raise PredictionInputError(messages)

        features = self.feature_engineer.transform(data)
        matrix = self.preprocessor.transform(features)
        log_predictions = np.asarray(self.model.predict(matrix), dtype=float)
        estimates = np.clip(np.expm1(log_predictions), 0.0, None)
        if not np.isfinite(estimates).all():
            raise RuntimeError("Model produced a non-finite selling-price estimate.")

        raw_contributions = self.model.predict(matrix, pred_contrib=True)
        if sparse.issparse(raw_contributions):
            contributions = raw_contributions.toarray()
        else:
            contributions = np.asarray(raw_contributions)
        contribution_values = contributions[:, :-1]
        explanation_text: list[str] = []
        for row_values in contribution_values:
            top_indices = np.argsort(np.abs(row_values))[-5:][::-1]
            parts = []
            for index in top_indices:
                direction = "raised" if row_values[index] >= 0 else "lowered"
                parts.append(
                    f"{_human_feature_name(str(self.feature_names[index]))} ({direction})"
                )
            explanation_text.append("; ".join(parts))

        uncertainty = self.business_context["uncertainty"]
        radius = float(uncertainty["log_error_radius"])
        lower = np.clip(np.expm1(log_predictions - radius), 0.0, None)
        upper = np.clip(np.expm1(log_predictions + radius), 0.0, None)

        category_column = self.business_context["category_column"]
        category_values = (
            data[category_column]
            if category_column in data
            else pd.Series(pd.NA, index=data.index)
        )
        category_results = [self._category_average(value) for value in category_values]
        warnings = [self._unfamiliar_warning(row) for _, row in data.iterrows()]

        return pd.DataFrame(
            {
                "estimated_selling_price": estimates,
                "range_lower": lower,
                "range_upper": upper,
                "major_influencing_features": explanation_text,
                "similar_category_historical_average": [
                    value for value, _ in category_results
                ],
                "category_average_note": [note for _, note in category_results],
                "unfamiliar_input_warning": warnings,
                "uncertainty_note": uncertainty["note"],
                "human_inspection_required": HUMAN_REVIEW_DISCLAIMER,
            },
            index=data.index,
        )
