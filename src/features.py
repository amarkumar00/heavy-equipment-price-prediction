"""Leakage-safe feature engineering shared by every EquipValue model.

There are two feature stages:

1. ``add_row_features`` creates features using only values from that row.
2. ``FeatureEngineer`` learns category frequencies from the training fold and
   applies those frozen mappings to validation or future rows.

Keeping the stages separate makes the no-leakage rule visible and testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "TargetValue"
DATE_COLUMN = "TransactionDate"

FREQUENCY_COLUMNS = (
    "AssetID",
    "ProductConfigID",
    "Spec_FullDescriptor",
    "Spec_BaseClass",
    "FunctionalClassification",
    "RegionCode",
)

CATEGORICAL_FEATURES = (
    "DataOriginCode",
    "VendorPartnerID",
    "Spec_SubClass",
    "FunctionalClassification",
    "RegionCode",
    "InventoryGroupCategory",
    "CabinType",
    "Forks",
    "DrivetrainType",
)

ROW_NUMERIC_FEATURES = (
    "SaleYear",
    "SaleMonth",
    "SaleQuarter",
    "SaleDayOfWeek",
    "SaleIsWeekend",
    "SaleDaysSince1990",
    "ManufactureYearCorrected",
    "Age",
    "OperationalHoursMeter",
    "LogOperationalHours",
    "HoursPerYear",
    "ManufactureYearMissing",
    "OperationalHoursMissing",
    "OperationalHoursZero",
    "UtilizationOrdinal",
    "EquipmentScaleOrdinal",
    "DescriptionLength",
    "ReleaseSeriesPresent",
    "TotalMissingCount",
)

FREQUENCY_FEATURES = tuple(f"Frequency_{column}" for column in FREQUENCY_COLUMNS)
NUMERIC_FEATURES = ROW_NUMERIC_FEATURES + FREQUENCY_FEATURES

MODEL_INPUT_COLUMNS = tuple(dict.fromkeys(NUMERIC_FEATURES + CATEGORICAL_FEATURES))

SOURCE_COLUMNS = tuple(
    dict.fromkeys(
        (
            "TransactionID",
            "AssetID",
            DATE_COLUMN,
            TARGET_COLUMN,
            "ManufactureYear",
            "ManufactureYearCorrected",
            "OperationalHoursMeter",
            "UtilizationTier",
            "AssetScaleFactor",
            "Spec_FullDescriptor",
            "ProductConfigID",
            "Spec_BaseClass",
            "InventoryGroupDescription",
        )
        + CATEGORICAL_FEATURES
        + ROW_NUMERIC_FEATURES
        + FREQUENCY_COLUMNS
    )
)

MISSINGNESS_SOURCE_COLUMNS = tuple(
    dict.fromkeys(
        (
            "ManufactureYear",
            "OperationalHoursMeter",
            "UtilizationTier",
            "AssetScaleFactor",
            "Spec_FullDescriptor",
            "ProductConfigID",
            "Spec_BaseClass",
        )
        + CATEGORICAL_FEATURES
    )
)

UTILIZATION_ORDER = {"Low": 0.0, "Medium": 1.0, "High": 2.0}
SCALE_ORDER = {
    "Mini": 0.0,
    "Compact": 1.0,
    "Small": 2.0,
    "Medium": 3.0,
    "Large / Medium": 4.0,
    "Large": 5.0,
}


class FeatureContractError(ValueError):
    """Raised when required modelling inputs are absent or invalid."""


def _numeric(data: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric series, or an all-missing series when absent."""

    if column not in data:
        return pd.Series(np.nan, index=data.index, dtype="float64")
    return pd.to_numeric(data[column], errors="coerce")


def _text(data: pd.DataFrame, column: str) -> pd.Series:
    """Return a nullable string series, or an all-missing series when absent."""

    if column not in data:
        return pd.Series(pd.NA, index=data.index, dtype="string")
    return data[column].astype("string")


def _frequency_key(values: pd.Series) -> pd.Series:
    """Create stable string keys while keeping missing values explicit."""

    return values.astype("string").fillna("__MISSING__")


def add_row_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic, row-local features without learning from other rows."""

    if DATE_COLUMN not in data:
        raise FeatureContractError(
            f"Required date column '{DATE_COLUMN}' is missing from model data."
        )

    featured = data.copy()
    dates = pd.to_datetime(featured[DATE_COLUMN], errors="coerce", format="mixed")
    if dates.isna().any():
        invalid = int(dates.isna().sum())
        raise FeatureContractError(
            f"{invalid} row(s) have an invalid {DATE_COLUMN}; clean dates before "
            "feature engineering."
        )
    featured[DATE_COLUMN] = dates

    featured["SaleYear"] = dates.dt.year.astype("int16")
    featured["SaleMonth"] = dates.dt.month.astype("int8")
    featured["SaleQuarter"] = dates.dt.quarter.astype("int8")
    featured["SaleDayOfWeek"] = dates.dt.dayofweek.astype("int8")
    featured["SaleIsWeekend"] = dates.dt.dayofweek.ge(5).astype("int8")
    epoch = pd.Timestamp("1990-01-01")
    featured["SaleDaysSince1990"] = (dates - epoch).dt.days.astype("int32")

    if "ManufactureYearCorrected" in featured:
        manufacture_year = _numeric(featured, "ManufactureYearCorrected")
    else:
        manufacture_year = _numeric(featured, "ManufactureYear")
        invalid_year = (
            manufacture_year.isin([1000, 1001])
            | manufacture_year.lt(1940)
            | manufacture_year.gt(featured["SaleYear"] + 1)
        )
        manufacture_year = manufacture_year.mask(invalid_year)
    featured["ManufactureYearCorrected"] = manufacture_year
    featured["ManufactureYearMissing"] = manufacture_year.isna().astype("int8")

    age = featured["SaleYear"] - manufacture_year
    featured["Age"] = age.where(age.between(0, 100))

    hours = _numeric(featured, "OperationalHoursMeter").mask(
        _numeric(featured, "OperationalHoursMeter") < 0
    )
    featured["OperationalHoursMeter"] = hours
    featured["OperationalHoursMissing"] = hours.isna().astype("int8")
    featured["OperationalHoursZero"] = hours.eq(0).astype("int8")
    featured["LogOperationalHours"] = np.log1p(hours.where(hours > 0))
    featured["HoursPerYear"] = (
        hours.where(hours > 0) / featured["Age"].clip(lower=1)
    ).replace([np.inf, -np.inf], np.nan)

    featured["UtilizationOrdinal"] = (
        _text(featured, "UtilizationTier").map(UTILIZATION_ORDER).astype("float64")
    )
    featured["EquipmentScaleOrdinal"] = (
        _text(featured, "AssetScaleFactor").map(SCALE_ORDER).astype("float64")
    )
    description = _text(featured, "Spec_FullDescriptor")
    featured["DescriptionLength"] = description.str.len().astype("float64")
    featured["ReleaseSeriesPresent"] = (
        description.fillna("").str.contains("series", case=False).astype("int8")
    )

    source_for_missing = [
        column for column in MISSINGNESS_SOURCE_COLUMNS if column in data
    ]
    featured["TotalMissingCount"] = data[source_for_missing].isna().sum(axis=1)
    return featured


@dataclass
class FeatureEngineer:
    """Fit training-only frequency maps and build the shared model matrix."""

    frequency_maps: dict[str, dict[str, float]] = field(default_factory=dict)
    fitted: bool = False

    def fit(self, training_data: pd.DataFrame) -> FeatureEngineer:
        row_features = add_row_features(training_data)
        self.frequency_maps = {}
        for column in FREQUENCY_COLUMNS:
            if column not in row_features:
                self.frequency_maps[column] = {}
                continue
            keys = _frequency_key(row_features[column])
            counts = keys.value_counts(dropna=False)
            self.frequency_maps[column] = {
                str(key): float(value) for key, value in counts.items()
            }
        self.fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise FeatureContractError(
                "FeatureEngineer must be fitted on a training fold before transform()."
            )

        featured = add_row_features(data)
        for source, output in zip(FREQUENCY_COLUMNS, FREQUENCY_FEATURES):
            if source not in featured:
                featured[output] = 0.0
                continue
            keys = _frequency_key(featured[source])
            featured[output] = (
                keys.map(self.frequency_maps.get(source, {})).fillna(0.0).astype(float)
            )

        for column in CATEGORICAL_FEATURES:
            featured[column] = _text(featured, column).fillna("__MISSING__")
        for column in NUMERIC_FEATURES:
            featured[column] = _numeric(featured, column)

        return featured.loc[:, MODEL_INPUT_COLUMNS].copy()

    def fit_transform(self, training_data: pd.DataFrame) -> pd.DataFrame:
        return self.fit(training_data).transform(training_data)


def build_preprocessor(min_category_frequency: int = 20) -> ColumnTransformer:
    """Create identical numeric/categorical preprocessing for every model."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=min_category_frequency,
                    sparse_output=True,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ],
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )


def prepare_model_data(data: pd.DataFrame) -> pd.DataFrame:
    """Keep only modelling/audit columns and deterministic row features."""

    featured = add_row_features(data)
    required = {"TransactionID", "AssetID", DATE_COLUMN, TARGET_COLUMN}
    missing = sorted(required.difference(featured.columns))
    if missing:
        raise FeatureContractError(
            "Model data is missing required column(s): " + ", ".join(missing)
        )
    keep = [column for column in SOURCE_COLUMNS if column in featured]
    return featured.loc[:, list(dict.fromkeys(keep))].copy()
