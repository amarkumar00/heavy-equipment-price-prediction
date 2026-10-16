"""Compare classical price models using leakage-safe time validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

from src.features import FeatureEngineer, build_preprocessor
from src.metrics import from_log_predictions, price_metrics
from src.validation import asset_overlap_details, expanding_date_folds

DEFAULT_INPUT = Path("data/processed/model_data.csv")
DEFAULT_REPORTS = Path("reports")
DEFAULT_MODELS = Path("models")
RANDOM_SEED = 42


def _model_factories() -> dict[str, Any]:
    return {
        "ridge": lambda: Ridge(alpha=10.0, solver="lsqr"),
        "decision_tree": lambda: DecisionTreeRegressor(
            max_depth=18,
            min_samples_leaf=20,
            random_state=RANDOM_SEED,
        ),
        "lightgbm": lambda: lgb.LGBMRegressor(
            objective="regression",
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=30,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=0.2,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=-1,
        ),
    }


def _metric_row(
    *, model: str, fold: int, scope: str, actual: np.ndarray, predicted: np.ndarray
) -> dict[str, Any]:
    return {
        "model": model,
        "fold": fold,
        "scope": scope,
        "rows": len(actual),
        **price_metrics(actual, predicted),
    }


def _comparison_table(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    all_rows = fold_metrics.loc[fold_metrics["scope"] == "all_validation"]
    comparison = (
        all_rows.groupby("model", sort=False)
        .agg(
            folds=("fold", "nunique"),
            rmsle_mean=("rmsle", "mean"),
            rmsle_std=("rmsle", "std"),
            mae_mean=("mae", "mean"),
            rmse_mean=("rmse", "mean"),
            r2_mean=("r2", "mean"),
        )
        .reset_index()
        .sort_values("rmsle_mean")
    )
    return comparison


def train_and_compare(
    data: pd.DataFrame,
    reports_dir: Path,
    models_dir: Path,
    n_splits: int = 3,
) -> pd.DataFrame:
    """Run expanding-window CV, save reports, and fit the final LightGBM model."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    data = data.sort_values(["TransactionDate", "TransactionID"]).reset_index(drop=True)
    target = pd.to_numeric(data["TargetValue"], errors="raise").to_numpy(float)

    metric_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    oof_rows: list[pd.DataFrame] = []
    best_iterations: list[int] = []

    for fold in expanding_date_folds(data, n_splits=n_splits):
        train_data = data.iloc[fold.train_indices].copy()
        validation_data = data.iloc[fold.validation_indices].copy()
        y_train = target[fold.train_indices]
        y_validation = target[fold.validation_indices]
        y_train_log = np.log1p(y_train)
        y_validation_log = np.log1p(y_validation)

        feature_engineer = FeatureEngineer()
        train_features = feature_engineer.fit_transform(train_data)
        validation_features = feature_engineer.transform(validation_data)
        preprocessor = build_preprocessor()
        x_train = preprocessor.fit_transform(train_features)
        x_validation = preprocessor.transform(validation_features)

        seen_asset_mask, overlap = asset_overlap_details(train_data, validation_data)
        overlap_rows.append(
            {
                "fold": fold.number,
                "train_start": fold.train_start.date().isoformat(),
                "train_end": fold.train_end.date().isoformat(),
                "validation_start": fold.validation_start.date().isoformat(),
                "validation_end": fold.validation_end.date().isoformat(),
                "training_rows": len(train_data),
                **overlap,
            }
        )

        predictions: dict[str, np.ndarray] = {
            "median_baseline": np.full_like(
                y_validation, np.median(y_train), dtype=float
            )
        }
        for name, factory in _model_factories().items():
            model = factory()
            if name == "lightgbm":
                model.fit(
                    x_train,
                    y_train_log,
                    eval_set=[(x_validation, y_validation_log)],
                    eval_metric="l2",
                    callbacks=[lgb.early_stopping(75, verbose=False)],
                )
                best_iterations.append(int(model.best_iteration_ or model.n_estimators))
            else:
                model.fit(x_train, y_train_log)
            predictions[name] = from_log_predictions(model.predict(x_validation))

        unseen_mask = ~seen_asset_mask.to_numpy()
        for name, values in predictions.items():
            metric_rows.append(
                _metric_row(
                    model=name,
                    fold=fold.number,
                    scope="all_validation",
                    actual=y_validation,
                    predicted=values,
                )
            )
            if unseen_mask.sum() >= 2:
                metric_rows.append(
                    _metric_row(
                        model=name,
                        fold=fold.number,
                        scope="unseen_asset_only",
                        actual=y_validation[unseen_mask],
                        predicted=values[unseen_mask],
                    )
                )

        lightgbm_oof = validation_data[
            [
                "TransactionID",
                "AssetID",
                "TransactionDate",
                "TargetValue",
                "Age",
                "RegionCode",
                "InventoryGroupCategory",
                "OperationalHoursMissing",
            ]
        ].copy()
        lightgbm_oof["fold"] = fold.number
        lightgbm_oof["seen_asset_in_training"] = seen_asset_mask.to_numpy()
        lightgbm_oof["prediction"] = predictions["lightgbm"]
        oof_rows.append(lightgbm_oof)
        lightgbm_fold_rmsle = price_metrics(y_validation, predictions["lightgbm"])[
            "rmsle"
        ]
        print(
            f"Fold {fold.number}: {len(train_data):,} train / "
            f"{len(validation_data):,} validation rows; LightGBM RMSLE "
            f"{lightgbm_fold_rmsle:.4f}"
        )

    fold_metrics = pd.DataFrame(metric_rows)
    comparison = _comparison_table(fold_metrics)
    fold_metrics.to_csv(reports_dir / "classical_fold_metrics.csv", index=False)
    comparison.to_csv(reports_dir / "model_comparison.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(
        reports_dir / "asset_overlap_report.csv", index=False
    )
    pd.concat(oof_rows, ignore_index=True).to_csv(
        reports_dir / "lightgbm_oof_predictions.csv", index=False
    )

    # Fit the deployable classical artifact on every available historical row.
    final_feature_engineer = FeatureEngineer()
    final_features = final_feature_engineer.fit_transform(data)
    final_preprocessor = build_preprocessor()
    x_all = final_preprocessor.fit_transform(final_features)
    final_iterations = int(np.median(best_iterations)) if best_iterations else 400
    final_model = _model_factories()["lightgbm"]()
    final_model.set_params(n_estimators=final_iterations)
    final_model.fit(x_all, np.log1p(target))

    joblib.dump(
        final_feature_engineer, models_dir / "classical_feature_engineer.joblib"
    )
    joblib.dump(final_preprocessor, models_dir / "classical_preprocessor.joblib")
    final_model.booster_.save_model(str(models_dir / "lightgbm_model.txt"))

    feature_names = final_preprocessor.get_feature_names_out()
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "gain_importance": final_model.booster_.feature_importance(
                importance_type="gain"
            ),
            "split_importance": final_model.booster_.feature_importance(
                importance_type="split"
            ),
        }
    ).sort_values("gain_importance", ascending=False)
    importance.to_csv(reports_dir / "feature_importance.csv", index=False)

    metadata = {
        "training_rows": len(data),
        "training_date_min": str(pd.to_datetime(data["TransactionDate"]).min().date()),
        "training_date_max": str(pd.to_datetime(data["TransactionDate"]).max().date()),
        "target_transform": "log1p",
        "prediction_inverse": "clip(expm1(prediction), lower=0)",
        "validation": f"{n_splits} expanding unique-date folds",
        "final_lightgbm_iterations": final_iterations,
        "best_cv_model": str(comparison.iloc[0]["model"]),
        "note": (
            "Cross-validation evidence is separate from the original competition "
            "score. Final model is retrained on all authorised historical rows."
        ),
    }
    (models_dir / "classical_model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--splits", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"Model data not found at '{args.input}'. Run 'python -m "
            "src.prepare_data' first."
        )
    data = pd.read_csv(args.input, low_memory=False)
    comparison = train_and_compare(
        data, args.reports_dir, args.models_dir, n_splits=args.splits
    )
    print("\nTime-aware model comparison (lower RMSLE is better):")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
