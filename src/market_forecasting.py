"""Forecast monthly heavy-equipment market prices with simple baselines.

This module answers a different question from the row-level price model:

* row-level model: "What may this specific machine sell for?"
* market forecast: "How may the typical market price move over coming months?"

The models are intentionally simple and auditable. They backtest on the latest
historical months before producing a future scenario. Because the authorised
dataset ends in 2013, the future rows are historical-study scenarios, not
current-market forecasts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATE_COLUMN = "TransactionDate"
TARGET_COLUMN = "TargetValue"
CATEGORY_COLUMN = "InventoryGroupDescription"


class ForecastContractError(ValueError):
    """Raised when data cannot support the documented forecast."""


def monthly_market_series(
    data: pd.DataFrame,
    *,
    category: str | None = None,
    category_column: str = CATEGORY_COLUMN,
    min_transactions: int = 30,
) -> pd.DataFrame:
    """Aggregate transactions into a monthly median-price time series."""

    required = {DATE_COLUMN, TARGET_COLUMN}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ForecastContractError(
            "Market forecasting requires column(s): " + ", ".join(missing)
        )
    if category is not None and category_column not in data:
        raise ForecastContractError(
            f"Category filter requested, but '{category_column}' is missing."
        )
    if min_transactions < 1:
        raise ForecastContractError("min_transactions must be at least 1.")

    frame = data.copy()
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    frame = frame.loc[
        frame[DATE_COLUMN].notna()
        & frame[TARGET_COLUMN].notna()
        & frame[TARGET_COLUMN].gt(0)
    ].copy()
    if category is not None:
        frame = frame.loc[frame[category_column].astype("string").eq(category)].copy()
    if frame.empty:
        raise ForecastContractError("No valid rows remain for the requested market series.")

    frame["month"] = frame[DATE_COLUMN].dt.to_period("M").dt.to_timestamp()
    monthly = (
        frame.groupby("month", as_index=False)
        .agg(
            transaction_count=(TARGET_COLUMN, "size"),
            median_price=(TARGET_COLUMN, "median"),
        )
        .sort_values("month")
    )
    monthly = monthly.loc[monthly["transaction_count"].ge(min_transactions)].copy()
    if len(monthly) < 24:
        raise ForecastContractError(
            "Need at least 24 sufficiently active months for a useful backtest; "
            f"found {len(monthly)}."
        )
    return monthly.reset_index(drop=True)


def _month_number(months: pd.Series, origin: pd.Timestamp) -> np.ndarray:
    return (
        (months.dt.year - origin.year) * 12 + (months.dt.month - origin.month)
    ).to_numpy(dtype=float)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    percentage_errors = np.abs((actual_values - predicted_values) / actual_values)
    return {
        "mae": float(mean_absolute_error(actual_values, predicted_values)),
        "rmse": float(np.sqrt(mean_squared_error(actual_values, predicted_values))),
        "mape_percent": float(np.mean(percentage_errors) * 100),
    }


def backtest_market_forecasts(
    monthly: pd.DataFrame, *, holdout_months: int = 12
) -> tuple[pd.DataFrame, dict[str, Any], LinearRegression]:
    """Compare last-price, seasonal-naive, and linear-trend forecasts."""

    if holdout_months < 3:
        raise ForecastContractError("holdout_months must be at least 3.")
    if len(monthly) < holdout_months + 12:
        raise ForecastContractError(
            "Need at least 12 training months in addition to the holdout period."
        )

    ordered = monthly.sort_values("month").reset_index(drop=True).copy()
    training = ordered.iloc[:-holdout_months].copy()
    validation = ordered.iloc[-holdout_months:].copy()
    origin = pd.Timestamp(ordered["month"].min())

    last_price_prediction = np.repeat(
        float(training["median_price"].iloc[-1]), len(validation)
    )

    historical_prices = ordered.set_index("month")["median_price"]
    seasonal_prediction = []
    for month in validation["month"]:
        previous_year = pd.Timestamp(month) - pd.DateOffset(years=1)
        seasonal_prediction.append(
            float(
                historical_prices.get(previous_year, training["median_price"].iloc[-1])
            )
        )
    seasonal_prediction_array = np.asarray(seasonal_prediction, dtype=float)

    trend_model = LinearRegression()
    training_x = _month_number(training["month"], origin).reshape(-1, 1)
    validation_x = _month_number(validation["month"], origin).reshape(-1, 1)
    trend_model.fit(training_x, np.log1p(training["median_price"].to_numpy()))
    trend_prediction = np.maximum(
        0.0, np.expm1(trend_model.predict(validation_x))
    )

    predictions = {
        "last_observed_price": last_price_prediction,
        "seasonal_naive_12_month": seasonal_prediction_array,
        "linear_log_trend": trend_prediction,
    }
    actual = validation["median_price"].to_numpy(dtype=float)
    model_metrics = {
        name: _metrics(actual, predicted) for name, predicted in predictions.items()
    }
    selected_model = min(
        model_metrics, key=lambda name: model_metrics[name]["rmse"]
    )

    backtest = validation.loc[:, ["month", "transaction_count", "median_price"]].copy()
    backtest = backtest.rename(columns={"median_price": "actual_median_price"})
    for name, values in predictions.items():
        backtest[name] = values

    report: dict[str, Any] = {
        "series_start": str(ordered["month"].min().date()),
        "series_end": str(ordered["month"].max().date()),
        "months": int(len(ordered)),
        "training_months": int(len(training)),
        "holdout_months": int(len(validation)),
        "validation_start": str(validation["month"].min().date()),
        "validation_end": str(validation["month"].max().date()),
        "models": model_metrics,
        "selected_by_lowest_holdout_rmse": selected_model,
        "warning": (
            "Historical market-trend study only. The source data ends in 2013, "
            "so these forecasts are not current-price guidance."
        ),
    }
    return backtest, report, trend_model


def future_market_scenario(
    monthly: pd.DataFrame,
    *,
    selected_model: str,
    trend_model: LinearRegression,
    horizon_months: int = 6,
) -> pd.DataFrame:
    """Create a short post-dataset scenario using the selected simple model."""

    if horizon_months < 1:
        raise ForecastContractError("horizon_months must be at least 1.")
    ordered = monthly.sort_values("month").reset_index(drop=True).copy()
    last_month = pd.Timestamp(ordered["month"].iloc[-1])
    months = pd.date_range(
        last_month + pd.offsets.MonthBegin(1), periods=horizon_months, freq="MS"
    )
    origin = pd.Timestamp(ordered["month"].min())
    historical = ordered.set_index("month")["median_price"]

    if selected_model == "linear_log_trend":
        x = _month_number(pd.Series(months), origin).reshape(-1, 1)
        values = np.maximum(0.0, np.expm1(trend_model.predict(x)))
    elif selected_model == "seasonal_naive_12_month":
        values = np.asarray(
            [
                historical.get(
                    pd.Timestamp(month) - pd.DateOffset(years=1),
                    ordered["median_price"].iloc[-1],
                )
                for month in months
            ],
            dtype=float,
        )
    elif selected_model == "last_observed_price":
        values = np.repeat(float(ordered["median_price"].iloc[-1]), horizon_months)
    else:
        raise ForecastContractError(f"Unknown selected model: {selected_model}")

    return pd.DataFrame(
        {
            "month": months,
            "forecast_median_price": values,
            "model": selected_model,
            "scenario_only": True,
        }
    )


def save_forecast_plot(backtest: pd.DataFrame, output_path: Path) -> None:
    """Save a readable backtest chart."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.plot(
        backtest["month"],
        backtest["actual_median_price"],
        marker="o",
        linewidth=2,
        label="Actual monthly median",
    )
    for column, label in (
        ("last_observed_price", "Last observed"),
        ("seasonal_naive_12_month", "Seasonal naive"),
        ("linear_log_trend", "Linear log trend"),
    ):
        axis.plot(backtest["month"], backtest[column], linewidth=1.6, label=label)
    axis.set_title("Monthly median-price forecast backtest")
    axis.set_ylabel("Median selling price")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/clean_train.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--category")
    parser.add_argument("--min-transactions", type=int, default=30)
    parser.add_argument("--holdout-months", type=int, default=12)
    parser.add_argument("--horizon-months", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"Cleaned data not found at '{args.input}'. Run python -m src.clean_data first."
        )
    data = pd.read_csv(args.input, low_memory=False)
    monthly = monthly_market_series(
        data,
        category=args.category,
        min_transactions=args.min_transactions,
    )
    backtest, report, trend_model = backtest_market_forecasts(
        monthly, holdout_months=args.holdout_months
    )
    future = future_market_scenario(
        monthly,
        selected_model=report["selected_by_lowest_holdout_rmse"],
        trend_model=trend_model,
        horizon_months=args.horizon_months,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    backtest.to_csv(args.output_dir / "market_forecast_backtest.csv", index=False)
    future.to_csv(args.output_dir / "market_forecast_future.csv", index=False)
    (args.output_dir / "market_forecast_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    save_forecast_plot(backtest, args.output_dir / "market_forecast_backtest.png")
    print(
        "Market forecast complete. Selected model:",
        report["selected_by_lowest_holdout_rmse"],
    )
    print("Reminder:", report["warning"])


if __name__ == "__main__":
    main()
