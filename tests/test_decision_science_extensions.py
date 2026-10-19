from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.causal_sensitivity import run_causal_sensitivity
from src.inventory_optimization import (
    InventoryAssumptions,
    build_inventory_plan,
    economic_order_quantity,
)
from src.market_forecasting import (
    backtest_market_forecasts,
    future_market_scenario,
    monthly_market_series,
)


def monthly_transactions(months: int = 48) -> pd.DataFrame:
    rows = []
    for month_number, month in enumerate(
        pd.date_range("2010-01-01", periods=months, freq="MS")
    ):
        for transaction in range(40):
            rows.append(
                {
                    "TransactionDate": month
                    + pd.DateOffset(days=int(transaction % 20)),
                    "TargetValue": 20_000 + 175 * month_number + transaction,
                    "InventoryGroupDescription": "Excavator",
                }
            )
    return pd.DataFrame(rows)


def test_market_forecast_backtest_and_future_scenario() -> None:
    monthly = monthly_market_series(monthly_transactions(), min_transactions=30)
    backtest, report, trend_model = backtest_market_forecasts(
        monthly, holdout_months=12
    )
    future = future_market_scenario(
        monthly,
        selected_model=report["selected_by_lowest_holdout_rmse"],
        trend_model=trend_model,
        horizon_months=6,
    )

    assert len(backtest) == 12
    assert len(future) == 6
    assert set(report["models"]) == {
        "last_observed_price",
        "seasonal_naive_12_month",
        "linear_log_trend",
    }
    assert report["models"]["linear_log_trend"]["rmse"] >= 0
    assert future["forecast_median_price"].gt(0).all()


def test_eoq_formula_and_inventory_plan() -> None:
    assert economic_order_quantity(1_000, 100, 10) == pytest.approx(
        np.sqrt(20_000)
    )
    assumptions = InventoryAssumptions(
        ordering_cost_per_cycle=100,
        annual_holding_rate=0.2,
        lead_time_days=30,
        service_level_z=1.65,
        review_years=3,
    )
    data = monthly_transactions(months=48)
    plan, metadata = build_inventory_plan(data, assumptions)

    assert list(plan["category"]) == ["Excavator"]
    assert plan.loc[0, "eoq_scenario_units"] > 0
    assert plan.loc[0, "reorder_point_scenario_units"] > 0
    assert "market-activity proxy" in metadata["limitations"][0]


def causal_fixture(rows: int = 2_000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    age = rng.normal(12, 4, size=rows).clip(1, 30)
    sale_year = rng.integers(2005, 2013, size=rows)
    category = rng.choice(["TEX", "WL"], size=rows)
    region = rng.choice(["North", "South", "West"], size=rows)
    treatment_probability = 1 / (1 + np.exp(-(-1.2 + 0.09 * age)))
    treatment = rng.binomial(1, treatment_probability)
    log_price = (
        10.7
        - 0.035 * age
        + 0.15 * treatment
        + 0.10 * (category == "TEX")
        + rng.normal(0, 0.18, size=rows)
    )
    return pd.DataFrame(
        {
            "Age": age,
            "SaleYear": sale_year,
            "InventoryGroupCategory": category,
            "RegionCode": region,
            "UtilizationTier": np.where(treatment == 1, "High", "Low"),
            "TargetValue": np.expm1(log_price),
        }
    )


def test_causal_sensitivity_returns_balanced_observational_report() -> None:
    report = run_causal_sensitivity(causal_fixture(), max_rows=1_200)

    age_balance = report["numeric_balance_diagnostics"]["Age"]
    assert report["analysis_rows_after_sampling_and_overlap_trim"] > 700
    assert np.isfinite(report["weighted_approximate_percent_difference"])
    assert abs(age_balance["standardized_mean_difference_after"]) < abs(
        age_balance["standardized_mean_difference_before"]
    )
    assert "does not prove a causal effect" in report["limitations"][-1]
