"""Scenario-based EOQ and reorder-point planning for equipment categories.

Auction transaction counts are used only as a market-activity proxy. Real
dealer demand, ordering cost, holding rate, lead time, and service target must
come from the decision maker. The output is a transparent planning scenario,
not an automatic procurement instruction.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATE_COLUMN = "TransactionDate"
TARGET_COLUMN = "TargetValue"
DEFAULT_CATEGORY_COLUMN = "InventoryGroupDescription"


class InventoryContractError(ValueError):
    """Raised when assumptions or source data are invalid."""


@dataclass(frozen=True)
class InventoryAssumptions:
    ordering_cost_per_cycle: float
    annual_holding_rate: float
    lead_time_days: float
    service_level_z: float
    review_years: int = 3

    def validate(self) -> None:
        if self.ordering_cost_per_cycle <= 0:
            raise InventoryContractError("ordering_cost_per_cycle must be positive.")
        if not 0 < self.annual_holding_rate < 1:
            raise InventoryContractError("annual_holding_rate must be between 0 and 1.")
        if self.lead_time_days <= 0:
            raise InventoryContractError("lead_time_days must be positive.")
        if self.service_level_z < 0:
            raise InventoryContractError("service_level_z cannot be negative.")
        if self.review_years < 1:
            raise InventoryContractError("review_years must be at least 1.")

    @classmethod
    def from_json(cls, path: Path) -> "InventoryAssumptions":
        values = json.loads(path.read_text(encoding="utf-8"))
        assumptions = cls(
            ordering_cost_per_cycle=float(values["ordering_cost_per_cycle"]),
            annual_holding_rate=float(values["annual_holding_rate"]),
            lead_time_days=float(values["lead_time_days"]),
            service_level_z=float(values["service_level_z"]),
            review_years=int(values.get("review_years", 3)),
        )
        assumptions.validate()
        return assumptions


def economic_order_quantity(
    annual_demand: float, ordering_cost: float, annual_holding_cost_per_unit: float
) -> float:
    """Return the classic EOQ: sqrt(2 * demand * order cost / holding cost)."""

    if annual_demand < 0 or ordering_cost <= 0 or annual_holding_cost_per_unit <= 0:
        raise InventoryContractError(
            "EOQ requires non-negative demand and positive ordering/holding costs."
        )
    return float(np.sqrt(2 * annual_demand * ordering_cost / annual_holding_cost_per_unit))


def build_inventory_plan(
    data: pd.DataFrame,
    assumptions: InventoryAssumptions,
    *,
    category_column: str = DEFAULT_CATEGORY_COLUMN,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build category-level EOQ and reorder-point scenarios."""

    assumptions.validate()
    required = {DATE_COLUMN, TARGET_COLUMN, category_column}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise InventoryContractError(
            "Inventory planning requires column(s): " + ", ".join(missing)
        )

    frame = data.loc[:, [DATE_COLUMN, TARGET_COLUMN, category_column]].copy()
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    frame[category_column] = frame[category_column].astype("string")
    frame = frame.loc[
        frame[DATE_COLUMN].notna()
        & frame[TARGET_COLUMN].gt(0)
        & frame[category_column].notna()
        & frame[category_column].ne("")
    ].copy()
    if frame.empty:
        raise InventoryContractError("No valid rows remain for inventory planning.")

    latest_date = pd.Timestamp(frame[DATE_COLUMN].max())
    latest_complete_year = latest_date.year if latest_date.month == 12 else latest_date.year - 1
    earliest_year = latest_complete_year - assumptions.review_years + 1
    frame = frame.loc[
        frame[DATE_COLUMN].dt.year.between(earliest_year, latest_complete_year)
    ].copy()
    if frame.empty:
        raise InventoryContractError(
            "No rows fall inside the requested complete-year review window."
        )
    frame["year"] = frame[DATE_COLUMN].dt.year
    frame["month"] = frame[DATE_COLUMN].dt.to_period("M").dt.to_timestamp()

    categories = sorted(frame[category_column].unique())
    years = list(range(earliest_year, latest_complete_year + 1))
    months = pd.date_range(
        f"{earliest_year}-01-01", f"{latest_complete_year}-12-01", freq="MS"
    )
    rows: list[dict[str, float | int | str]] = []
    for category in categories:
        category_data = frame.loc[frame[category_column].eq(category)]
        yearly_counts = (
            category_data.groupby("year").size().reindex(years, fill_value=0)
        )
        monthly_counts = (
            category_data.groupby("month").size().reindex(months, fill_value=0)
        )
        annual_demand = float(yearly_counts.mean())
        monthly_mean = float(monthly_counts.mean())
        monthly_standard_deviation = float(monthly_counts.std(ddof=1))
        median_price = float(category_data[TARGET_COLUMN].median())
        holding_cost = median_price * assumptions.annual_holding_rate
        eoq = economic_order_quantity(
            annual_demand,
            assumptions.ordering_cost_per_cycle,
            holding_cost,
        )
        lead_time_months = assumptions.lead_time_days / 30.4375
        safety_stock = (
            assumptions.service_level_z
            * monthly_standard_deviation
            * np.sqrt(lead_time_months)
        )
        reorder_point = monthly_mean * lead_time_months + safety_stock
        annual_orders = annual_demand / eoq if eoq > 0 else 0.0
        annual_ordering_cost = annual_orders * assumptions.ordering_cost_per_cycle
        annual_cycle_holding_cost = (eoq / 2) * holding_cost
        rows.append(
            {
                "category": str(category),
                "review_start_year": earliest_year,
                "review_end_year": latest_complete_year,
                "annual_transaction_demand_proxy": annual_demand,
                "median_historical_price": median_price,
                "annual_holding_cost_per_unit": holding_cost,
                "eoq_scenario_units": eoq,
                "reorder_point_scenario_units": float(reorder_point),
                "safety_stock_scenario_units": float(safety_stock),
                "estimated_orders_per_year": annual_orders,
                "annual_ordering_plus_cycle_holding_cost": float(
                    annual_ordering_cost + annual_cycle_holding_cost
                ),
            }
        )

    plan = pd.DataFrame(rows).sort_values(
        "annual_transaction_demand_proxy", ascending=False
    )
    metadata: dict[str, Any] = {
        "category_column": category_column,
        "review_window": [earliest_year, latest_complete_year],
        "assumptions": {
            "ordering_cost_per_cycle": assumptions.ordering_cost_per_cycle,
            "annual_holding_rate": assumptions.annual_holding_rate,
            "lead_time_days": assumptions.lead_time_days,
            "service_level_z": assumptions.service_level_z,
            "review_years": assumptions.review_years,
        },
        "interpretation": (
            "EOQ minimizes the classic ordering-plus-cycle-holding-cost formula "
            "under the supplied scenario assumptions."
        ),
        "limitations": [
            "Auction transaction volume is a market-activity proxy, not dealer demand.",
            "Ordering cost, holding rate, lead time, and service target are user assumptions.",
            "The output is a scenario for discussion, not an automatic purchase instruction.",
        ],
    }
    return plan.reset_index(drop=True), metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/clean_train.csv")
    )
    parser.add_argument(
        "--assumptions",
        type=Path,
        default=Path("config/inventory_assumptions.example.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/inventory_optimization.csv")
    )
    parser.add_argument(
        "--metadata", type=Path, default=Path("reports/inventory_optimization.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"Cleaned data not found at '{args.input}'. Run python -m src.clean_data first."
        )
    if not args.assumptions.exists():
        raise FileNotFoundError(
            f"Assumptions file not found at '{args.assumptions}'. Copy and edit the example config."
        )
    data = pd.read_csv(args.input, low_memory=False)
    assumptions = InventoryAssumptions.from_json(args.assumptions)
    plan, metadata = build_inventory_plan(data, assumptions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(args.output, index=False)
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Inventory scenario: {args.output}")
    print("Reminder:", metadata["limitations"][0])


if __name__ == "__main__":
    main()
