"""Compare LightGBM and PyTorch failure patterns without claiming causation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics import price_metrics

DEFAULT_LIGHTGBM = Path("reports/lightgbm_oof_predictions.csv")
DEFAULT_PYTORCH = Path("reports/pytorch_oof_predictions.csv")
DEFAULT_REPORTS = Path("reports")


def _age_band(age: pd.Series) -> pd.Series:
    return (
        pd.cut(
            pd.to_numeric(age, errors="coerce"),
            bins=[-np.inf, 5, 15, np.inf],
            labels=["new_0_to_5", "mid_6_to_15", "old_over_15"],
        )
        .astype("string")
        .fillna("unknown")
    )


def _price_band(target: pd.Series) -> pd.Series:
    """Descriptive OOF terciles; never used as a training feature."""

    ranked = target.rank(method="first")
    return pd.qcut(ranked, q=3, labels=["low", "middle", "high"])


def _segment_metrics(
    data: pd.DataFrame, model: str, segment_type: str, segment_column: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for segment, group in data.groupby(segment_column, dropna=False):
        if len(group) < 2:
            continue
        rows.append(
            {
                "model": model,
                "segment_type": segment_type,
                "segment": str(segment),
                "rows": len(group),
                **price_metrics(
                    group["TargetValue"].to_numpy(),
                    group[f"{model}_prediction"].to_numpy(),
                ),
            }
        )
    return rows


def analyze(
    lightgbm_path: Path, pytorch_path: Path, reports_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in [lightgbm_path, pytorch_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Prediction file not found at '{path}'. Run both training scripts first."
            )
    lightgbm = pd.read_csv(lightgbm_path, low_memory=False)
    pytorch = pd.read_csv(pytorch_path, low_memory=False)
    key = ["TransactionID", "fold"]
    keep = key + ["prediction"]
    merged = lightgbm.merge(
        pytorch[keep], on=key, how="inner", suffixes=("_lightgbm", "_pytorch")
    ).rename(
        columns={
            "prediction_lightgbm": "lightgbm_prediction",
            "prediction_pytorch": "pytorch_prediction",
        }
    )
    if merged.empty:
        raise ValueError("The two model files contain no matching OOF transactions.")

    merged["age_band"] = _age_band(merged["Age"])
    merged["price_band"] = _price_band(merged["TargetValue"])
    merged["hours_availability"] = np.where(
        merged["OperationalHoursMissing"].astype(bool), "hours_missing", "hours_present"
    )
    merged["asset_novelty"] = np.where(
        merged["seen_asset_in_training"].astype(bool),
        "seen_in_training",
        "unseen_equipment",
    )
    merged["RegionCode"] = merged["RegionCode"].fillna("unknown").astype(str)
    merged["InventoryGroupCategory"] = (
        merged["InventoryGroupCategory"].fillna("unknown").astype(str)
    )

    summary_rows = []
    segment_rows: list[dict[str, object]] = []
    for model in ["lightgbm", "pytorch"]:
        summary_rows.append(
            {
                "model": model,
                "rows": len(merged),
                **price_metrics(
                    merged["TargetValue"].to_numpy(),
                    merged[f"{model}_prediction"].to_numpy(),
                ),
            }
        )
        for segment_type, column in [
            ("equipment_age", "age_band"),
            ("observed_price_tercile", "price_band"),
            ("region", "RegionCode"),
            ("equipment_category", "InventoryGroupCategory"),
            ("operational_hours", "hours_availability"),
            ("asset_novelty", "asset_novelty"),
        ]:
            segment_rows.extend(_segment_metrics(merged, model, segment_type, column))

    summary = pd.DataFrame(summary_rows).sort_values("rmsle")
    segments = pd.DataFrame(segment_rows).sort_values(
        ["segment_type", "segment", "rmsle"]
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(reports_dir / "model_error_summary.csv", index=False)
    segments.to_csv(reports_dir / "error_analysis.csv", index=False)

    selected = segments.loc[
        segments["segment_type"].isin(
            ["equipment_age", "operational_hours", "asset_novelty"]
        )
    ].copy()
    selected["label"] = selected["segment_type"] + ": " + selected["segment"]
    pivot = selected.pivot(index="label", columns="model", values="rmsle")
    axis = pivot.plot(kind="barh", figsize=(10, 6), color=["#2f6690", "#f4a261"])
    axis.set_title("Out-of-Fold Error Patterns (Descriptive, Not Causal)")
    axis.set_xlabel("RMSLE (lower is better)")
    axis.set_ylabel("")
    axis.grid(axis="x", alpha=0.25)
    axis.figure.tight_layout()
    axis.figure.savefig(reports_dir / "error_analysis.png", dpi=160)
    plt.close(axis.figure)

    notes = {
        "interpretation_boundary": (
            "These are associations and failure patterns in out-of-fold predictions. "
            "They do not prove why an error occurred or establish causal effects."
        ),
        "price_band_note": (
            "Observed-price terciles are created only after prediction for descriptive "
            "error analysis and are never model inputs."
        ),
        "comparison_rows": len(merged),
    }
    (reports_dir / "error_analysis_notes.json").write_text(
        json.dumps(notes, indent=2), encoding="utf-8"
    )
    return summary, segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lightgbm", type=Path, default=DEFAULT_LIGHTGBM)
    parser.add_argument("--pytorch", type=Path, default=DEFAULT_PYTORCH)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, _ = analyze(args.lightgbm, args.pytorch, args.reports_dir)
    print("Fair OOF comparison on matching time-validation rows:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
