"""Create the compact, deterministic model-data table for milestones 5-9."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.features import prepare_model_data

DEFAULT_INPUT = Path("data/processed/clean_train.csv")
DEFAULT_OUTPUT = Path("data/processed/model_data.csv")


def run(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Cleaned data not found at '{input_path}'. Run 'python -m "
            "src.clean_data' first."
        )
    data = pd.read_csv(input_path, low_memory=False)
    prepared = prepare_model_data(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    print(f"Prepared {len(prepared):,} rows for modelling: {output_path}")
    print(
        "Frequency mappings are intentionally NOT stored here; each validation "
        "fold learns them from its own training rows."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
