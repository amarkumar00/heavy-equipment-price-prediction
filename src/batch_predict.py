"""Chunked batch inference, invalid-row reporting, and honest benchmarks."""

from __future__ import annotations

import argparse
import json
import resource
import time
import tracemalloc
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.predictor import (
    DEFAULT_ARTIFACT_DIR,
    PricePredictor,
    validate_prediction_inputs,
)

DEFAULT_OUTPUT = Path("predictions/equipment_predictions.csv")
DEFAULT_INVALID = Path("predictions/invalid_rows.csv")
DEFAULT_BENCHMARK_INPUT = Path("data/processed/model_data.csv")
DEFAULT_BENCHMARK_REPORT = Path("reports/batch_benchmark.json")


def _process_peak_rss_mb() -> float:
    """Return process lifetime peak RSS with macOS/Linux unit normalization."""

    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    divisor = 1024**2 if peak > 10_000_000 else 1024
    return peak / divisor


def predict_dataframe_in_chunks(
    data: pd.DataFrame,
    predictor: PricePredictor,
    *,
    chunk_size: int = 1_000,
    source_row_number_start: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score a DataFrame in bounded chunks and separate invalid rows."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    predicted_chunks: list[pd.DataFrame] = []
    invalid_chunks: list[pd.DataFrame] = []

    for start in range(0, len(data), chunk_size):
        chunk = data.iloc[start : start + chunk_size].copy()
        row_numbers = pd.Series(
            np.arange(
                start + source_row_number_start,
                start + source_row_number_start + len(chunk),
            ),
            index=chunk.index,
        )
        validation = validate_prediction_inputs(chunk)
        valid_mask = validation["is_valid"]

        if valid_mask.any():
            valid_input = chunk.loc[valid_mask].copy()
            predictions = predictor.predict(valid_input, validate=False)
            safe_input = valid_input.drop(
                columns=["TargetValue", "source_row_number"], errors="ignore"
            )
            combined = pd.concat([safe_input, predictions], axis=1)
            combined.insert(0, "source_row_number", row_numbers.loc[valid_mask])
            predicted_chunks.append(combined)

        if (~valid_mask).any():
            invalid = chunk.loc[~valid_mask].drop(
                columns=["source_row_number"], errors="ignore"
            )
            invalid.insert(0, "source_row_number", row_numbers.loc[~valid_mask])
            invalid["validation_error"] = validation.loc[
                ~valid_mask, "validation_error"
            ]
            invalid_chunks.append(invalid)

    predictions = (
        pd.concat(predicted_chunks, ignore_index=True)
        if predicted_chunks
        else pd.DataFrame()
    )
    invalid = (
        pd.concat(invalid_chunks, ignore_index=True)
        if invalid_chunks
        else pd.DataFrame()
    )
    if len(predictions) + len(invalid) != len(data):
        raise RuntimeError("Batch output row count does not match input row count.")
    return predictions, invalid


def predict_csv(
    input_path: Path,
    output_path: Path,
    invalid_path: Path,
    predictor: PricePredictor,
    *,
    chunk_size: int = 1_000,
) -> dict[str, Any]:
    """Stream a CSV file through the model without loading it all at once."""

    if not input_path.exists():
        raise FileNotFoundError(f"Batch input CSV not found at '{input_path}'.")
    input_columns = list(pd.read_csv(input_path, nrows=0).columns)
    prediction_columns = list(predictor.predict(pd.DataFrame()).columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    invalid_path.parent.mkdir(parents=True, exist_ok=True)
    predicted_rows = 0
    invalid_rows = 0
    input_rows = 0
    wrote_predictions = False
    wrote_invalid = False
    source_row_number_start = 2

    for chunk in pd.read_csv(input_path, chunksize=chunk_size, low_memory=False):
        predictions, invalid = predict_dataframe_in_chunks(
            chunk.reset_index(drop=True),
            predictor,
            chunk_size=chunk_size,
            source_row_number_start=source_row_number_start,
        )
        input_rows += len(chunk)
        source_row_number_start += len(chunk)
        if not predictions.empty:
            predictions.to_csv(
                output_path,
                mode="a" if wrote_predictions else "w",
                header=not wrote_predictions,
                index=False,
            )
            wrote_predictions = True
            predicted_rows += len(predictions)
        if not invalid.empty:
            invalid.to_csv(
                invalid_path,
                mode="a" if wrote_invalid else "w",
                header=not wrote_invalid,
                index=False,
            )
            wrote_invalid = True
            invalid_rows += len(invalid)

    if not wrote_predictions:
        safe_input_columns = [
            column
            for column in input_columns
            if column not in {"TargetValue", "source_row_number"}
        ]
        pd.DataFrame(
            columns=["source_row_number", *safe_input_columns, *prediction_columns]
        ).to_csv(output_path, index=False)
    if not wrote_invalid:
        invalid_columns = [
            column for column in input_columns if column != "source_row_number"
        ]
        pd.DataFrame(
            columns=["source_row_number", *invalid_columns, "validation_error"]
        ).to_csv(invalid_path, index=False)
    if predicted_rows + invalid_rows != input_rows:
        raise RuntimeError("CSV batch output row count does not match input row count.")
    return {
        "input_rows": input_rows,
        "predicted_rows": predicted_rows,
        "invalid_rows": invalid_rows,
        "chunk_size": chunk_size,
        "output_path": str(output_path),
        "invalid_path": str(invalid_path),
    }


def _sizes(values: Iterable[int]) -> list[int]:
    sizes = sorted({int(value) for value in values})
    if not sizes or any(value <= 0 for value in sizes):
        raise ValueError("Benchmark sizes must be positive integers.")
    return sizes


def benchmark_batch_prediction(
    source_data: pd.DataFrame,
    predictor: PricePredictor,
    *,
    sizes: Iterable[int] = (1_000, 10_000, 50_000),
    chunk_size: int = 1_000,
) -> dict[str, Any]:
    """Benchmark chunked inference and report scope/measurement limitations."""

    results: list[dict[str, Any]] = []
    for requested_rows in _sizes(sizes):
        if len(source_data) < requested_rows:
            repeats = int(np.ceil(requested_rows / max(len(source_data), 1)))
            sample = pd.concat([source_data] * repeats, ignore_index=True).iloc[
                :requested_rows
            ]
        else:
            sample = source_data.iloc[:requested_rows].copy()

        tracemalloc.start()
        started = time.perf_counter()
        predicted_rows = 0
        invalid_rows = 0
        for start in range(0, len(sample), chunk_size):
            predictions, invalid = predict_dataframe_in_chunks(
                sample.iloc[start : start + chunk_size],
                predictor,
                chunk_size=chunk_size,
                source_row_number_start=start + 2,
            )
            predicted_rows += len(predictions)
            invalid_rows += len(invalid)
        runtime_seconds = time.perf_counter() - started
        _, python_peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results.append(
            {
                "requested_rows": requested_rows,
                "predicted_rows": predicted_rows,
                "invalid_rows": invalid_rows,
                "runtime_seconds": runtime_seconds,
                "rows_per_second": requested_rows / max(runtime_seconds, 1e-9),
                "python_peak_memory_mb": python_peak_bytes / 1024**2,
                "process_lifetime_peak_rss_mb": _process_peak_rss_mb(),
            }
        )

    return {
        "benchmark_scope": "Single-machine, chunked local inference benchmark",
        "chunk_size": chunk_size,
        "results": results,
        "memory_note": (
            "python_peak_memory_mb measures allocations traced during each run. "
            "process_lifetime_peak_rss_mb is the process lifetime peak and may "
            "include imported libraries. Neither is a distributed-system claim."
        ),
        "claim_boundary": (
            "This demonstrates scale-aware batch inference, not production-scale "
            "deployment or Amazon infrastructure experience."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--invalid-output", type=Path, default=DEFAULT_INVALID)
    parser.add_argument("--chunk-size", type=int, default=1_000)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-input", type=Path, default=DEFAULT_BENCHMARK_INPUT)
    parser.add_argument(
        "--benchmark-report", type=Path, default=DEFAULT_BENCHMARK_REPORT
    )
    parser.add_argument(
        "--benchmark-sizes", type=int, nargs="+", default=[1_000, 10_000, 50_000]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = PricePredictor.load(args.artifacts)
    if args.benchmark:
        if not args.benchmark_input.exists():
            raise FileNotFoundError(
                f"Benchmark input not found at '{args.benchmark_input}'."
            )
        source = pd.read_csv(args.benchmark_input, low_memory=False)
        report = benchmark_batch_prediction(
            source,
            predictor,
            sizes=args.benchmark_sizes,
            chunk_size=args.chunk_size,
        )
        args.benchmark_report.parent.mkdir(parents=True, exist_ok=True)
        args.benchmark_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    if args.input is None:
        raise ValueError("Provide --input for CSV prediction or use --benchmark.")
    summary = predict_csv(
        args.input,
        args.output,
        args.invalid_output,
        predictor,
        chunk_size=args.chunk_size,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
