"""Train a small, explainable PyTorch MLP on the same time-aware folds."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch import nn

from src.features import FeatureEngineer, build_preprocessor
from src.metrics import from_log_predictions, price_metrics
from src.validation import asset_overlap_details, expanding_date_folds

DEFAULT_INPUT = Path("data/processed/model_data.csv")
DEFAULT_REPORTS = Path("reports")
DEFAULT_MODELS = Path("models")
RANDOM_SEED = 42


class PriceMLP(nn.Module):
    """Beginner-readable network for log-price regression."""

    def __init__(self, input_features: int, dropout: float = 0.20) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_reproducible_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _dense_batch(matrix: Any, indices: np.ndarray) -> np.ndarray:
    batch = matrix[indices]
    if sparse.issparse(batch):
        batch = batch.toarray()
    return np.asarray(batch, dtype=np.float32)


def _batch_indices(
    row_count: int, batch_size: int, shuffle: bool, seed: int
) -> Iterator[np.ndarray]:
    indices = np.arange(row_count)
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    for start in range(0, row_count, batch_size):
        yield indices[start : start + batch_size]


def predict_log_prices(
    model: PriceMLP,
    matrix: Any,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for indices in _batch_indices(
            int(matrix.shape[0]), batch_size, False, RANDOM_SEED
        ):
            features = torch.from_numpy(_dense_batch(matrix, indices)).to(device)
            outputs.append(model(features).detach().cpu().numpy())
    return np.concatenate(outputs)


def train_one_fold(
    x_train: Any,
    y_train_scaled: np.ndarray,
    x_validation: Any,
    y_validation_scaled: np.ndarray,
    *,
    device: torch.device,
    max_epochs: int,
    batch_size: int,
    patience: int,
) -> tuple[PriceMLP, list[dict[str, float]], np.ndarray]:
    """Train with early stopping and return the best validation state."""

    set_reproducible_seed()
    model = PriceMLP(int(x_train.shape[1])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_function = nn.MSELoss()
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_squared_error = 0.0
        total_rows = 0
        for indices in _batch_indices(
            int(x_train.shape[0]), batch_size, True, RANDOM_SEED + epoch
        ):
            features = torch.from_numpy(_dense_batch(x_train, indices)).to(device)
            target = torch.from_numpy(
                np.asarray(y_train_scaled[indices], dtype=np.float32)
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(features)
            loss = loss_function(predictions, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_squared_error += float(loss.detach().cpu()) * len(indices)
            total_rows += len(indices)

        validation_predictions = predict_log_prices(
            model, x_validation, device, batch_size
        )
        validation_loss = float(
            np.mean((validation_predictions - y_validation_scaled) ** 2)
        )
        training_loss = total_squared_error / max(total_rows, 1)
        history.append(
            {
                "epoch": float(epoch),
                "training_scaled_log_mse": training_loss,
                "validation_scaled_log_mse": validation_loss,
            }
        )

        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError(
            "PyTorch training completed without producing a model state."
        )
    model.load_state_dict(best_state)
    best_predictions = predict_log_prices(model, x_validation, device, batch_size)
    return model, history, best_predictions


def train_fixed_epochs(
    matrix: Any,
    scaled_target: np.ndarray,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
) -> PriceMLP:
    """Retrain the chosen architecture on all history for a usable artifact."""

    set_reproducible_seed()
    model = PriceMLP(int(matrix.shape[1])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_function = nn.MSELoss()
    for epoch in range(1, epochs + 1):
        model.train()
        for indices in _batch_indices(
            int(matrix.shape[0]), batch_size, True, RANDOM_SEED + epoch
        ):
            features = torch.from_numpy(_dense_batch(matrix, indices)).to(device)
            target = torch.from_numpy(
                np.asarray(scaled_target[indices], dtype=np.float32)
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
    return model


def _plot_history(history: list[dict[str, float]], output_path: Path) -> None:
    frame = pd.DataFrame(history)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(frame["epoch"], frame["training_scaled_log_mse"], label="Training")
    axis.plot(frame["epoch"], frame["validation_scaled_log_mse"], label="Validation")
    axis.set_title("PyTorch MLP Training Curve (Final Time Fold)")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Standardized log-price MSE")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def train_pytorch(
    data: pd.DataFrame,
    reports_dir: Path,
    models_dir: Path,
    *,
    n_splits: int = 3,
    max_epochs: int = 25,
    batch_size: int = 1024,
    patience: int = 4,
    requested_device: str = "auto",
) -> dict[str, Any]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    data = data.sort_values(["TransactionDate", "TransactionID"]).reset_index(drop=True)
    target = pd.to_numeric(data["TargetValue"], errors="raise").to_numpy(float)
    device = choose_device(requested_device)

    fold_results: list[dict[str, Any]] = []
    oof_rows: list[pd.DataFrame] = []
    last_artifact: dict[str, Any] | None = None

    for fold in expanding_date_folds(data, n_splits=n_splits):
        train_data = data.iloc[fold.train_indices].copy()
        validation_data = data.iloc[fold.validation_indices].copy()
        y_train = target[fold.train_indices]
        y_validation = target[fold.validation_indices]

        feature_engineer = FeatureEngineer()
        train_features = feature_engineer.fit_transform(train_data)
        validation_features = feature_engineer.transform(validation_data)
        preprocessor = build_preprocessor()
        x_train = preprocessor.fit_transform(train_features)
        x_validation = preprocessor.transform(validation_features)

        y_train_log = np.log1p(y_train)
        y_validation_log = np.log1p(y_validation)
        target_mean = float(y_train_log.mean())
        target_std = float(y_train_log.std())
        if target_std <= 0:
            raise ValueError("Training log-price has zero variance; MLP cannot train.")

        _model, history, scaled_predictions = train_one_fold(
            x_train,
            (y_train_log - target_mean) / target_std,
            x_validation,
            (y_validation_log - target_mean) / target_std,
            device=device,
            max_epochs=max_epochs,
            batch_size=batch_size,
            patience=patience,
        )
        # Dense networks can extrapolate far beyond observed prices for unusual
        # feature combinations. Training-only bounds prevent impossible numeric
        # explosions without looking at validation targets.
        log_predictions = np.clip(
            scaled_predictions * target_std + target_mean,
            y_train_log.min(),
            y_train_log.max(),
        )
        predictions = from_log_predictions(log_predictions)
        seen_asset_mask, overlap = asset_overlap_details(train_data, validation_data)
        all_metrics = price_metrics(y_validation, predictions)
        unseen = ~seen_asset_mask.to_numpy()
        unseen_metrics = (
            price_metrics(y_validation[unseen], predictions[unseen])
            if unseen.sum() >= 2
            else None
        )
        fold_results.append(
            {
                "fold": fold.number,
                "training_rows": len(train_data),
                "validation_rows": len(validation_data),
                "train_end": fold.train_end.date().isoformat(),
                "validation_start": fold.validation_start.date().isoformat(),
                "epochs_completed": len(history),
                "best_epoch": int(
                    min(history, key=lambda row: row["validation_scaled_log_mse"])[
                        "epoch"
                    ]
                ),
                "all_validation": all_metrics,
                "unseen_asset_only": unseen_metrics,
                "asset_overlap": overlap,
            }
        )

        oof = validation_data[
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
        oof["fold"] = fold.number
        oof["seen_asset_in_training"] = seen_asset_mask.to_numpy()
        oof["prediction"] = predictions
        oof_rows.append(oof)

        last_artifact = {
            "history": history,
        }
        print(
            f"Fold {fold.number}: {len(history)} epoch(s), PyTorch RMSLE "
            f"{all_metrics['rmsle']:.4f} on {len(validation_data):,} future rows"
        )

    if last_artifact is None:
        raise RuntimeError("No validation folds were produced.")

    oof_predictions = pd.concat(oof_rows, ignore_index=True)
    oof_predictions.to_csv(reports_dir / "pytorch_oof_predictions.csv", index=False)
    fold_table = pd.DataFrame(
        [
            {
                "fold": result["fold"],
                "training_rows": result["training_rows"],
                "validation_rows": result["validation_rows"],
                "epochs_completed": result["epochs_completed"],
                "best_epoch": result["best_epoch"],
                **result["all_validation"],
            }
            for result in fold_results
        ]
    )
    fold_table.to_csv(reports_dir / "pytorch_fold_metrics.csv", index=False)

    summary = {
        "device": str(device),
        "architecture": "input -> 128 ReLU -> Dropout(0.20) -> 64 ReLU -> 1",
        "target_transform": "log1p",
        "validation": f"{n_splits} expanding unique-date folds",
        "folds": fold_results,
        "mean_metrics": {
            metric: float(fold_table[metric].mean())
            for metric in ["rmsle", "mae", "rmse", "r2"]
        },
        "saved_model_note": (
            "Cross-validation chooses a typical best epoch count; pytorch_mlp.pt "
            "is then retrained on all authorised history. OOF metrics still come "
            "only from untouched future folds and are not deployment claims."
        ),
    }
    _plot_history(last_artifact["history"], reports_dir / "pytorch_training_curve.png")
    final_feature_engineer = FeatureEngineer()
    final_features = final_feature_engineer.fit_transform(data)
    final_preprocessor = build_preprocessor()
    x_all = final_preprocessor.fit_transform(final_features)
    all_log_target = np.log1p(target)
    final_target_mean = float(all_log_target.mean())
    final_target_std = float(all_log_target.std())
    final_target_min = float(all_log_target.min())
    final_target_max = float(all_log_target.max())
    final_epochs = max(
        1, int(np.median([result["best_epoch"] for result in fold_results]))
    )
    final_model = train_fixed_epochs(
        x_all,
        (all_log_target - final_target_mean) / final_target_std,
        device=device,
        epochs=final_epochs,
        batch_size=batch_size,
    )
    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "input_features": int(x_all.shape[1]),
            "training_rows": len(data),
            "training_epochs": final_epochs,
            "architecture": summary["architecture"],
            "target_log_mean": final_target_mean,
            "target_log_std": final_target_std,
            "target_log_min": final_target_min,
            "target_log_max": final_target_max,
        },
        models_dir / "pytorch_mlp.pt",
    )
    joblib.dump(
        final_feature_engineer,
        models_dir / "pytorch_feature_engineer.joblib",
    )
    joblib.dump(final_preprocessor, models_dir / "pytorch_preprocessor.joblib")
    summary["final_training"] = {
        "rows": len(data),
        "epochs": final_epochs,
        "input_features": int(x_all.shape[1]),
        "prediction_log_bounds": [final_target_min, final_target_max],
    }
    (reports_dir / "pytorch_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"Model data not found at '{args.input}'. Run 'python -m "
            "src.prepare_data' first."
        )
    data = pd.read_csv(args.input, low_memory=False)
    summary = train_pytorch(
        data,
        args.reports_dir,
        args.models_dir,
        n_splits=args.splits,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        requested_device=args.device,
    )
    print("\nMean PyTorch metrics:")
    print(json.dumps(summary["mean_metrics"], indent=2))


if __name__ == "__main__":
    main()
