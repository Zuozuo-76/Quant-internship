#!/usr/bin/env python3
"""Train a multi-stock PyTorch LSTM with strict walk-forward evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "quant-assignment-matplotlib"),
)

import matplotlib  # noqa: E402
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


FEATURES = [
    "log_return",
    "range",
    "log_volume",
    "buy_sell_imbalance",
    "time_position",
]


class TorchLSTM(nn.Module):
    """A single-layer LSTM followed by a binary classification head."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _state = self.lstm(x)
        return self.classifier(sequence[:, -1]).squeeze(-1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def read_field(path: Path, codes: list[str]) -> pd.DataFrame:
    table = pd.read_csv(path, usecols=["minute", *codes])
    table["minute"] = table["minute"].astype(int)
    return table.set_index("minute")[codes].astype(float)


def select_codes(
    first_close_file: Path,
    requested_codes: list[str] | None,
    max_codes: int,
) -> list[str]:
    available = pd.read_csv(first_close_file, nrows=0).columns.tolist()[1:]
    available = [str(code).zfill(6) for code in available]
    if requested_codes:
        codes = [str(code).zfill(6) for code in requested_codes]
        missing = sorted(set(codes) - set(available))
        if missing:
            raise ValueError(f"Codes not present in minute tables: {missing}")
    else:
        codes = available[:max_codes]
    if len(codes) < 2:
        raise ValueError("Walk-forward LSTM requires at least two stocks")
    return codes


def make_dataset(
    processed: Path,
    requested_codes: list[str] | None,
    max_codes: int,
    max_dates: int,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    close_files = sorted((processed / "minute" / "close").glob("*.csv"))
    if max_dates > 0:
        close_files = close_files[:max_dates]
    if not close_files:
        raise FileNotFoundError("No minute close tables")
    codes = select_codes(close_files[0], requested_codes, max_codes)

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    metadata_parts: list[pd.DataFrame] = []
    for number, close_file in enumerate(close_files, start=1):
        date = close_file.stem
        fields = {
            "close": read_field(close_file, codes),
            "high": read_field(
                processed / "minute" / "high" / f"{date}.csv", codes
            ),
            "low": read_field(
                processed / "minute" / "low" / f"{date}.csv", codes
            ),
            "volume": read_field(
                processed / "minute" / "volume" / f"{date}.csv", codes
            ),
            "buy": read_field(
                processed / "minute" / "buy_volume" / f"{date}.csv", codes
            ),
            "sell": read_field(
                processed / "minute" / "sell_volume" / f"{date}.csv", codes
            ),
        }
        minutes = fields["close"].index.to_numpy(int)
        for code in codes:
            close = fields["close"][code]
            features = pd.DataFrame(index=close.index)
            features["log_return"] = np.log(close.clip(lower=1e-12)).diff().fillna(0)
            features["range"] = (
                fields["high"][code] - fields["low"][code]
            ) / close.clip(lower=1e-12)
            features["log_volume"] = np.log1p(fields["volume"][code])
            features["buy_sell_imbalance"] = (
                fields["buy"][code] - fields["sell"][code]
            ) / fields["volume"][code].clip(lower=1)
            features["time_position"] = np.linspace(0, 1, len(features))
            values = (
                features[FEATURES]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
                .to_numpy(float)
            )
            prices = close.to_numpy(float)
            ends = np.arange(sequence_length - 1, len(values) - 1)
            if not len(ends):
                continue
            sequences = np.stack([
                values[end - sequence_length + 1:end + 1]
                for end in ends
            ])
            labels = (prices[ends + 1] > prices[ends]).astype(np.float32)
            x_parts.append(sequences.astype(np.float32))
            y_parts.append(labels)
            metadata_parts.append(pd.DataFrame({
                "date": date,
                "code": code,
                "minute": minutes[ends + 1],
                "current_close": prices[ends],
                "next_close": prices[ends + 1],
            }))
        if number % 20 == 0 or number == len(close_files):
            print(
                f"prepared {number}/{len(close_files)} dates for {len(codes)} stocks",
                flush=True,
            )
    if not x_parts:
        raise ValueError("No LSTM samples were created")
    return (
        np.concatenate(x_parts).astype(np.float32),
        np.concatenate(y_parts).astype(np.float32),
        pd.concat(metadata_parts, ignore_index=True),
        codes,
    )


def make_walk_forward_folds(
    dates: list[str],
    min_train_dates: int,
    validation_dates: int,
    test_dates: int,
    step_dates: int,
) -> list[dict[str, object]]:
    unique_dates = sorted(dict.fromkeys(str(date) for date in dates))
    folds = []
    train_end = min_train_dates
    fold_number = 1
    while train_end + validation_dates + test_dates <= len(unique_dates):
        validation_end = train_end + validation_dates
        test_end = validation_end + test_dates
        folds.append({
            "fold": fold_number,
            "train_dates": unique_dates[:train_end],
            "validation_dates": unique_dates[train_end:validation_end],
            "test_dates": unique_dates[validation_end:test_end],
        })
        fold_number += 1
        train_end += step_dates
    if len(folds) < 2:
        raise ValueError(
            "Walk-forward configuration creates fewer than two test folds; "
            "increase --max-dates or reduce fold windows"
        )
    return folds


def predict_proba(
    model: TorchLSTM,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x)),
        batch_size=batch_size,
        shuffle=False,
    )
    pieces = []
    model.eval()
    with torch.inference_mode():
        for (batch_x,) in loader:
            probability = torch.sigmoid(model(batch_x.to(device)))
            pieces.append(probability.cpu().numpy())
    return np.concatenate(pieces)


def metrics_from_probability(
    y: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    prediction = probability >= 0.5
    clipped = probability.clip(1e-8, 1 - 1e-8)
    loss = float(np.mean(
        -(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))
    ))
    auc = (
        roc_auc_score(y, probability)
        if len(np.unique(y)) > 1
        else math.nan
    )
    pr_auc = (
        average_precision_score(y, probability)
        if len(np.unique(y)) > 1
        else math.nan
    )
    tn, fp, fn, tp = confusion_matrix(
        y.astype(int), prediction.astype(int), labels=[0, 1]
    ).ravel()
    return {
        "positive_rate": float(y.mean()),
        "majority_baseline_accuracy": max(float(y.mean()), 1 - float(y.mean())),
        "loss": loss,
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "auc": float(auc),
        "pr_auc": float(pr_auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def sequence_summary_features(x: np.ndarray) -> np.ndarray:
    """Create a simple 15-column sequence summary for logistic regression."""
    if x.ndim != 3:
        raise ValueError("Expected [sample, time, feature] sequence input")
    return np.concatenate([x[:, -1], x.mean(axis=1), x.std(axis=1)], axis=1)


def write_lstm_figures(
    comparison: pd.DataFrame,
    lstm_fold_metrics: pd.DataFrame,
    baseline_fold_metrics: pd.DataFrame,
    output: Path,
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    metrics = ["balanced_accuracy", "auc", "pr_auc", "f1"]
    positions = np.arange(len(metrics))
    width = 0.24
    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    for index, row in enumerate(comparison.itertuples(index=False)):
        values = [getattr(row, metric) for metric in metrics]
        axis.bar(positions + (index - 1) * width, values, width, label=row.model)
    axis.set_xticks(positions, metrics, rotation=15)
    axis.set_ylim(0, 1)
    axis.set_title("Walk-forward out-of-sample model comparison")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figures / "model_comparison.png", dpi=160)
    plt.close(figure)

    # Accuracy needs a dedicated diagnostic because the up/down labels are
    # imbalanced.  Overall accuracy alone can look high even when a model
    # mostly predicts the majority (down) class, so show balanced accuracy and
    # the fold-level majority baseline beside it.
    model_labels = {
        "majority_class": "Majority class",
        "logistic_regression": "Logistic regression",
        "lstm": "LSTM",
    }
    ordered_models = ["majority_class", "logistic_regression", "lstm"]
    overall_accuracy = comparison.set_index("model").loc[ordered_models]
    test_lstm = lstm_fold_metrics.loc[
        lstm_fold_metrics["split"].eq("test")
    ].copy()
    test_logistic = baseline_fold_metrics.loc[
        baseline_fold_metrics["split"].eq("test")
        & baseline_fold_metrics["model"].eq("logistic_regression")
    ].copy()
    majority_by_fold = test_lstm[["fold", "majority_baseline_accuracy"]].rename(
        columns={"majority_baseline_accuracy": "accuracy"}
    )
    majority_by_fold["model"] = "majority_class"
    fold_accuracy = pd.concat(
        [
            majority_by_fold[["fold", "model", "accuracy"]],
            test_logistic[["fold", "model", "accuracy"]],
            test_lstm.assign(model="lstm")[["fold", "model", "accuracy"]],
        ],
        ignore_index=True,
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    overall_metrics = ["accuracy", "balanced_accuracy"]
    positions = np.arange(len(overall_metrics))
    width = 0.24
    for index, model in enumerate(ordered_models):
        row = overall_accuracy.loc[model]
        values = [float(row[metric]) for metric in overall_metrics]
        bars = axes[0].bar(
            positions + (index - 1) * width,
            values,
            width,
            label=model_labels[model],
        )
        axes[0].bar_label(bars, labels=[f"{value:.2%}" for value in values], fontsize=8)
    axes[0].set_xticks(positions, ["Accuracy", "Balanced accuracy"])
    axes[0].set_ylim(0, 0.80)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Combined out-of-sample accuracy")
    axes[0].legend(fontsize=8, loc="lower left")

    for model in ordered_models:
        group = fold_accuracy.loc[fold_accuracy["model"].eq(model)].sort_values("fold")
        axes[1].plot(
            group["fold"],
            group["accuracy"],
            marker="o",
            linewidth=1.8,
            label=model_labels[model],
        )
        # The three curves are separated by only a few basis points.  Label
        # the LSTM curve only so the exact values remain readable instead of
        # stacking nine nearly identical annotations on top of one another.
        if model == "lstm":
            for fold, value in zip(group["fold"], group["accuracy"], strict=True):
                axes[1].annotate(
                    f"LSTM {value:.2%}",
                    (fold, value),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color="tab:green",
                )
    axes[1].set_xticks(sorted(fold_accuracy["fold"].unique()))
    axes[1].set_ylim(0.66, 0.73)
    axes[1].set_xlabel("Walk-forward test fold")
    axes[1].set_ylabel("Accuracy (zoomed scale)")
    axes[1].set_title("Test accuracy by fold")
    axes[1].legend(fontsize=8, loc="upper right")
    figure.suptitle("LSTM accuracy diagnostic", fontsize=12)
    figure.tight_layout()
    figure.savefig(figures / "lstm_accuracy_diagnostics.png", dpi=180)
    plt.close(figure)

    lstm_row = comparison.loc[comparison["model"].eq("lstm")].iloc[0]
    matrix = np.array([[lstm_row.tn, lstm_row.fp], [lstm_row.fn, lstm_row.tp]], dtype=int)
    figure, axis = plt.subplots(figsize=(5.4, 4.8))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], ["pred down", "pred up"])
    axis.set_yticks([0, 1], ["actual down", "actual up"])
    for row in range(2):
        for column in range(2):
            axis.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center")
    axis.set_title("LSTM out-of-sample confusion matrix")
    figure.colorbar(image, ax=axis, shrink=0.82)
    figure.tight_layout()
    figure.savefig(figures / "lstm_confusion_matrix.png", dpi=160)
    plt.close(figure)

    fold_table = pd.concat([
        lstm_fold_metrics.loc[lstm_fold_metrics["split"].eq("test")].assign(model="lstm"),
        baseline_fold_metrics.loc[baseline_fold_metrics["split"].eq("test")],
    ], ignore_index=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharex=True)
    for axis, metric in zip(axes, ("auc", "pr_auc"), strict=True):
        for model, group in fold_table.groupby("model", sort=True):
            axis.plot(group["fold"], group[metric], marker="o", label=model)
        axis.set_title(f"Test {metric} by fold")
        axis.set_xlabel("Fold")
        axis.set_ylim(0, 1)
        axis.set_xticks(sorted(fold_table["fold"].unique()))
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figures / "fold_model_metrics.png", dpi=160)
    plt.close(figure)


def train_fold(
    fold: dict[str, object],
    x: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    hidden_size: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[
    list[dict[str, float]],
    list[dict[str, float]],
    list[dict[str, float]],
    pd.DataFrame,
    dict[str, object],
]:
    fold_number = int(fold["fold"])
    date_values = metadata["date"].astype(str)
    masks = {
        "train": date_values.isin(fold["train_dates"]).to_numpy(),
        "validation": date_values.isin(fold["validation_dates"]).to_numpy(),
        "test": date_values.isin(fold["test_dates"]).to_numpy(),
    }
    split_arrays = {
        split: (x[mask], y[mask], metadata.loc[mask].reset_index(drop=True))
        for split, mask in masks.items()
    }
    x_train, y_train, _ = split_arrays["train"]
    mean = x_train.reshape(-1, x.shape[-1]).mean(axis=0)
    std = x_train.reshape(-1, x.shape[-1]).std(axis=0)
    std[std < 1e-10] = 1
    standardized = {
        split: (
            ((split_x - mean) / std).astype(np.float32),
            split_y,
            split_meta,
        )
        for split, (split_x, split_y, split_meta) in split_arrays.items()
    }

    seed_everything(seed + fold_number)
    model = TorchLSTM(x.shape[-1], hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed + fold_number)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(standardized["train"][0]),
            torch.from_numpy(standardized["train"][1]),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    print(
        f"fold={fold_number} device={device.type} "
        f"train={len(standardized['train'][0])} "
        f"validation={len(standardized['validation'][0])} "
        f"test={len(standardized['test'][0])}",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        train_probability = predict_proba(
            model, standardized["train"][0], device, batch_size
        )
        validation_probability = predict_proba(
            model, standardized["validation"][0], device, batch_size
        )
        train_metrics = metrics_from_probability(
            standardized["train"][1], train_probability
        )
        validation_metrics = metrics_from_probability(
            standardized["validation"][1], validation_probability
        )
        history.append({
            "fold": fold_number,
            "epoch": epoch,
            "batch_loss": float(np.mean(batch_losses)),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": validation_metrics["loss"],
            "val_accuracy": validation_metrics["accuracy"],
            "val_auc": validation_metrics["auc"],
        })
        if validation_metrics["loss"] < best_val_loss:
            best_val_loss = validation_metrics["loss"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(
            f"fold {fold_number} epoch {epoch:02d}: "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"val_auc={validation_metrics['auc']:.4f}",
            flush=True,
        )
    if best_state is None:
        raise RuntimeError(f"Fold {fold_number} produced no checkpoint")
    model.load_state_dict(best_state)

    metrics_rows = []
    probabilities = {}
    for split, (split_x, split_y, _split_meta) in standardized.items():
        probability = predict_proba(model, split_x, device, batch_size)
        probabilities[split] = probability
        metrics_rows.append({
            "fold": fold_number,
            "model": "lstm",
            "split": split,
            "n_samples": len(split_y),
            **metrics_from_probability(split_y, probability),
        })

    logistic = LogisticRegression(
        max_iter=500,
        solver="lbfgs",
        random_state=seed + fold_number,
    )
    logistic.fit(
        sequence_summary_features(standardized["train"][0]),
        standardized["train"][1].astype(int),
    )
    baseline_rows = []
    logistic_probabilities = {}
    for split, (split_x, split_y, _split_meta) in standardized.items():
        probability = logistic.predict_proba(sequence_summary_features(split_x))[:, 1]
        logistic_probabilities[split] = probability
        baseline_rows.append({
            "fold": fold_number,
            "model": "logistic_regression",
            "split": split,
            "n_samples": len(split_y),
            **metrics_from_probability(split_y, probability),
        })
    test_x, test_y, test_meta = standardized["test"]
    predictions = test_meta.copy()
    predictions.insert(0, "fold", fold_number)
    predictions["label_up"] = test_y.astype(int)
    predictions["probability_up"] = probabilities["test"]
    predictions["prediction_up"] = (
        probabilities["test"] >= 0.5
    ).astype(int)
    predictions["logistic_probability_up"] = logistic_probabilities["test"]
    predictions["logistic_prediction_up"] = (
        logistic_probabilities["test"] >= 0.5
    ).astype(int)

    checkpoint = {
        "model_type": "torch.nn.LSTM",
        "model_state_dict": best_state,
        "input_size": len(FEATURES),
        "hidden_size": hidden_size,
        "feature_mean": torch.from_numpy(mean.copy()),
        "feature_std": torch.from_numpy(std.copy()),
        "features": FEATURES,
        "sequence_length": int(x.shape[1]),
        "fold": fold_number,
        "best_epoch": best_epoch,
        "train_start": str(fold["train_dates"][0]),
        "train_end": str(fold["train_dates"][-1]),
        "validation_start": str(fold["validation_dates"][0]),
        "validation_end": str(fold["validation_dates"][-1]),
        "test_start": str(fold["test_dates"][0]),
        "test_end": str(fold["test_dates"][-1]),
        "device_used": device.type,
    }
    return history, metrics_rows, baseline_rows, predictions, checkpoint


def train_walk_forward(
    processed: Path,
    output: Path,
    requested_codes: list[str] | None,
    max_codes: int,
    max_dates: int,
    sequence_length: int,
    hidden_size: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    requested_device: str,
    min_train_dates: int,
    validation_dates: int,
    test_dates: int,
    step_dates: int,
) -> pd.DataFrame:
    seed_everything(seed)
    device = resolve_device(requested_device)
    x, y, metadata, codes = make_dataset(
        processed,
        requested_codes,
        max_codes,
        max_dates,
        sequence_length,
    )
    unique_dates = sorted(metadata["date"].astype(str).unique().tolist())
    folds = make_walk_forward_folds(
        unique_dates,
        min_train_dates,
        validation_dates,
        test_dates,
        step_dates,
    )
    output.mkdir(parents=True, exist_ok=True)
    fold_dir = output / "walk_forward"
    fold_dir.mkdir(parents=True, exist_ok=True)

    history_parts = []
    metric_rows = []
    baseline_metric_rows = []
    prediction_parts = []
    checkpoints = []
    for fold in folds:
        history, metrics, baseline_metrics, predictions, checkpoint = train_fold(
            fold,
            x,
            y,
            metadata,
            hidden_size,
            epochs,
            learning_rate,
            batch_size,
            seed,
            device,
        )
        history_parts.extend(history)
        metric_rows.extend(metrics)
        baseline_metric_rows.extend(baseline_metrics)
        prediction_parts.append(predictions)
        checkpoints.append(checkpoint)
        torch.save(
            checkpoint,
            fold_dir / f"fold_{int(fold['fold']):02d}.pt",
        )

    history_table = pd.DataFrame(history_parts)
    fold_metrics = pd.DataFrame(metric_rows)
    baseline_fold_metrics = pd.DataFrame(baseline_metric_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    history_table.to_csv(output / "training_history.csv", index=False)
    fold_metrics.to_csv(output / "walk_forward_metrics.csv", index=False)
    baseline_fold_metrics.to_csv(
        output / "baseline_walk_forward_metrics.csv", index=False
    )
    predictions.to_csv(output / "test_predictions.csv", index=False)

    overall = metrics_from_probability(
        predictions["label_up"].to_numpy(float),
        predictions["probability_up"].to_numpy(float),
    )
    overall_table = pd.DataFrame([{
        "split": "walk_forward_oos",
        "n_folds": len(folds),
        "n_codes": len(codes),
        "n_dates": len(unique_dates),
        "n_samples": len(predictions),
        **overall,
    }])
    overall_table.to_csv(output / "metrics.csv", index=False)

    labels = predictions["label_up"].to_numpy(float)
    majority_probability = np.full(len(labels), labels.mean(), dtype=float)
    comparison = pd.DataFrame([
        {
            "model": "majority_class",
            "n_samples": len(labels),
            **metrics_from_probability(labels, majority_probability),
        },
        {
            "model": "logistic_regression",
            "n_samples": len(labels),
            **metrics_from_probability(
                labels,
                predictions["logistic_probability_up"].to_numpy(float),
            ),
        },
        {
            "model": "lstm",
            "n_samples": len(labels),
            **overall,
        },
    ])
    comparison.to_csv(output / "model_comparison.csv", index=False)
    comparison[["model", "tn", "fp", "fn", "tp"]].to_csv(
        output / "confusion_matrix.csv", index=False
    )
    write_lstm_figures(comparison, fold_metrics, baseline_fold_metrics, output)

    final_checkpoint = dict(checkpoints[-1])
    final_checkpoint.update({
        "walk_forward": True,
        "codes": codes,
        "n_dates": len(unique_dates),
        "fold_count": len(folds),
    })
    torch.save(final_checkpoint, output / "model.pt")
    metadata_doc = {
        key: value
        for key, value in final_checkpoint.items()
        if key not in {"model_state_dict", "feature_mean", "feature_std"}
    }
    metadata_doc["feature_mean"] = final_checkpoint["feature_mean"].numpy().tolist()
    metadata_doc["feature_std"] = final_checkpoint["feature_std"].numpy().tolist()
    metadata_doc["date_start"] = unique_dates[0]
    metadata_doc["date_end"] = unique_dates[-1]
    metadata_doc["folds"] = [
        {
            "fold": int(fold["fold"]),
            "train_start": fold["train_dates"][0],
            "train_end": fold["train_dates"][-1],
            "validation_start": fold["validation_dates"][0],
            "validation_end": fold["validation_dates"][-1],
            "test_start": fold["test_dates"][0],
            "test_end": fold["test_dates"][-1],
        }
        for fold in folds
    ]
    (output / "model_metadata.json").write_text(
        json.dumps(metadata_doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    result = overall_table.iloc[0]
    logistic_result = comparison.loc[
        comparison["model"].eq("logistic_regression")
    ].iloc[0]
    report = f"""# 多股票分钟线 LSTM Walk-forward 拓展

- 股票数：{len(codes)}（{', '.join(codes)}）
- 日期范围：{unique_dates[0]} 至 {unique_dates[-1]}，共 {len(unique_dates)} 个交易日
- 输入窗口：过去 {sequence_length} 分钟
- 特征：{', '.join(FEATURES)}
- 模型：PyTorch ``torch.nn.LSTM``（hidden={hidden_size}）+ 线性二分类头
- 验证方式：{len(folds)} 个 expanding-window walk-forward 折
- 每折：至少 {min_train_dates} 个训练日、{validation_dates} 个验证日、{test_dates} 个不重叠测试日
- 标准化和模型训练均在每折内重新估计，不使用测试期信息

## 合并样本外测试

- 样本数：{int(result['n_samples'])}
- 上涨占比：{result['positive_rate']:.2%}
- 多数类基线准确率：{result['majority_baseline_accuracy']:.2%}
- LSTM 准确率：{result['accuracy']:.2%}
- 平衡准确率：{result['balanced_accuracy']:.2%}
- Precision：{result['precision']:.2%}
- Recall：{result['recall']:.2%}
- F1：{result['f1']:.2%}
- ROC AUC：{result['auc']:.4f}
- PR AUC：{result['pr_auc']:.4f}
- 混淆矩阵：TN={int(result['tn'])}, FP={int(result['fp'])}, FN={int(result['fn'])}, TP={int(result['tp'])}

## 简单模型基线

- 逻辑回归准确率：{logistic_result['accuracy']:.2%}
- 逻辑回归平衡准确率：{logistic_result['balanced_accuracy']:.2%}
- 逻辑回归 ROC AUC：{logistic_result['auc']:.4f}
- 逻辑回归 PR AUC：{logistic_result['pr_auc']:.4f}
- 逻辑回归 F1：{logistic_result['f1']:.2%}

这是多股票、长时间和严格滚动样本外结果。是否可交易仍需结合阈值、换手、滑点和延迟进行策略级验证。
"""
    (output / "LSTM实验报告.md").write_text(
        report.replace("`", "\x60"),
        encoding="utf-8",
    )
    return overall_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=Path("processed"))
    parser.add_argument("--output", type=Path, default=Path("lstm"))
    parser.add_argument(
        "--codes",
        help="Comma-separated stock codes; defaults to the first --max-codes columns",
    )
    parser.add_argument("--max-codes", type=int, default=6)
    parser.add_argument("--max-dates", type=int, default=140)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
    )
    parser.add_argument("--min-train-dates", type=int, default=60)
    parser.add_argument("--validation-dates", type=int, default=20)
    parser.add_argument("--test-dates", type=int, default=20)
    parser.add_argument("--step-dates", type=int, default=20)
    args = parser.parse_args()
    requested_codes = (
        [code.strip() for code in args.codes.split(",") if code.strip()]
        if args.codes
        else None
    )
    metrics = train_walk_forward(
        args.processed,
        args.output,
        requested_codes,
        args.max_codes,
        args.max_dates,
        args.sequence_length,
        args.hidden_size,
        args.epochs,
        args.learning_rate,
        args.batch_size,
        args.seed,
        args.device,
        args.min_train_dates,
        args.validation_dates,
        args.test_dates,
        args.step_dates,
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
