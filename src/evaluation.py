"""Avaliação dos modelos treinados: métricas, matriz de confusão e comparação."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.modeling import SplitData, TrainedModel

FIGURES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def evaluate_model(model: TrainedModel, split: SplitData) -> dict:
    y_pred = model.pipeline.predict(split.X_test)
    y_proba = model.pipeline.predict_proba(split.X_test)[:, 1]

    return {
        "model": model.name,
        "cv_accuracy_mean": model.cv_accuracy_mean,
        "cv_accuracy_std": model.cv_accuracy_std,
        "test_accuracy": accuracy_score(split.y_test, y_pred),
        # pos_label=0 -> classe "malignant": é a classe clinicamente crítica
        # (falso negativo = câncer não detectado), por isso recall é calculado
        # em relação a ela e não à classe majoritária "benign".
        "test_precision_malignant": precision_score(split.y_test, y_pred, pos_label=0),
        "test_recall_malignant": recall_score(split.y_test, y_pred, pos_label=0),
        "test_f1_malignant": f1_score(split.y_test, y_pred, pos_label=0),
        "test_roc_auc": roc_auc_score(split.y_test, y_proba),
    }


def evaluate_all_models(models: list[TrainedModel], split: SplitData) -> pd.DataFrame:
    rows = [evaluate_model(m, split) for m in models]
    return pd.DataFrame(rows).sort_values("test_recall_malignant", ascending=False).reset_index(drop=True)


def plot_confusion_matrices(models: list[TrainedModel], split: SplitData, save: bool = True) -> Path | None:
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        y_pred = model.pipeline.predict(split.X_test)
        ConfusionMatrixDisplay.from_predictions(
            split.y_test, y_pred,
            display_labels=["malignant", "benign"],
            cmap="Blues", ax=ax, colorbar=False,
        )
        ax.set_title(model.name)
    fig.suptitle("Matrizes de confusão por modelo (conjunto de teste)", y=1.05)
    fig.tight_layout()
    if save:
        return _save(fig, "05_confusion_matrices.png")
    plt.show()
    return None


def plot_roc_curves(models: list[TrainedModel], split: SplitData, save: bool = True) -> Path | None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for model in models:
        RocCurveDisplay.from_estimator(model.pipeline, split.X_test, split.y_test, ax=ax, name=model.name)
    ax.set_title("Curvas ROC por modelo")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="baseline aleatório")
    ax.legend()
    if save:
        return _save(fig, "06_roc_curves.png")
    plt.show()
    return None


def plot_metrics_comparison(metrics_df: pd.DataFrame, save: bool = True) -> Path | None:
    plot_df = metrics_df.melt(
        id_vars="model",
        value_vars=["test_accuracy", "test_precision_malignant", "test_recall_malignant", "test_f1_malignant", "test_roc_auc"],
        var_name="metric", value_name="value",
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_df, x="metric", y="value", hue="model", ax=ax, palette="viridis")
    ax.set_ylim(0, 1.05)
    ax.set_title("Comparação de métricas entre modelos (conjunto de teste)")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    if save:
        return _save(fig, "07_metrics_comparison.png")
    plt.show()
    return None
