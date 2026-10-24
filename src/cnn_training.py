"""Loop de treino, avaliação e visualizações para a CNN de mamografia."""

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

FIGURES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
MODELS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "models"


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


@dataclass
class EpochHistory:
    train_loss: list = field(default_factory=list)
    train_acc: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    val_acc: list = field(default_factory=list)
    val_recall_malignant: list = field(default_factory=list)


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss, all_preds, all_labels = 0.0, [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()

            logits = model(images).view(-1)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    recall_malignant = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    return avg_loss, acc, recall_malignant


def train_model(model, train_loader, val_loader, device, epochs: int, lr: float = 1e-3,
                 pos_weight: float | None = None, checkpoint_name: str = "cnn_best_model.pt",
                 early_stopping_patience: int | None = None) -> EpochHistory:
    """Treina o modelo, salvando o checkpoint com o MENOR val_loss (critério padrão e
    mais robusto que recall puro, que tende a selecionar modelos degenerados
    enviesados para a classe positiva)."""
    pos_weight_tensor = torch.tensor([pos_weight], device=device) if pos_weight is not None else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    history = EpochHistory()
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, _ = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_recall = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        history.train_loss.append(train_loss)
        history.train_acc.append(train_acc)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)
        history.val_recall_malignant.append(val_recall)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODELS_DIR / checkpoint_name)
        else:
            epochs_without_improvement += 1

        marker = " *" if improved else ""
        print(f"Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_recall_malignant={val_recall:.4f}{marker}")

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping: sem melhora de val_loss por {early_stopping_patience} épocas.")
            break

    return history


@torch.no_grad()
def evaluate_on_test(model, test_loader, device) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in test_loader:
        images = images.to(device)
        logits = model(images).view(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= 0.5).astype(int)

    metrics = {
        "test_accuracy": accuracy_score(all_labels, all_preds),
        "test_precision_malignant": precision_score(all_labels, all_preds, pos_label=1, zero_division=0),
        "test_recall_malignant": recall_score(all_labels, all_preds, pos_label=1, zero_division=0),
        "test_f1_malignant": f1_score(all_labels, all_preds, pos_label=1, zero_division=0),
        "test_roc_auc": roc_auc_score(all_labels, all_probs),
    }
    return metrics, all_labels, all_preds, all_probs


def plot_training_curves(history: EpochHistory, save: bool = True, prefix: str = "cnn") -> Path | None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = range(1, len(history.train_loss) + 1)

    axes[0].plot(epochs, history.train_loss, label="treino")
    axes[0].plot(epochs, history.val_loss, label="validação")
    axes[0].set_title("Loss por época")
    axes[0].set_xlabel("Época")
    axes[0].legend()

    axes[1].plot(epochs, history.train_acc, label="acurácia treino")
    axes[1].plot(epochs, history.val_acc, label="acurácia validação")
    axes[1].plot(epochs, history.val_recall_malignant, label="recall maligno (val)", linestyle="--")
    axes[1].set_title("Acurácia / Recall por época")
    axes[1].set_xlabel("Época")
    axes[1].legend()

    fig.tight_layout()
    if save:
        return _save(fig, f"{prefix}_01_training_curves.png")
    plt.show()
    return None


def plot_cnn_confusion_matrix(labels: np.ndarray, preds: np.ndarray, save: bool = True, prefix: str = "cnn") -> Path | None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        labels, preds, display_labels=["benign", "malignant"], cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_title("Matriz de confusão — CNN (conjunto de teste)")
    if save:
        return _save(fig, f"{prefix}_02_confusion_matrix.png")
    plt.show()
    return None


def plot_cnn_roc_curve(labels: np.ndarray, probs: np.ndarray, save: bool = True, prefix: str = "cnn") -> Path | None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    RocCurveDisplay.from_predictions(labels, probs, ax=ax, name="CNN")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="baseline aleatório")
    ax.set_title("Curva ROC — CNN")
    ax.legend()
    if save:
        return _save(fig, f"{prefix}_03_roc_curve.png")
    plt.show()
    return None


def plot_sample_predictions(dataset, labels, preds, probs, n: int = 12, save: bool = True, prefix: str = "cnn") -> Path | None:
    label_names = {0: "benign", 1: "malignant"}
    rng = np.random.default_rng(42)
    idx = rng.choice(len(labels), size=min(n, len(labels)), replace=False)

    n_cols = 4
    n_rows = (len(idx) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for ax, i in zip(axes, idx):
        image_tensor, _ = dataset[i]
        if image_tensor.shape[0] == 3:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = (image_tensor * std + mean).permute(1, 2, 0).clamp(0, 1).numpy()
            ax.imshow(image)
        else:
            image = image_tensor.squeeze(0).numpy() * 0.5 + 0.5
            ax.imshow(image, cmap="gray")
        correct = labels[i] == preds[i]
        color = "green" if correct else "red"
        ax.set_title(
            f"real: {label_names[labels[i]]} | previsto: {label_names[preds[i]]} ({probs[i]:.2f})",
            color=color, fontsize=9,
        )
        ax.axis("off")

    for ax in axes[len(idx):]:
        ax.set_visible(False)

    fig.suptitle("Exemplos de previsões no conjunto de teste (verde = correto, vermelho = erro)", y=1.02)
    fig.tight_layout()
    if save:
        return _save(fig, f"{prefix}_04_sample_predictions.png")
    plt.show()
    return None
