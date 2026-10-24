"""Explicabilidade dos modelos: feature importance e SHAP."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.modeling import SplitData, TrainedModel

FIGURES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def get_output_feature_names(model: TrainedModel) -> list[str]:
    preprocessor = model.pipeline.named_steps["preprocessor"]
    return [name.split("__", 1)[-1] for name in preprocessor.get_feature_names_out()]


def feature_importance_table(model: TrainedModel) -> pd.DataFrame:
    """Retorna a importância de cada feature: coeficientes (modelos lineares)
    ou feature_importances_ (modelos baseados em árvore)."""
    classifier = model.pipeline.named_steps["classifier"]
    feature_names = get_output_feature_names(model)

    if isinstance(classifier, LogisticRegression):
        values = classifier.coef_[0]
        kind = "coefficient"
    elif isinstance(classifier, (DecisionTreeClassifier, RandomForestClassifier)):
        values = classifier.feature_importances_
        kind = "importance"
    else:
        raise ValueError(f"Modelo {model.name} não expõe importância de features nativamente.")

    df = pd.DataFrame({"feature": feature_names, kind: values})
    return df.sort_values(kind, key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def plot_feature_importance(model: TrainedModel, top_n: int = 15, save: bool = True) -> Path | None:
    table = feature_importance_table(model).head(top_n)
    value_col = "coefficient" if "coefficient" in table.columns else "importance"

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#c0392b" if v < 0 else "#2c7fb8" for v in table[value_col]]
    ax.barh(table["feature"][::-1], table[value_col][::-1], color=colors[::-1])
    ax.set_title(f"Top {top_n} features mais relevantes — {model.name}")
    ax.set_xlabel(value_col)
    fig.tight_layout()
    if save:
        return _save(fig, f"08_feature_importance_{model.name}.png")
    plt.show()
    return None


def compute_shap_values(model: TrainedModel, split: SplitData, max_samples: int = 150):
    """Calcula valores SHAP para um modelo baseado em árvore (mais eficiente
    e exato via TreeExplainer). Retorna (explanation, X_test_transformed_df)."""
    classifier = model.pipeline.named_steps["classifier"]
    if not isinstance(classifier, (DecisionTreeClassifier, RandomForestClassifier)):
        raise ValueError("SHAP TreeExplainer requer um modelo baseado em árvore.")

    preprocessor = model.pipeline.named_steps["preprocessor"]
    feature_names = get_output_feature_names(model)

    X_test_transformed = preprocessor.transform(split.X_test)
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()
    X_test_transformed = X_test_transformed[:max_samples]

    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)

    explainer = shap.TreeExplainer(classifier)
    explanation = explainer(X_test_df)

    # Para classificação binária alguns modelos retornam valores para as 2 classes;
    # mantenho a classe 1 ("benign") como referência.
    if len(explanation.shape) == 3:
        explanation = explanation[..., 1]

    return explanation, X_test_df


def plot_shap_summary(explanation, save: bool = True, name: str = "09_shap_summary.png") -> Path | None:
    fig = plt.figure(figsize=(9, 7))
    shap.summary_plot(explanation, show=False)
    fig = plt.gcf()
    if save:
        return _save(fig, name)
    plt.show()
    return None


def plot_shap_waterfall(explanation, sample_idx: int, save: bool = True) -> Path | None:
    fig = plt.figure(figsize=(9, 6))
    shap.plots.waterfall(explanation[sample_idx], show=False)
    fig = plt.gcf()
    if save:
        return _save(fig, f"10_shap_waterfall_sample_{sample_idx}.png")
    plt.show()
    return None
