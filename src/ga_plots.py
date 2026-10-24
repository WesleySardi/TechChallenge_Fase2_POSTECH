"""Visualizações dos experimentos de algoritmo genético."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

FIGURES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"

# Paleta categórica validada (ordem fixa — a cor segue o experimento, nunca o ranking).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
NEUTRAL = "#8a8985"
TEXT_SECONDARY = "#52514e"

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c9c8c3", "axes.labelcolor": TEXT_SECONDARY,
    "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
    "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.8,
    "lines.linewidth": 2, "legend.frameon": False, "font.size": 10,
})


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def _model_grid(models: list[str]):
    n = len(models)
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 4.2 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, list(axes.flat)


def plot_convergence(history: pd.DataFrame, baseline: dict[str, float], name: str = "ga_01_convergence.png") -> Path:
    """Melhor fitness até a geração g (média entre seeds, faixa = min–max), por experimento."""
    models = list(dict.fromkeys(history["model"]))
    experiments = list(dict.fromkeys(history["experiment"]))
    fig, axes = _model_grid(models)
    for ax, model in zip(axes, models):
        for color, exp in zip(SERIES, experiments):
            data = history[(history["model"] == model) & (history["experiment"] == exp)]
            agg = data.groupby("generation")["best_so_far"].agg(["mean", "min", "max"])
            ax.plot(agg.index, agg["mean"], color=color, label=exp)
            if data["seed"].nunique() > 1:
                ax.fill_between(agg.index, agg["min"], agg["max"], color=color, alpha=0.12, linewidth=0)
        ax.axhline(baseline[model], color=NEUTRAL, linestyle="--", linewidth=1.5, label="modelo original (Fase 1)")
        ax.set_title(model, loc="left", fontweight="bold")
        ax.set_xlabel("geração")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel("fitness (CV)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Convergência do algoritmo genético — melhor fitness até cada geração", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, name)


def plot_population_fitness(history: pd.DataFrame, experiment: str, name: str = "ga_02_population.png") -> Path:
    """Fitness médio vs. melhor da população e diversidade genética, num experimento."""
    data = history[history["experiment"] == experiment]
    models = list(dict.fromkeys(data["model"]))
    fig, axes = _model_grid(models)
    for ax, model in zip(axes, models):
        d = data[data["model"] == model].groupby("generation")[["best_fitness", "mean_fitness", "diversity"]].mean()
        ax.plot(d.index, d["best_fitness"], color=SERIES[0], label="melhor da geração")
        ax.plot(d.index, d["mean_fitness"], color=SERIES[1], label="média da população")
        ax.set_title(model, loc="left", fontweight="bold")
        ax.set_xlabel("geração")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel("fitness (CV)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"População ao longo das gerações — {experiment}", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, name)


def plot_diversity(history: pd.DataFrame, name: str = "ga_06_diversity.png") -> Path:
    """Diversidade genética (desvio padrão médio dos genes) por experimento.

    Queda rápida = população convergiu (todos parecidos); diversidade que se mantém
    alta = mais exploração (ex.: mutação alta).
    """
    models = list(dict.fromkeys(history["model"]))
    experiments = list(dict.fromkeys(history["experiment"]))
    fig, axes = _model_grid(models)
    for ax, model in zip(axes, models):
        for color, exp in zip(SERIES, experiments):
            d = history[(history["model"] == model) & (history["experiment"] == exp)].groupby("generation")["diversity"].mean()
            ax.plot(d.index, d.values, color=color, label=exp)
        ax.set_title(model, loc="left", fontweight="bold")
        ax.set_xlabel("geração")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel("diversidade (dp médio dos genes)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Diversidade genética da população ao longo das gerações", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, name)


def plot_ga_vs_random(history: pd.DataFrame, random_curves: dict[str, list[float]], experiment: str,
                      name: str = "ga_03_ga_vs_random_search.png") -> Path:
    """Melhor fitness em função do nº de avaliações: GA vs. busca aleatória (mesmo orçamento)."""
    data = history[history["experiment"] == experiment]
    models = list(dict.fromkeys(data["model"]))
    fig, axes = _model_grid(models)
    for ax, model in zip(axes, models):
        for seed, d in data[data["model"] == model].groupby("seed"):
            ax.plot(d["unique_evaluations"], d["best_so_far"], color=SERIES[0], alpha=0.85,
                    label="algoritmo genético" if seed == data["seed"].min() else None)
        curve = random_curves[model]
        ax.plot(np.arange(1, len(curve) + 1), curve, color=SERIES[1], label="busca aleatória")
        ax.set_title(model, loc="left", fontweight="bold")
        ax.set_xlabel("nº de configurações avaliadas")
        ax.set_ylabel("melhor fitness (CV)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"GA ({experiment}) vs. busca aleatória com o mesmo orçamento", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, name)


def plot_original_vs_optimized(comparison: pd.DataFrame, name: str = "ga_04_original_vs_optimized.png") -> Path:
    """Métricas no TESTE: original (Fase 1) vs. otimizado pelo GA, com IC 95% por bootstrap.

    Gráfico de pontos (e não barras) porque as diferenças são pequenas e o eixo precisa
    ser ampliado — barras com eixo truncado exageram visualmente as diferenças.
    """
    metrics = ["test_accuracy", "test_precision_malignant", "test_recall_malignant", "test_f1_malignant", "test_roc_auc"]
    labels = ["acurácia", "precisão (maligno)", "recall (maligno)", "F1 (maligno)", "ROC-AUC"]
    models = list(dict.fromkeys(comparison["model"]))
    fig, axes = _model_grid(models)
    y = np.arange(len(metrics))
    versions = (("original", NEUTRAL, "original (Fase 1)", 0.12), ("otimizado_ga", SERIES[0], "otimizado (GA)", -0.12))
    for ax, model in zip(axes, models):
        d = comparison[comparison["model"] == model].set_index("version")
        for version, color, label, offset in versions:
            values = d.loc[version, metrics].astype(float).values
            has_ci = all(f"{m}_ci_low" in d.columns for m in metrics)
            if has_ci:
                low = d.loc[version, [f"{m}_ci_low" for m in metrics]].astype(float).values
                high = d.loc[version, [f"{m}_ci_high" for m in metrics]].astype(float).values
                ax.hlines(y + offset, low, high, color=color, linewidth=1.5, alpha=0.6)
            ax.scatter(values, y + offset, color=color, s=40, zorder=3, label=label,
                       edgecolor="white", linewidth=1)
            for v, yy in zip(values, y + offset):
                ax.annotate(f"{v:.3f}", (v, yy), textcoords="offset points", xytext=(0, 5),
                            ha="center", fontsize=7, color=TEXT_SECONDARY)
        ax.set_yticks(y, labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(right=1.005)
        ax.set_title(model, loc="left", fontweight="bold")
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Conjunto de teste: original vs. otimizado pelo GA (linha = IC 95% bootstrap)", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, name)


def plot_experiment_summary(summary: pd.DataFrame, name: str = "ga_05_experiments_summary.png") -> Path:
    """Melhor fitness médio (± dp entre seeds) por experimento, em cada modelo."""
    models = list(dict.fromkeys(summary["model"]))
    experiments = list(dict.fromkeys(summary["experiment"]))
    fig, axes = _model_grid(models)
    for ax, model in zip(axes, models):
        d = summary[summary["model"] == model].set_index("experiment").loc[experiments]
        y = np.arange(len(experiments))
        ax.barh(y, d["best_fitness_mean"], xerr=d["best_fitness_std"].fillna(0), color=SERIES[: len(experiments)],
                height=0.6, error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1})
        ax.axvline(d["baseline_cv_fitness"].iloc[0], color=NEUTRAL, linestyle="--", linewidth=1.5)
        ax.axvline(d["random_search_fitness"].iloc[0], color=SERIES[7], linestyle=":", linewidth=1.5)
        ax.set_yticks(y, experiments, fontsize=8)
        ax.invert_yaxis()
        lo = min(d["best_fitness_mean"].min(), d["baseline_cv_fitness"].iloc[0], d["random_search_fitness"].iloc[0])
        ax.set_xlim(lo - 0.01, d["best_fitness_mean"].max() + d["best_fitness_std"].fillna(0).max() + 0.005)
        ax.set_title(model, loc="left", fontweight="bold")
        ax.set_xlabel("melhor fitness (CV)")
    fig.legend([plt.Line2D([], [], color=NEUTRAL, linestyle="--"), plt.Line2D([], [], color=SERIES[7], linestyle=":")],
               ["modelo original (Fase 1)", "busca aleatória"], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Resultado por experimento (média ± desvio padrão entre seeds)", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, name)


def load_random_curves(path: Path) -> dict[str, list[float]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
