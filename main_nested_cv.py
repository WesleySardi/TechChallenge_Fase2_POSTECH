"""Comparação robusta original vs. otimizado via VALIDAÇÃO CRUZADA ANINHADA.

O teste da Fase 1 tem só 114 pacientes (1 paciente = 0,9 ponto de acurácia), e a
escolha da melhor entre centenas de configurações infla o fitness em CV ("maldição do
vencedor"). Para separar ganho real de ruído:

    para cada fold EXTERNO (5 folds x 2 repetições = 10 partições dos 569 pacientes):
        GA (config. E1) roda SÓ no treino externo (fitness = CV interna 5x3)
        melhor config. e config. original da Fase 1 são treinadas no treino externo
        ambas são avaliadas no fold externo de teste, que o GA nunca viu

Cada paciente vira teste 2 vezes, e a comparação é PAREADA (mesmos folds para os dois).

    python main_nested_cv.py                  # ~30 min (Random Forest domina o tempo)
    python main_nested_cv.py --models knn decision_tree --repeats 1
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import RepeatedStratifiedKFold

from src import data_loader, ga_plots, modeling, optimization, preprocessing
from src.ga.engine import GeneticAlgorithm
from src.ga.fitness import CVFitness
from src.ga.search_spaces import BASELINE_PARAMS, SEARCH_SPACES
from src.monitoring.logging_config import setup_logging

logger = logging.getLogger("main_nested_cv")
METRICS = ["test_accuracy", "test_precision_malignant", "test_recall_malignant", "test_f1_malignant", "test_roc_auc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(SEARCH_SPACES), choices=list(SEARCH_SPACES))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, d in folds.groupby("model", sort=False):
        orig = d[d["version"] == "original"].sort_values(["repeat", "fold"])
        opt = d[d["version"] == "otimizado_ga"].sort_values(["repeat", "fold"])
        for metric in METRICS:
            diff = opt[metric].to_numpy() - orig[metric].to_numpy()
            nonzero = diff[np.abs(diff) > 1e-12]
            p = wilcoxon(nonzero).pvalue if len(nonzero) >= 5 else float("nan")
            rows.append({
                "model": model, "metric": metric,
                "original_mean": orig[metric].mean(), "original_std": orig[metric].std(),
                "otimizado_mean": opt[metric].mean(), "otimizado_std": opt[metric].std(),
                "diff_mean": diff.mean(), "diff_std": diff.std(),
                "wins": int((diff > 1e-12).sum()), "ties": int((np.abs(diff) <= 1e-12).sum()),
                "losses": int((diff < -1e-12).sum()), "wilcoxon_p": p,
            })
    return pd.DataFrame(rows)


def plot_paired(folds: pd.DataFrame, path) -> None:
    """Diferença pareada (otimizado − original) por fold externo, por métrica."""
    labels = {"test_accuracy": "acurácia", "test_precision_malignant": "precisão (mal.)",
              "test_recall_malignant": "recall (mal.)", "test_f1_malignant": "F1 (mal.)", "test_roc_auc": "ROC-AUC"}
    models = list(dict.fromkeys(folds["model"]))
    fig, axes = ga_plots._model_grid(models)
    for ax, model in zip(axes, models):
        d = folds[folds["model"] == model]
        orig = d[d["version"] == "original"].sort_values(["repeat", "fold"])
        opt = d[d["version"] == "otimizado_ga"].sort_values(["repeat", "fold"])
        for i, metric in enumerate(METRICS):
            diff = opt[metric].to_numpy() - orig[metric].to_numpy()
            jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(diff))
            ax.scatter(diff, i + jitter, color=ga_plots.SERIES[0], s=22, alpha=0.75, edgecolor="white", linewidth=0.8)
            ax.scatter(diff.mean(), i, color=ga_plots.SERIES[1], s=70, marker="D", zorder=3,
                       edgecolor="white", linewidth=1, label="média" if i == 0 else None)
        ax.axvline(0, color=ga_plots.NEUTRAL, linestyle="--", linewidth=1.2)
        ax.set_yticks(range(len(METRICS)), [labels[m] for m in METRICS], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("otimizado − original (por fold externo)")
        ax.set_title(model, loc="left", fontweight="bold")
    handles = [plt.Line2D([], [], marker="o", color=ga_plots.SERIES[0], linestyle="", label="fold externo"),
               plt.Line2D([], [], marker="D", color=ga_plots.SERIES[1], linestyle="", label="média")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Validação cruzada aninhada: ganho do GA por fold externo (à direita de 0 = GA melhor)", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    setup_logging("nested_cv")
    start = time.perf_counter()

    df, _ = preprocessing.clean_dataset(data_loader.load_dataset())
    X, y = df[preprocessing.get_feature_columns(df)], df["diagnosis"]
    outer = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=2025)
    config = next(e for e in optimization.EXPERIMENTS if e.name == "E1_referencia").config

    rows = []
    for k, (train_idx, test_idx) in enumerate(outer.split(X, y)):
        repeat, fold = divmod(k, args.folds)
        split = modeling.SplitData(X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx])
        for model in args.models:
            fitness = CVFitness(model, split.X_train, split.y_train)
            ga = GeneticAlgorithm(SEARCH_SPACES[model], fitness, replace(config, seed=42 + k),
                                  initial_params=[BASELINE_PARAMS[model]], run_name=f"nested/{model}/r{repeat}f{fold}",
                                  n_jobs=args.n_jobs)
            result = ga.run()
            baseline = modeling.build_model_pipeline(df, model).fit(split.X_train, split.y_train)
            optimized = optimization.fit_final_pipeline(model, result.best_params, split)
            for version, pipe, params in (("original", baseline, BASELINE_PARAMS[model]),
                                          ("otimizado_ga", optimized, result.best_params)):
                rows.append({"model": model, "repeat": repeat, "fold": fold, "version": version,
                             **optimization.test_metrics(model, pipe, split, n_bootstrap=0),
                             "inner_cv_fitness": result.best_evaluation.fitness if version == "otimizado_ga" else None,
                             "params": json.dumps(params, default=str)})
            o, g = rows[-2], rows[-1]
            logger.info(f"split {k + 1}/{args.folds * args.repeats} {model}: F1 {o['test_f1_malignant']:.3f} -> "
                        f"{g['test_f1_malignant']:.3f} | recall {o['test_recall_malignant']:.3f} -> "
                        f"{g['test_recall_malignant']:.3f} ({result.elapsed_seconds:.0f}s)")

    folds = pd.DataFrame(rows)
    summary = summarize(folds)
    optimization.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    folds.round(5).to_csv(optimization.REPORTS_DIR / "nested_cv_folds.csv", index=False)
    summary.round(5).to_csv(optimization.REPORTS_DIR / "nested_cv_summary.csv", index=False)
    plot_paired(folds, ga_plots.FIGURES_DIR / "ga_07_nested_cv_paired.png")

    pd.set_option("display.width", 220)
    print("\n=== Validação cruzada aninhada: original vs. otimizado (média nos folds externos) ===")
    print(summary.round(4).to_string(index=False))
    print(f"\nConcluído em {(time.perf_counter() - start) / 60:.1f} min.")


if __name__ == "__main__":
    main()
