"""Orquestra os experimentos de otimização de hiperparâmetros via algoritmo genético.

Para cada modelo da Fase 1:
1. mede o modelo ORIGINAL com o mesmo protocolo de validação cruzada do fitness;
2. roda o GA com cada configuração de experimento (e cada seed);
3. roda uma busca aleatória com o mesmo orçamento de avaliações (linha de base);
4. escolhe a melhor configuração pelo fitness em CV (nunca pelo teste);
5. treina no treino completo e compara original vs. otimizado no TESTE.
"""

import json
import logging
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src import evaluation, modeling
from src.ga.engine import Evaluation, GAConfig, GAResult, GeneticAlgorithm, random_search
from src.ga.fitness import CVFitness
from src.ga.search_spaces import BASELINE_PARAMS, SEARCH_SPACES

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "outputs" / "reports"
MODELS_DIR = ROOT / "outputs" / "models"


@dataclass
class Experiment:
    name: str
    description: str
    config: GAConfig


# Cada experimento varia UM fator em relação ao E1 (referência), para que a
# diferença de resultado seja atribuível àquele fator.
EXPERIMENTS: list[Experiment] = [
    Experiment(
        "E1_referencia",
        "Configuração de referência: pop. 20, mutação 0.10, torneio (k=3), crossover uniforme",
        GAConfig(population_size=20, generations=15, mutation_rate=0.10, selection="tournament", crossover="uniform"),
    ),
    Experiment(
        "E2_populacao_grande",
        "População maior (40): mais diversidade/exploração, custo ~2x por geração",
        GAConfig(population_size=40, generations=15, mutation_rate=0.10, selection="tournament", crossover="uniform"),
    ),
    Experiment(
        "E3_mutacao_alta",
        "Mutação alta (0.30, sigma 0.25): mais exploração, menos refinamento",
        GAConfig(population_size=20, generations=15, mutation_rate=0.30, mutation_sigma=0.25,
                 selection="tournament", crossover="uniform"),
    ),
    Experiment(
        "E4_roleta_blend",
        "Operadores alternativos: seleção por roleta + crossover BLX-alpha",
        GAConfig(population_size=20, generations=15, mutation_rate=0.10, selection="roulette", crossover="blend"),
    ),
]


@dataclass
class ModelOptimizationResult:
    model_name: str
    baseline_cv: Evaluation
    runs: list[dict[str, Any]] = field(default_factory=list)       # 1 por (experimento, seed)
    histories: list[pd.DataFrame] = field(default_factory=list)
    random_search: dict[str, Any] | None = None
    best_run: dict[str, Any] | None = None


def _run_record(model_name: str, exp: Experiment, seed: int, result: GAResult) -> dict[str, Any]:
    return {
        "model": model_name,
        "experiment": exp.name,
        "seed": seed,
        "best_fitness": result.best_evaluation.fitness,
        **{k: v for k, v in result.best_evaluation.metrics.items() if k.endswith("_mean")},
        "best_params": result.best_params,
        "n_evaluations": result.n_evaluations,
        "generations_run": len(result.history) - 1,
        "elapsed_seconds": result.elapsed_seconds,
        "config": asdict(result.config),
    }


def optimize_model(
    model_name: str,
    split: modeling.SplitData,
    experiments: list[Experiment],
    seeds: list[int],
    generations: int | None = None,
    n_jobs: int = -1,
) -> ModelOptimizationResult:
    fitness = CVFitness(model_name, split.X_train, split.y_train)
    space = SEARCH_SPACES[model_name]

    baseline_pipeline = modeling.build_model_pipeline(
        pd.concat([split.X_train, split.y_train], axis=1), model_name
    )
    baseline_cv = fitness.evaluate_pipeline(baseline_pipeline)
    logger.info(
        f"{model_name}: fitness do modelo original (CV) = {baseline_cv.fitness:.4f}",
        extra={"event": "baseline_cv", "model": model_name, "fitness": baseline_cv.fitness, **baseline_cv.metrics},
    )

    result = ModelOptimizationResult(model_name=model_name, baseline_cv=baseline_cv)
    for exp in experiments:
        for seed in seeds:
            config = replace(exp.config, seed=seed)
            if generations is not None:
                config = replace(config, generations=generations)
            run_name = f"{model_name}/{exp.name}/seed{seed}"
            ga = GeneticAlgorithm(space, fitness, config, initial_params=[BASELINE_PARAMS[model_name]],
                                  run_name=run_name, n_jobs=n_jobs)
            ga_result = ga.run()
            record = _run_record(model_name, exp, seed, ga_result)
            result.runs.append(record)

            history = pd.DataFrame(ga_result.history_records())
            history.insert(0, "seed", seed)
            history.insert(0, "experiment", exp.name)
            history.insert(0, "model", model_name)
            result.histories.append(history)
            logger.info(
                f"{run_name}: melhor fitness = {ga_result.best_evaluation.fitness:.4f} "
                f"({ga_result.n_evaluations} avaliações, {ga_result.elapsed_seconds:.1f}s)",
                extra={"event": "ga_run_done", **{k: v for k, v in record.items() if k != "config"}},
            )

    # Busca aleatória com o orçamento médio de avaliações do experimento de referência.
    ref_runs = [r for r in result.runs if r["experiment"] == experiments[0].name]
    budget = int(np.mean([r["n_evaluations"] for r in ref_runs]))
    rs_params, rs_eval, rs_curve = random_search(space, fitness, n_evaluations=budget, seed=seeds[0], n_jobs=n_jobs)
    result.random_search = {
        "model": model_name, "best_fitness": rs_eval.fitness, "best_params": rs_params,
        "n_evaluations": len(rs_curve), "curve": rs_curve,
    }
    logger.info(
        f"{model_name}: busca aleatória ({len(rs_curve)} avaliações) = {rs_eval.fitness:.4f}",
        extra={"event": "random_search_done", "model": model_name, "best_fitness": rs_eval.fitness},
    )

    # Seleção final pelo fitness em CV (o teste continua intocado).
    result.best_run = max(result.runs, key=lambda r: r["best_fitness"])
    return result


def fit_final_pipeline(model_name: str, params: dict, split: modeling.SplitData) -> Pipeline:
    fitness = CVFitness(model_name, split.X_train, split.y_train)
    pipeline = fitness.build_pipeline(params)
    pipeline.fit(split.X_train, split.y_train)
    return pipeline


def test_metrics(name: str, pipeline: Pipeline, split: modeling.SplitData, n_bootstrap: int = 1000) -> dict[str, float]:
    """Métricas no teste + IC 95% por bootstrap (o teste tem só 114 amostras: 1 paciente
    = 0,9 ponto de acurácia, então diferenças pequenas podem ser ruído)."""
    trained = modeling.TrainedModel(name=name, pipeline=pipeline, cv_accuracy_mean=np.nan, cv_accuracy_std=np.nan)
    metrics = {k: v for k, v in evaluation.evaluate_model(trained, split).items() if k.startswith("test_")}
    if n_bootstrap:
        y_true = split.y_test.to_numpy()
        y_pred = pipeline.predict(split.X_test)
        y_proba = pipeline.predict_proba(split.X_test)[:, 1]
        metrics.update(bootstrap_ci(y_true, y_pred, y_proba, n_bootstrap))
    return metrics


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, n_bootstrap: int = 1000,
                 seed: int = 42) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    fns = {
        "test_accuracy": lambda t, p, s: accuracy_score(t, p),
        "test_precision_malignant": lambda t, p, s: precision_score(t, p, pos_label=0, zero_division=0),
        "test_recall_malignant": lambda t, p, s: recall_score(t, p, pos_label=0, zero_division=0),
        "test_f1_malignant": lambda t, p, s: f1_score(t, p, pos_label=0, zero_division=0),
        "test_roc_auc": lambda t, p, s: roc_auc_score(t, s),
    }
    samples = {k: [] for k in fns}
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        for k, fn in fns.items():
            samples[k].append(fn(y_true[idx], y_pred[idx], y_proba[idx]))
    out = {}
    for k, values in samples.items():
        out[f"{k}_ci_low"], out[f"{k}_ci_high"] = np.percentile(values, [2.5, 97.5])
    return out


def compare_on_test(results: list[ModelOptimizationResult], split: modeling.SplitData, df: pd.DataFrame,
                    save_models: bool = True) -> pd.DataFrame:
    """Treina original e otimizado no treino completo e mede ambos no teste."""
    rows = []
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for res in results:
        baseline = modeling.build_model_pipeline(df, res.model_name).fit(split.X_train, split.y_train)
        optimized = fit_final_pipeline(res.model_name, res.best_run["best_params"], split)

        for version, pipeline, cv_fitness in (
            ("original", baseline, res.baseline_cv.fitness),
            ("otimizado_ga", optimized, res.best_run["best_fitness"]),
        ):
            rows.append({"model": res.model_name, "version": version, "cv_fitness": cv_fitness,
                         **test_metrics(res.model_name, pipeline, split)})

        if save_models:
            joblib.dump(optimized, MODELS_DIR / f"{res.model_name}_optimized.joblib")
            joblib.dump(baseline, MODELS_DIR / f"{res.model_name}_original.joblib")
    return pd.DataFrame(rows)


def save_reports(results: list[ModelOptimizationResult], comparison: pd.DataFrame) -> dict[str, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}

    runs = pd.DataFrame([{k: v for k, v in r.items() if k not in ("config", "best_params")}
                         | {"best_params": json.dumps(r["best_params"], default=str)}
                         for res in results for r in res.runs])
    paths["runs"] = REPORTS_DIR / "ga_runs.csv"
    runs.round(5).to_csv(paths["runs"], index=False)

    summary = (runs.groupby(["model", "experiment"])
               .agg(best_fitness_mean=("best_fitness", "mean"), best_fitness_std=("best_fitness", "std"),
                    best_fitness_max=("best_fitness", "max"), evaluations_mean=("n_evaluations", "mean"),
                    elapsed_seconds_mean=("elapsed_seconds", "mean"), n_seeds=("seed", "count"))
               .reset_index())
    baseline = pd.DataFrame([{"model": r.model_name, "baseline_cv_fitness": r.baseline_cv.fitness} for r in results])
    rs = pd.DataFrame([{"model": r.model_name, "random_search_fitness": r.random_search["best_fitness"],
                        "random_search_evaluations": r.random_search["n_evaluations"]} for r in results])
    summary = summary.merge(baseline, on="model").merge(rs, on="model")
    paths["summary"] = REPORTS_DIR / "ga_experiments_summary.csv"
    summary.round(5).to_csv(paths["summary"], index=False)

    paths["history"] = REPORTS_DIR / "ga_history.csv"
    pd.concat([h for r in results for h in r.histories]).round(5).to_csv(paths["history"], index=False)

    paths["random_search"] = REPORTS_DIR / "random_search_curves.json"
    paths["random_search"].write_text(json.dumps(
        {r.model_name: r.random_search["curve"] for r in results}), encoding="utf-8")

    paths["best_params"] = REPORTS_DIR / "ga_best_params.json"
    paths["best_params"].write_text(json.dumps({
        r.model_name: {
            "experiment": r.best_run["experiment"], "seed": r.best_run["seed"],
            "cv_fitness": r.best_run["best_fitness"], "baseline_cv_fitness": r.baseline_cv.fitness,
            "params": r.best_run["best_params"], "baseline_params": BASELINE_PARAMS[r.model_name],
        } for r in results
    }, indent=2, default=str), encoding="utf-8")

    paths["comparison"] = REPORTS_DIR / "comparison_original_vs_optimized.csv"
    comparison.round(4).to_csv(paths["comparison"], index=False)
    return paths
