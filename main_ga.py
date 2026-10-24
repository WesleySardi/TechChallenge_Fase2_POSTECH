"""Tech Challenge Fase 2 — otimização de hiperparâmetros via algoritmo genético.

Roda os experimentos de GA (src/optimization.py::EXPERIMENTS) nos 4 modelos da Fase 1,
compara com a busca aleatória e com os modelos originais, e salva relatórios,
figuras e os modelos otimizados (usados pela API e pela camada de LLM).

Uso:
    python main_ga.py                       # completo (4 experimentos x 3 seeds)
    python main_ga.py --quick               # rápido p/ testar (1 seed, 4 gerações)
    python main_ga.py --models knn --seeds 42
"""

import argparse
import logging
import sys
import time

import pandas as pd

from src import data_loader, ga_plots, modeling, optimization, preprocessing
from src.ga.search_spaces import SEARCH_SPACES
from src.monitoring.logging_config import setup_logging

logger = logging.getLogger("main_ga")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(SEARCH_SPACES), choices=list(SEARCH_SPACES))
    parser.add_argument("--experiments", nargs="+", default=[e.name for e in optimization.EXPERIMENTS],
                        choices=[e.name for e in optimization.EXPERIMENTS])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 2024])
    parser.add_argument("--generations", type=int, default=None, help="sobrescreve o nº de gerações")
    parser.add_argument("--n-jobs", type=int, default=-1, help="processos p/ avaliar indivíduos em paralelo")
    parser.add_argument("--quick", action="store_true", help="1 seed e 4 gerações (teste rápido)")
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.quick:
        args.seeds, args.generations = args.seeds[:1], args.generations or 4
    log_path = setup_logging("ga")
    experiments = [e for e in optimization.EXPERIMENTS if e.name in args.experiments]
    start = time.perf_counter()

    logger.info("1/5 - Carregando e dividindo dataset (mesmo split da Fase 1)...")
    df, _ = preprocessing.clean_dataset(data_loader.load_dataset())
    split = modeling.train_test_split_dataset(df)
    logger.info(f"   treino={len(split.X_train)} teste={len(split.X_test)}")

    logger.info(f"2/5 - Rodando GA: modelos={args.models} experimentos={[e.name for e in experiments]} seeds={args.seeds}")
    results = [
        optimization.optimize_model(name, split, experiments, args.seeds, generations=args.generations, n_jobs=args.n_jobs)
        for name in args.models
    ]

    logger.info("3/5 - Comparando original vs. otimizado no conjunto de TESTE...")
    comparison = optimization.compare_on_test(results, split, df)

    logger.info("4/5 - Salvando relatórios...")
    paths = optimization.save_reports(results, comparison)

    logger.info("5/5 - Gerando figuras...")
    history = pd.read_csv(paths["history"])
    summary = pd.read_csv(paths["summary"])
    baseline = {r.model_name: r.baseline_cv.fitness for r in results}
    ref = experiments[0].name
    ga_plots.plot_convergence(history, baseline)
    ga_plots.plot_population_fitness(history, ref)
    ga_plots.plot_ga_vs_random(history, ga_plots.load_random_curves(paths["random_search"]), ref)
    ga_plots.plot_original_vs_optimized(comparison)
    ga_plots.plot_experiment_summary(summary)
    ga_plots.plot_diversity(history)

    pd.set_option("display.width", 200)
    print("\n=== Resumo por experimento (fitness em CV no treino) ===")
    print(summary[["model", "experiment", "best_fitness_mean", "best_fitness_std", "baseline_cv_fitness",
                   "random_search_fitness", "evaluations_mean", "elapsed_seconds_mean"]].round(4).to_string(index=False))
    print("\n=== Conjunto de teste: original vs. otimizado ===")
    print(comparison.round(4).to_string(index=False))
    print(f"\nConcluído em {time.perf_counter() - start:.0f}s. Relatórios em outputs/reports/, "
          f"figuras em outputs/figures/ga_*, log estruturado em {log_path}.")


if __name__ == "__main__":
    main()
