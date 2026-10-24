"""EXTRA — algoritmo genético nos hiperparâmetros da CNN (ResNet18, transfer learning).

Pré-requisitos: PyTorch com CUDA (requirements-cnn.txt) e o dataset CBIS-DDSM
(usa ../1_Fase/data/archive ou a variável CBIS_DDSM_DIR).

    python main_ga_cnn.py                         # pop. 12, 10 gerações, 6 épocas x 2 seeds por indivíduo
    python main_ga_cnn.py --population 6 --generations 2 --fitness-epochs 2   # teste rápido
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src import ga_plots
from src.cnn_ga import (
    CNN_BASELINE_PARAMS,
    CNN_SEARCH_SPACE,
    CNNFitness,
    image_metrics,
    load_image_splits,
    predict_proba,
    train_cnn,
)
from src.ga.engine import GAConfig, GeneticAlgorithm
from src.monitoring.logging_config import setup_logging
from src.optimization import bootstrap_ci

logger = logging.getLogger("main_ga_cnn")
ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "outputs" / "reports"
MODELS_DIR = ROOT / "outputs" / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--fitness-epochs", type=int, default=6, help="épocas por avaliação de indivíduo")
    parser.add_argument("--fitness-repeats", type=int, default=2, help="treinos (seeds) por indivíduo; fitness = média")
    parser.add_argument("--final-epochs", type=int, default=20, help="épocas do re-treino final (com early stopping)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evaluate_final(name: str, params: dict, data, device, epochs: int) -> tuple[dict, list[dict]]:
    """Re-treino completo (como na Fase 1: early stopping paciência 6) e avaliação no teste."""
    model, history = train_cnn(params, data, device, epochs=epochs, patience=6)
    probs = predict_proba(model, data["test"], device)
    y = data["test"].y.numpy().astype(int)
    metrics = image_metrics(y, probs, prefix="test_")
    # bootstrap_ci usa a convenção do dataset tabular (0 = maligno): inverte os rótulos.
    ci = bootstrap_ci(1 - y, 1 - (probs >= 0.5).astype(int), 1 - probs, n_bootstrap=1000)
    torch.save(model.state_dict(), MODELS_DIR / f"cnn_transfer_{name}.pt")
    return {"version": name, **metrics, **ci, "epochs_trained": len(history)}, history


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    setup_logging("ga_cnn")
    if not torch.cuda.is_available():
        sys.exit("CUDA indisponível: este experimento precisa de GPU (instale requirements-cnn.txt).")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    start = time.perf_counter()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"1/4 - Carregando imagens (cache em outputs/cache) | GPU: {torch.cuda.get_device_name(0)}")
    data = load_image_splits()
    logger.info("   " + " | ".join(f"{k}={len(v.y)} ({int(v.y.sum())} malignos)" for k, v in data.items()))

    logger.info(f"2/4 - GA: população {args.population}, {args.generations} gerações, "
                f"{args.fitness_epochs} épocas x {args.fitness_repeats} seeds por indivíduo")
    config = GAConfig(population_size=args.population, generations=args.generations, mutation_rate=0.15,
                      elitism=2, tournament_size=3, seed=args.seed)
    fitness = CNNFitness(data, device, epochs=args.fitness_epochs, repeats=args.fitness_repeats)
    ga = GeneticAlgorithm(CNN_SEARCH_SPACE, fitness, config, initial_params=[CNN_BASELINE_PARAMS], run_name="cnn")
    result = ga.run()
    baseline_eval = ga.evaluate(CNN_SEARCH_SPACE.encode(CNN_BASELINE_PARAMS))  # já em cache (semente)
    logger.info(f"   melhor fitness (val) = {result.best_evaluation.fitness:.4f} | "
                f"config. da Fase 1 = {baseline_eval.fitness:.4f} | {result.n_evaluations} avaliações")

    logger.info(f"3/4 - Re-treino completo ({args.final_epochs} épocas) da config. da Fase 1 e da otimizada -> teste")
    rows, histories = [], {}
    for name, params in (("original", CNN_BASELINE_PARAMS), ("otimizado_ga", result.best_params)):
        row, hist = evaluate_final(name, params, data, device, args.final_epochs)
        rows.append({"model": "cnn_resnet18_transfer", **row})
        histories[name] = hist
    comparison = pd.DataFrame(rows)

    logger.info("4/4 - Salvando relatórios e figuras...")
    pd.DataFrame(result.history_records()).round(5).to_csv(REPORTS_DIR / "cnn_ga_history.csv", index=False)
    comparison.round(4).to_csv(REPORTS_DIR / "cnn_comparison_original_vs_optimized.csv", index=False)
    (REPORTS_DIR / "cnn_ga_best_params.json").write_text(json.dumps({
        "best_params": result.best_params, "best_val_fitness": result.best_evaluation.fitness,
        "best_val_metrics": result.best_evaluation.metrics, "baseline_params": CNN_BASELINE_PARAMS,
        "baseline_val_fitness": baseline_eval.fitness, "n_evaluations": result.n_evaluations,
        "elapsed_seconds": result.elapsed_seconds, "config": asdict(config),
    }, indent=2, default=str), encoding="utf-8")
    all_evals = [{**{"fitness": e.fitness}, **e.metrics} for e in ga._cache.values()]
    pd.DataFrame(all_evals).round(5).to_csv(REPORTS_DIR / "cnn_ga_evaluations.csv", index=False)

    ga_plots.plot_original_vs_optimized(comparison, name="cnn_ga_01_original_vs_optimized.png")
    hist = pd.DataFrame(result.history_records())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist["generation"], hist["best_so_far"], color=ga_plots.SERIES[0], label="melhor até a geração")
    ax.plot(hist["generation"], hist["mean_fitness"], color=ga_plots.SERIES[1], label="média da população")
    ax.axhline(baseline_eval.fitness, color=ga_plots.NEUTRAL, linestyle="--", label="config. da Fase 1")
    ax.set_xlabel("geração")
    ax.set_ylabel("fitness (validação)")
    ax.legend()
    ax.set_title("GA na CNN (ResNet18) — convergência", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(ga_plots.FIGURES_DIR / "cnn_ga_02_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    pd.set_option("display.width", 200)
    print("\n=== CNN: melhores hiperparâmetros ===")
    print(json.dumps(result.best_params, indent=2, default=str))
    print("\n=== CNN: teste (original vs. otimizado) ===")
    print(comparison[[c for c in comparison.columns if "_ci_" not in c]].round(4).to_string(index=False))
    print(f"\nConcluído em {(time.perf_counter() - start) / 60:.1f} min.")


if __name__ == "__main__":
    main()
