"""Laço principal do algoritmo genético (geracional, com elitismo)."""

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np
from joblib import Parallel, delayed

from src.ga.genes import SearchSpace, params_key
from src.ga.operators import (
    CROSSOVER_OPERATORS,
    SELECTION_OPERATORS,
    gaussian_mutation,
    random_reset_mutation,
)

logger = logging.getLogger(__name__)


@dataclass
class GAConfig:
    population_size: int = 20
    generations: int = 15
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1       # prob. de cada gene sofrer mutação gaussiana
    mutation_sigma: float = 0.15     # desvio do ruído gaussiano (genes em [0, 1))
    reset_rate: float = 0.02         # prob. de cada gene ser re-sorteado (salto grande)
    selection: str = "tournament"
    tournament_size: int = 3
    crossover: str = "uniform"
    elitism: int = 2                 # melhores indivíduos copiados sem alteração
    early_stopping_generations: int | None = None  # para se o melhor não melhora por N gerações
    seed: int = 42

    def __post_init__(self):
        if self.selection not in SELECTION_OPERATORS:
            raise ValueError(f"selection deve ser um de {list(SELECTION_OPERATORS)}")
        if self.crossover not in CROSSOVER_OPERATORS:
            raise ValueError(f"crossover deve ser um de {list(CROSSOVER_OPERATORS)}")
        if not 0 <= self.elitism < self.population_size:
            raise ValueError("elitism deve estar em [0, population_size)")


@dataclass
class Evaluation:
    """Resultado de avaliar um conjunto de hiperparâmetros."""
    fitness: float
    metrics: dict[str, float] = field(default_factory=dict)


FitnessFunction = Callable[[dict[str, Any]], Evaluation]


def evaluate_many(fitness_fn: FitnessFunction, params_list: list[dict[str, Any]], n_jobs: int = 1) -> list[Evaluation]:
    if n_jobs == 1 or len(params_list) <= 1:
        return [fitness_fn(p) for p in params_list]
    return Parallel(n_jobs=n_jobs)(delayed(fitness_fn)(p) for p in params_list)


@dataclass
class GenerationStats:
    generation: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    worst_fitness: float
    best_so_far: float
    unique_evaluations: int
    elapsed_seconds: float
    diversity: float  # desvio padrão médio dos genes — mede convergência da população


@dataclass
class GAResult:
    best_params: dict[str, Any]
    best_evaluation: Evaluation
    history: list[GenerationStats]
    n_evaluations: int
    elapsed_seconds: float
    config: GAConfig

    def history_records(self) -> list[dict]:
        return [asdict(h) for h in self.history]


class GeneticAlgorithm:
    def __init__(
        self,
        space: SearchSpace,
        fitness_fn: FitnessFunction,
        config: GAConfig | None = None,
        initial_params: list[dict[str, Any]] | None = None,
        run_name: str = "ga",
        n_jobs: int = 1,
    ):
        """
        initial_params: indivíduos "semente" inseridos na população inicial (ex.: os
        hiperparâmetros originais da Fase 1). Garante que o GA parte de, no mínimo, a
        solução atual — o resto da população é aleatório.
        n_jobs: processos para avaliar os indivíduos de uma geração em paralelo
        (-1 = todos os núcleos). Não altera o resultado, só o tempo.
        """
        self.space = space
        self.fitness_fn = fitness_fn
        self.config = config or GAConfig()
        self.initial_params = initial_params or []
        self.run_name = run_name
        self.n_jobs = n_jobs
        self.rng = np.random.default_rng(self.config.seed)
        self._cache: dict[tuple, Evaluation] = {}

    # ------------------------------------------------------------ avaliação

    def evaluate(self, genome: np.ndarray) -> Evaluation:
        params = self.space.decode(genome)
        key = params_key(params)
        if key not in self._cache:
            self._cache[key] = self.fitness_fn(params)
        return self._cache[key]

    def evaluate_population(self, population: list[np.ndarray]) -> list[Evaluation]:
        """Avalia em paralelo só os indivíduos ainda não vistos (cache por params)."""
        decoded = [self.space.decode(g) for g in population]
        pending = {}
        for params in decoded:
            key = params_key(params)
            if key not in self._cache:
                pending.setdefault(key, params)
        self._cache.update(zip(pending, evaluate_many(self.fitness_fn, list(pending.values()), self.n_jobs)))
        return [self._cache[params_key(p)] for p in decoded]

    @property
    def n_evaluations(self) -> int:
        return len(self._cache)

    # ------------------------------------------------------------ população

    def _initial_population(self) -> list[np.ndarray]:
        cfg = self.config
        seeds = [self.space.encode(p) for p in self.initial_params[: cfg.population_size]]
        randoms = [self.space.random_genome(self.rng) for _ in range(cfg.population_size - len(seeds))]
        return seeds + randoms

    def _next_generation(self, population: list[np.ndarray], fitnesses: np.ndarray) -> list[np.ndarray]:
        cfg = self.config
        select = SELECTION_OPERATORS[cfg.selection]
        crossover = CROSSOVER_OPERATORS[cfg.crossover]

        order = np.argsort(fitnesses)[::-1]
        next_pop = [population[i].copy() for i in order[: cfg.elitism]]

        while len(next_pop) < cfg.population_size:
            p1 = population[select(fitnesses, self.rng, k=cfg.tournament_size)]
            p2 = population[select(fitnesses, self.rng, k=cfg.tournament_size)]

            if self.rng.random() < cfg.crossover_rate:
                c1, c2 = crossover(p1, p2, self.rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            for child in (c1, c2):
                child = gaussian_mutation(child, self.rng, cfg.mutation_rate, cfg.mutation_sigma)
                child = random_reset_mutation(child, self.rng, cfg.reset_rate)
                if len(next_pop) < cfg.population_size:
                    next_pop.append(child)

        return next_pop

    # ------------------------------------------------------------ execução

    def run(self) -> GAResult:
        cfg = self.config
        start = time.perf_counter()
        population = self._initial_population()

        history: list[GenerationStats] = []
        best_params: dict[str, Any] | None = None
        best_eval: Evaluation | None = None
        stall = 0

        for gen in range(cfg.generations + 1):  # geração 0 = população inicial
            evaluations = self.evaluate_population(population)
            fitnesses = np.array([e.fitness for e in evaluations])

            gen_best_idx = int(np.argmax(fitnesses))
            if best_eval is None or fitnesses[gen_best_idx] > best_eval.fitness:
                best_eval = evaluations[gen_best_idx]
                best_params = self.space.decode(population[gen_best_idx])
                stall = 0
            else:
                stall += 1

            stats = GenerationStats(
                generation=gen,
                best_fitness=float(fitnesses.max()),
                mean_fitness=float(fitnesses.mean()),
                std_fitness=float(fitnesses.std()),
                worst_fitness=float(fitnesses.min()),
                best_so_far=float(best_eval.fitness),
                unique_evaluations=self.n_evaluations,
                elapsed_seconds=time.perf_counter() - start,
                diversity=float(np.stack(population).std(axis=0).mean()),
            )
            history.append(stats)
            logger.info(
                f"{self.run_name} | geração {gen:02d} | melhor={stats.best_fitness:.4f} "
                f"média={stats.mean_fitness:.4f} melhor_global={stats.best_so_far:.4f} "
                f"avaliações={stats.unique_evaluations}",
                extra={"event": "ga_generation", "run": self.run_name, **asdict(stats)},
            )

            if cfg.early_stopping_generations is not None and stall >= cfg.early_stopping_generations:
                logger.info("early stopping", extra={"event": "ga_early_stop", "run": self.run_name, "generation": gen})
                break
            if gen < cfg.generations:
                population = self._next_generation(population, fitnesses)

        elapsed = time.perf_counter() - start
        return GAResult(
            best_params=best_params,
            best_evaluation=best_eval,
            history=history,
            n_evaluations=self.n_evaluations,
            elapsed_seconds=elapsed,
            config=cfg,
        )


def random_search(
    space: SearchSpace,
    fitness_fn: FitnessFunction,
    n_evaluations: int,
    seed: int = 42,
    n_jobs: int = 1,
) -> tuple[dict[str, Any], Evaluation, list[float]]:
    """Linha de base para comparar com o GA: amostra `n_evaluations` configurações
    distintas aleatoriamente (mesmo orçamento de avaliações do GA).

    Retorna (melhores params, avaliação, curva do melhor-até-agora por avaliação).
    """
    rng = np.random.default_rng(seed)
    candidates: dict[tuple, dict[str, Any]] = {}
    attempts = 0
    while len(candidates) < n_evaluations and attempts < n_evaluations * 50:
        attempts += 1
        params = space.decode(space.random_genome(rng))
        candidates.setdefault(params_key(params), params)

    params_list = list(candidates.values())
    evaluations = evaluate_many(fitness_fn, params_list, n_jobs)
    best_idx = int(np.argmax([e.fitness for e in evaluations]))
    curve = np.maximum.accumulate([e.fitness for e in evaluations]).tolist()
    return params_list[best_idx], evaluations[best_idx], curve
