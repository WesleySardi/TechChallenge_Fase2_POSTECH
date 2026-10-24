"""Operadores genéticos: seleção, cruzamento e mutação.

Todos operam sobre genomas normalizados em [0, 1) (ver `genes.py`), então valem
para qualquer espaço de busca.
"""

import numpy as np

from src.ga.genes import clip_genome


# ---------------------------------------------------------------- seleção

def tournament_selection(fitnesses: np.ndarray, rng: np.random.Generator, k: int = 3) -> int:
    """Sorteia k indivíduos e devolve o índice do melhor.

    k controla a pressão seletiva: k maior favorece mais os melhores (converge mais
    rápido, mas com mais risco de convergência prematura).
    """
    contenders = rng.choice(len(fitnesses), size=min(k, len(fitnesses)), replace=False)
    return int(contenders[np.argmax(fitnesses[contenders])])


def roulette_selection(fitnesses: np.ndarray, rng: np.random.Generator, **_) -> int:
    """Seleção proporcional ao fitness (roleta).

    As métricas aqui ficam todas em uma faixa estreita (ex.: 0.90–0.98), o que tornaria
    a roleta quase uniforme. Por isso desloco pelo pior fitness da população — a
    probabilidade passa a ser proporcional à *vantagem* sobre o pior indivíduo.
    """
    shifted = fitnesses - fitnesses.min() + 1e-9
    probs = shifted / shifted.sum()
    return int(rng.choice(len(fitnesses), p=probs))


SELECTION_OPERATORS = {
    "tournament": tournament_selection,
    "roulette": roulette_selection,
}


# ---------------------------------------------------------------- cruzamento

def uniform_crossover(parent1: np.ndarray, parent2: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Cada gene do filho vem de um dos pais com probabilidade 0.5.

    Adequado aqui porque os hiperparâmetros são, em geral, independentes entre si —
    não há "blocos" de genes que precisem ser herdados juntos.
    """
    mask = rng.random(len(parent1)) < 0.5
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return child1, child2


def blend_crossover(parent1: np.ndarray, parent2: np.ndarray, rng: np.random.Generator, alpha: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """BLX-alpha: cada gene do filho é sorteado num intervalo que envolve os genes dos
    pais, estendido em alpha * |p1 - p2| para cada lado. Explora valores *entre* e
    *ao redor* dos pais — útil para genes contínuos (C, max_features...)."""
    low = np.minimum(parent1, parent2)
    high = np.maximum(parent1, parent2)
    span = high - low
    low, high = low - alpha * span, high + alpha * span
    child1 = rng.uniform(low, high)
    child2 = rng.uniform(low, high)
    return clip_genome(child1), clip_genome(child2)


def single_point_crossover(parent1: np.ndarray, parent2: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if len(parent1) < 2:
        return parent1.copy(), parent2.copy()
    point = int(rng.integers(1, len(parent1)))
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2


CROSSOVER_OPERATORS = {
    "uniform": uniform_crossover,
    "blend": blend_crossover,
    "single_point": single_point_crossover,
}


# ---------------------------------------------------------------- mutação

def gaussian_mutation(genome: np.ndarray, rng: np.random.Generator, rate: float, sigma: float) -> np.ndarray:
    """Cada gene sofre, com probabilidade `rate`, um ruído N(0, sigma).

    Pequenas perturbações refinam a solução (exploitation); para não ficar preso,
    `random_reset_mutation` complementa com saltos grandes ocasionais.
    """
    mutated = genome.copy()
    mask = rng.random(len(genome)) < rate
    mutated[mask] += rng.normal(0.0, sigma, size=mask.sum())
    return clip_genome(mutated)


def random_reset_mutation(genome: np.ndarray, rng: np.random.Generator, rate: float) -> np.ndarray:
    """Cada gene é, com probabilidade `rate`, sorteado de novo em [0, 1)."""
    mutated = genome.copy()
    mask = rng.random(len(genome)) < rate
    mutated[mask] = rng.random(mask.sum())
    return mutated
