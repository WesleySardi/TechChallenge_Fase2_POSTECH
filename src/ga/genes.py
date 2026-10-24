"""Codificação genética dos hiperparâmetros.

Cada indivíduo é um vetor de floats em [0, 1) — um gene por hiperparâmetro.
A decodificação para o valor real fica a cargo de cada tipo de gene:

- IntGene: inteiro em [low, high], mapeado linearmente;
- FloatGene: real em [low, high], linear ou em escala log (útil para C, learning rate);
- CategoricalGene: escolha entre opções discretas (o intervalo [0, 1) é dividido em
  faixas iguais, uma por opção).

Essa representação "real normalizada" permite usar os MESMOS operadores de crossover e
mutação para qualquer modelo, independentemente do tipo de cada hiperparâmetro.
"""

import math
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

# Limite superior exclusivo: garante que floor(u * n) nunca estoure o índice.
_U_MAX = np.nextafter(1.0, 0.0)


class Gene(Protocol):
    name: str

    def decode(self, u: float) -> Any: ...

    def encode(self, value: Any) -> float: ...


@dataclass(frozen=True)
class IntGene:
    name: str
    low: int
    high: int

    def decode(self, u: float) -> int:
        n_values = self.high - self.low + 1
        return self.low + min(int(u * n_values), n_values - 1)

    def encode(self, value: int) -> float:
        n_values = self.high - self.low + 1
        # Centro da faixa correspondente ao valor (robusto a pequenas mutações).
        return (value - self.low + 0.5) / n_values


@dataclass(frozen=True)
class FloatGene:
    name: str
    low: float
    high: float
    log: bool = False

    def decode(self, u: float) -> float:
        if self.log:
            log_low, log_high = math.log10(self.low), math.log10(self.high)
            return float(10 ** (log_low + u * (log_high - log_low)))
        return float(self.low + u * (self.high - self.low))

    def encode(self, value: float) -> float:
        if self.log:
            log_low, log_high = math.log10(self.low), math.log10(self.high)
            return (math.log10(value) - log_low) / (log_high - log_low)
        return (value - self.low) / (self.high - self.low)


@dataclass(frozen=True)
class CategoricalGene:
    name: str
    choices: tuple

    def decode(self, u: float) -> Any:
        n = len(self.choices)
        return self.choices[min(int(u * n), n - 1)]

    def encode(self, value: Any) -> float:
        return (self.choices.index(value) + 0.5) / len(self.choices)


class SearchSpace:
    """Conjunto ordenado de genes que define o cromossomo de um modelo."""

    def __init__(self, genes: list[Gene]):
        names = [g.name for g in genes]
        if len(names) != len(set(names)):
            raise ValueError(f"Nomes de genes duplicados: {names}")
        self.genes = genes

    def __len__(self) -> int:
        return len(self.genes)

    @property
    def names(self) -> list[str]:
        return [g.name for g in self.genes]

    def random_genome(self, rng: np.random.Generator) -> np.ndarray:
        return rng.random(len(self.genes))

    def decode(self, genome: np.ndarray) -> dict[str, Any]:
        return {g.name: g.decode(float(u)) for g, u in zip(self.genes, clip_genome(genome))}

    def encode(self, params: dict[str, Any]) -> np.ndarray:
        return clip_genome(np.array([g.encode(params[g.name]) for g in self.genes], dtype=float))


def clip_genome(genome: np.ndarray) -> np.ndarray:
    return np.clip(genome, 0.0, _U_MAX)


def params_key(params: dict[str, Any]) -> tuple:
    """Chave hashable de um conjunto de hiperparâmetros decodificados.

    Genomas diferentes podem decodificar para os mesmos hiperparâmetros (ex.: dois
    valores de u que caem na mesma faixa de um IntGene). Cachear pela versão
    decodificada evita reavaliar o mesmo modelo — e a avaliação é a parte cara.
    """
    return tuple(sorted((k, round(v, 10) if isinstance(v, float) else v) for k, v in params.items()))
