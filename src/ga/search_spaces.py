"""Espaços de busca (cromossomos) dos 4 modelos da Fase 1 e construção do estimador.

Os hiperparâmetros escolhidos são os que mais influenciam viés/variância de cada
modelo. `class_weight` aparece em todos que o suportam porque o dataset é
desbalanceado (~37% malignos) e o objetivo clínico prioriza recall de malignos.
"""

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.ga.genes import CategoricalGene, FloatGene, IntGene, SearchSpace
from src.modeling import RANDOM_STATE

SEARCH_SPACES: dict[str, SearchSpace] = {
    "logistic_regression": SearchSpace([
        FloatGene("C", 1e-3, 1e3, log=True),          # inverso da regularização
        FloatGene("l1_ratio", 0.0, 1.0),               # 0 = L2 (ridge), 1 = L1 (lasso)
        CategoricalGene("class_weight", (None, "balanced")),
    ]),
    "decision_tree": SearchSpace([
        IntGene("max_depth", 2, 20),
        IntGene("min_samples_split", 2, 30),
        IntGene("min_samples_leaf", 1, 15),
        CategoricalGene("criterion", ("gini", "entropy", "log_loss")),
        CategoricalGene("max_features", (None, "sqrt", "log2")),
        CategoricalGene("class_weight", (None, "balanced")),
    ]),
    "knn": SearchSpace([
        IntGene("n_neighbors", 1, 40),
        CategoricalGene("weights", ("uniform", "distance")),
        IntGene("p", 1, 2),                            # 1 = Manhattan, 2 = Euclidiana
    ]),
    "random_forest": SearchSpace([
        IntGene("n_estimators", 50, 400),
        IntGene("max_depth", 2, 20),
        IntGene("min_samples_leaf", 1, 10),
        CategoricalGene("max_features", ("sqrt", "log2", 0.3, 0.5)),
        CategoricalGene("criterion", ("gini", "entropy")),
        CategoricalGene("class_weight", (None, "balanced", "balanced_subsample")),
    ]),
}

# Hiperparâmetros usados na Fase 1 (src/modeling.py::MODEL_REGISTRY), expressos no
# espaço de busca acima — usados como indivíduo "semente" da população inicial.
BASELINE_PARAMS: dict[str, dict[str, Any]] = {
    "logistic_regression": {"C": 1.0, "l1_ratio": 0.0, "class_weight": None},
    "decision_tree": {
        "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1,
        "criterion": "gini", "max_features": None, "class_weight": None,
    },
    "knn": {"n_neighbors": 7, "weights": "uniform", "p": 2},
    "random_forest": {
        "n_estimators": 300, "max_depth": 6, "min_samples_leaf": 1,
        "max_features": "sqrt", "criterion": "gini", "class_weight": None,
    },
}


def build_estimator(model_name: str, params: dict[str, Any]) -> BaseEstimator:
    if model_name == "logistic_regression":
        # saga é o solver que suporta elastic-net (l1_ratio entre 0 e 1).
        return LogisticRegression(solver="saga", max_iter=10000, random_state=RANDOM_STATE, **params)
    if model_name == "decision_tree":
        return DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
    if model_name == "knn":
        return KNeighborsClassifier(**params)
    if model_name == "random_forest":
        # n_jobs=1: o paralelismo fica no GA (vários indivíduos ao mesmo tempo), que
        # escala melhor que paralelizar as árvores de um único modelo pequeno.
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, **params)
    raise KeyError(f"Modelo desconhecido: {model_name}")
