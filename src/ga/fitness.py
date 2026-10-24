"""Função fitness: desempenho do modelo em validação cruzada no conjunto de TREINO.

O conjunto de teste nunca é usado durante a otimização — só na comparação final
(original vs. otimizado). Usá-lo no fitness seria vazamento de dados: o GA
"aprenderia" o teste e a comparação ficaria otimista.
"""

import warnings
from dataclasses import dataclass, field

import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.ga.engine import Evaluation
from src.ga.search_spaces import build_estimator
from src.modeling import RANDOM_STATE
from src.preprocessing import build_preprocessing_pipeline

# Classe 0 = malignant (codificação do sklearn, mantida desde a Fase 1).
MALIGNANT = 0

SCORERS = {
    "accuracy": "accuracy",
    "precision_malignant": make_scorer(precision_score, pos_label=MALIGNANT, zero_division=0),
    "recall_malignant": make_scorer(recall_score, pos_label=MALIGNANT, zero_division=0),
    "f1_malignant": make_scorer(f1_score, pos_label=MALIGNANT, zero_division=0),
    "roc_auc": "roc_auc",
}

# Pesos padrão do fitness composto. Racional clínico:
# - F1 (malignos) equilibra precisão e recall — evita o modelo degenerado que chama
#   tudo de maligno (recall 1.0, precisão ruim);
# - recall (malignos) recebe peso extra porque falso negativo = câncer não detectado;
# - ROC-AUC mede a qualidade da probabilidade, que é o que a LLM vai explicar.
DEFAULT_WEIGHTS = {"f1_malignant": 0.5, "recall_malignant": 0.3, "roc_auc": 0.2}


@dataclass
class CVFitness:
    """Callable de fitness (params -> Evaluation) para um modelo específico."""
    model_name: str
    X_train: pd.DataFrame
    y_train: pd.Series
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    cv_folds: int = 5
    cv_repeats: int = 3       # CV repetida: estimativa menos ruidosa, reduz overfitting aos folds
    std_penalty: float = 0.0  # >0 penaliza configurações instáveis entre folds

    def __post_init__(self):
        unknown = set(self.weights) - set(SCORERS)
        if unknown:
            raise ValueError(f"Métricas desconhecidas nos pesos: {unknown}")
        # Mesmos folds para todos os indivíduos: diferenças de fitness refletem os
        # hiperparâmetros, não a sorte do particionamento.
        self.cv = RepeatedStratifiedKFold(n_splits=self.cv_folds, n_repeats=self.cv_repeats,
                                          random_state=RANDOM_STATE)
        self.preprocessor_template = build_preprocessing_pipeline(
            pd.concat([self.X_train, self.y_train.rename("diagnosis")], axis=1)
        )

    def build_pipeline(self, params: dict) -> Pipeline:
        return Pipeline(steps=[
            ("preprocessor", clone(self.preprocessor_template)),
            ("classifier", build_estimator(self.model_name, params)),
        ])

    def __call__(self, params: dict) -> Evaluation:
        return self.evaluate_pipeline(self.build_pipeline(params))

    def evaluate_pipeline(self, pipeline: Pipeline) -> Evaluation:
        """Avalia um pipeline arbitrário com o mesmo protocolo (folds, métricas, pesos)
        — usado também para medir o modelo ORIGINAL da Fase 1 de forma comparável."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            scores = cross_validate(pipeline, self.X_train, self.y_train, cv=self.cv, scoring=SCORERS, n_jobs=None)

        metrics = {}
        for name in SCORERS:
            values = scores[f"test_{name}"]
            metrics[f"cv_{name}_mean"] = float(values.mean())
            metrics[f"cv_{name}_std"] = float(values.std())

        fitness = sum(w * metrics[f"cv_{m}_mean"] for m, w in self.weights.items())
        if self.std_penalty:
            fitness -= self.std_penalty * sum(w * metrics[f"cv_{m}_std"] for m, w in self.weights.items())
        metrics["fit_time_mean"] = float(scores["fit_time"].mean())
        return Evaluation(fitness=float(fitness), metrics=metrics)
