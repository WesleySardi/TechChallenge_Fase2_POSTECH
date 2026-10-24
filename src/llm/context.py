"""Transforma a saída numérica do modelo em um "contexto clínico" estruturado.

A LLM não recebe o modelo nem o dataset — recebe apenas este contexto, já com:
- predição, probabilidade de malignidade e nível de confiança;
- os fatores (features) que mais pesaram NESTE paciente, via SHAP, em pontos de
  probabilidade de malignidade;
- onde o valor do paciente se situa em relação aos casos benignos e malignos da base
  de treino (percentil e medianas) — é isso que vira "insight acionável";
- desempenho do modelo no teste (para a LLM comunicar a confiabilidade).

Limitar a LLM a esse contexto é a principal defesa contra alucinação: tudo que ela
afirmar deve ser rastreável a um campo daqui (ver `quality.py`).
"""

import logging
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

MALIGNANT = 0  # codificação do sklearn (0 = malignant, 1 = benign)

_BASE_FEATURES_PT = {
    "radius": "raio do núcleo celular",
    "texture": "textura (variação dos tons de cinza)",
    "perimeter": "perímetro do núcleo",
    "area": "área do núcleo",
    "smoothness": "suavidade do contorno (variação local do raio)",
    "compactness": "compacidade (perímetro² / área − 1)",
    "concavity": "concavidade (profundidade das reentrâncias do contorno)",
    "concave points": "pontos côncavos (nº de reentrâncias do contorno)",
    "symmetry": "simetria do núcleo",
    "fractal dimension": "dimensão fractal (irregularidade do contorno)",
}
_PREFIX_PT = {"mean": "média", "worst": "pior valor", "error": "erro padrão"}


def feature_name_pt(feature: str) -> str:
    """'worst concave points' -> 'pontos côncavos (...) — pior valor'."""
    for prefix, label in _PREFIX_PT.items():
        if feature.startswith(prefix + " "):
            base = feature[len(prefix) + 1:]
            return f"{_BASE_FEATURES_PT.get(base, base)} — {label}"
        if feature.endswith(" " + prefix):
            base = feature[: -len(prefix) - 1]
            return f"{_BASE_FEATURES_PT.get(base, base)} — {label}"
    return feature


def confidence_level(prob_malignant: float) -> str:
    """Faixas de confiança pela distância da probabilidade ao limiar de 0.5."""
    distance = abs(prob_malignant - 0.5)
    if distance >= 0.35:
        return "alta"
    if distance >= 0.15:
        return "moderada"
    return "baixa"


@dataclass
class FactorContext:
    feature: str
    feature_pt: str
    value: float
    contribution: float            # pontos de probabilidade de malignidade (+ aumenta risco)
    effect: str                    # "aumenta_risco" | "reduz_risco"
    percentile: float              # percentil do valor na base de treino (0–100)
    median_benign: float
    median_malignant: float
    closer_to: str                 # "malignos" | "benignos"


@dataclass
class CaseContext:
    patient_id: str
    model_name: str
    predicted_label: str           # "maligno" | "benigno"
    prob_malignant: float
    confidence: str
    base_rate_malignant: float     # prob. média de malignidade na base (ponto de partida do SHAP)
    top_factors: list[FactorContext]
    model_performance: dict[str, float] = field(default_factory=dict)
    true_label: str | None = None  # só para avaliação — NUNCA vai para o prompt
    clinical_notes: str | None = None  # gancho p/ Módulo 3 (dados textuais)

    @classmethod
    def from_prompt_dict(cls, d: dict[str, Any], true_label: str | None = None) -> "CaseContext":
        """Reconstrói o contexto a partir do que foi enviado à LLM (ex.: para reavaliar
        explicações salvas sem gerar de novo)."""
        d = {k: v for k, v in d.items() if k in {f.name for f in fields(cls)}}  # ignora campos derivados
        factor_fields = {f.name for f in fields(FactorContext)}
        factors = [FactorContext(**{k: v for k, v in f.items() if k in factor_fields}) for f in d.pop("top_factors")]
        return cls(top_factors=factors, true_label=true_label, **d)

    def to_prompt_dict(self, enriched: bool = False) -> dict[str, Any]:
        """Versão enviada à LLM: arredondada e sem o rótulo verdadeiro.

        enriched=True acrescenta valores DERIVADOS já calculados (probabilidade em %,
        distância ao limiar, contribuição em pontos percentuais, ranking de impacto).
        Motivo: na avaliação, os erros residuais das LLMs de 7–8B foram justamente de
        aritmética ("a 22,3 pontos do limiar" com p=0,677) e de ordenação ("o fator que
        mais pesou" ≠ maior |contribuição|). Conta se faz no código, não na LLM.
        """
        d = asdict(self)
        d.pop("true_label")
        if not d["clinical_notes"]:
            d.pop("clinical_notes")
        d["prob_malignant"] = round(d["prob_malignant"], 3)
        d["base_rate_malignant"] = round(d["base_rate_malignant"], 3)
        for f in d["top_factors"]:
            for k in ("value", "median_benign", "median_malignant"):
                f[k] = _round_sig(f[k])
            f["contribution"] = round(f["contribution"], 3)
            f["percentile"] = round(f["percentile"])
        d["model_performance"] = {k: round(v, 3) for k, v in d["model_performance"].items()}
        if enriched:
            p = self.prob_malignant
            d["prob_malignant_pct"] = round(p * 100, 1)
            d["distance_to_threshold_pp"] = round(abs(p - 0.5) * 100, 1)
            d["base_rate_malignant_pct"] = round(self.base_rate_malignant * 100, 1)
            for rank, f in enumerate(d["top_factors"], start=1):
                f["impact_rank"] = rank  # 1 = maior |contribuição| (já vêm ordenados)
                f["contribution_pp"] = round(f["contribution"] * 100, 1)
        return d


def _round_sig(x: float, sig: int = 4) -> float:
    if x == 0 or not np.isfinite(x):
        return float(x)
    return float(round(x, sig - int(np.floor(np.log10(abs(x)))) - 1))


class CaseContextBuilder:
    """Constrói `CaseContext` para pacientes, dado um pipeline treinado e a base de treino."""

    def __init__(self, pipeline: Pipeline, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                 model_performance: dict[str, float] | None = None, top_k: int = 5,
                 background_size: int = 100, seed: int = 42):
        self.pipeline = pipeline
        self.model_name = model_name
        self.X_train = X_train
        self.y_train = y_train
        self.model_performance = model_performance or {}
        self.top_k = top_k

        background = X_train.sample(min(background_size, len(X_train)), random_state=seed)
        # Explainer agnóstico ao modelo (permutation) sobre a probabilidade de MALIGNIDADE:
        # funciona igual para LR, árvore, KNN e RF, e o valor SHAP fica em "pontos de
        # probabilidade" — unidade fácil de comunicar ao médico.
        self._explainer = shap.Explainer(
            self._prob_malignant, shap.maskers.Independent(background), seed=seed
        )
        self._medians_benign = X_train[y_train != MALIGNANT].median()
        self._medians_malignant = X_train[y_train == MALIGNANT].median()

    def _prob_malignant(self, X) -> np.ndarray:
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.X_train.columns)
        classes = list(self.pipeline.classes_)
        return self.pipeline.predict_proba(X_df)[:, classes.index(MALIGNANT)]

    def build(self, X: pd.DataFrame, patient_ids: list[str] | None = None, y_true: pd.Series | None = None,
              clinical_notes: list[str | None] | None = None) -> list[CaseContext]:
        X = X[self.X_train.columns]
        probs = self._prob_malignant(X)
        explanation = self._explainer(X)
        contexts = []
        for i in range(len(X)):
            row = X.iloc[i]
            contributions = pd.Series(explanation.values[i], index=X.columns)
            top = contributions.abs().sort_values(ascending=False).head(self.top_k).index
            factors = [self._factor(name, float(row[name]), float(contributions[name])) for name in top]
            prob = float(probs[i])
            contexts.append(CaseContext(
                patient_id=patient_ids[i] if patient_ids else str(X.index[i]),
                model_name=self.model_name,
                predicted_label="maligno" if prob >= 0.5 else "benigno",
                prob_malignant=prob,
                confidence=confidence_level(prob),
                base_rate_malignant=float(np.atleast_1d(explanation.base_values[i])[0]),
                top_factors=factors,
                model_performance=dict(self.model_performance),
                true_label=None if y_true is None else ("maligno" if y_true.iloc[i] == MALIGNANT else "benigno"),
                clinical_notes=clinical_notes[i] if clinical_notes else None,
            ))
        return contexts

    def _factor(self, name: str, value: float, contribution: float) -> FactorContext:
        med_b, med_m = float(self._medians_benign[name]), float(self._medians_malignant[name])
        return FactorContext(
            feature=name,
            feature_pt=feature_name_pt(name),
            value=value,
            contribution=contribution,
            effect="aumenta_risco" if contribution > 0 else "reduz_risco",
            percentile=float((self.X_train[name] < value).mean() * 100),
            median_benign=med_b,
            median_malignant=med_m,
            closer_to="malignos" if abs(value - med_m) < abs(value - med_b) else "benignos",
        )
