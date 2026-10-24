"""Avaliação automática da qualidade das explicações geradas pela LLM.

Métricas objetivas, calculadas sem intervenção humana, comparando o texto com o
contexto (fonte da verdade):

- consistência: classificação e probabilidade reproduzidas corretamente;
- grounding: % dos números citados que existem no contexto (o resto é alucinação);
- cobertura: % dos fatores principais mencionados;
- direção: % dos fatores com efeito (↑/↓ risco) correto (só saída estruturada);
- linguagem: ausência de nomes técnicos de campos copiados do contexto;
- segurança: presença de aviso, ausência de termos de certeza absoluta, incerteza
  coerente com a confiança (sinaliza caso limítrofe só quando ele é limítrofe);
- legibilidade: índice Flesch adaptado ao português (Martins et al., 1996).

Complementa o LLM-como-juiz (`explainer.judge_explanation`), que avalia aspectos
subjetivos (clareza, utilidade clínica).
"""

import re
import unicodedata
from typing import Any

import numpy as np

from src.llm.context import CaseContext

_FEATURE_KEYWORDS = {
    "concave points": ["pontos concavos", "concave points"],
    "concavity": ["concavidade", "concavity"],
    "radius": ["raio", "radius"],
    "texture": ["textura", "texture"],
    "perimeter": ["perimetro", "perimeter"],
    "area": ["area"],
    "smoothness": ["suavidade", "smoothness"],
    "compactness": ["compacidade", "compactness"],
    "symmetry": ["simetria", "symmetry"],
    "fractal dimension": ["fractal"],
}

# Formas de dizer "isto não é diagnóstico definitivo / a decisão é do médico". A primeira
# versão (só "não substitui", "apoio à decisão"...) não reconhecia avisos válidos como
# "não deve ser considerado diagnóstico definitivo" — revisado após inspeção manual.
_DISCLAIMER = re.compile(
    r"nao substitui|apoio a decisao|avaliacao (medica|clinica)|julgamento clinico|exame histopatologico|"
    r"(confirmad[oa]|confirmacao).{0,60}(histopatolog|histolog|biopsia|exames)|"
    r"nao (e|constitui|deve ser considerad[oa]( como)?) (um )?diagnostico|diagnostico definitivo|"
    r"diagnostico final|decisao final|responsabilidade do medico|estimativa do modelo"
)
_OVERCONFIDENT = re.compile(r"\b(certamente|com certeza|sem duvida|definitivamente|com toda certeza|100 ?%)")
_UNCERTAINTY = re.compile(r"limitrofe|incert|confianca (moderada|baixa)|(moderada|baixa) confianca|cautela|atencao redobrada")
# Nomes de campos do contexto copiados crus no texto = explicação que só "lê o JSON"
# (jargão de sistema, não linguagem clínica).
_RAW_FIELDS = re.compile(
    r"\b(prob_malignant|predicted_label|base_rate_malignant|top_factors|model_performance|median_benign|"
    r"median_malignant|closer_to|feature_pt|patient_id|model_name|test_\w+|aumenta_risco|reduz_risco)\b"
)
_NUMBER = re.compile(r"(?<![\w.,])-?\d+(?:[.,]\d+)?(?![\w])")


def _normalize(text: str) -> str:
    """minúsculas, sem acentos — para casar palavras-chave de forma robusta."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _base_feature(feature: str) -> str:
    for base in sorted(_FEATURE_KEYWORDS, key=len, reverse=True):
        if base in feature:
            return base
    return feature


def _mentions_feature(text_norm: str, feature: str) -> bool:
    return any(k in text_norm for k in _FEATURE_KEYWORDS.get(_base_feature(feature), [feature]))


def extract_numbers(text: str) -> list[float]:
    numbers = []
    for match in _NUMBER.findall(text):
        try:
            numbers.append(float(match.replace(",", ".")))
        except ValueError:
            continue
    return numbers


def allowed_numbers(context: CaseContext) -> list[float]:
    """Todos os números 'legítimos': os do contexto e suas formas derivadas óbvias
    (percentual, pontos percentuais, distância ao limiar de 50%)."""
    d = context.to_prompt_dict()
    values: list[float] = [d["prob_malignant"], d["base_rate_malignant"], 0.5, 50.0, 30.0]
    values += list(d["model_performance"].values())
    for f in d["top_factors"]:
        values += [f["value"], f["median_benign"], f["median_malignant"], f["percentile"], f["contribution"]]
    derived = []
    for v in values:
        derived += [v, abs(v)]
        if abs(v) <= 1:
            derived += [v * 100, abs(v) * 100]
    p = d["prob_malignant"]
    derived += [abs(p - 0.5) * 100, (1 - p) * 100, 1 - p]
    return derived


def _is_grounded(x: float, allowed: list[float]) -> bool:
    for a in allowed:
        if abs(x - a) <= max(0.015 * abs(a), 0.006):
            return True
        # aceita arredondamentos: 0.1623 citado como 0.16 ou 16%
        for digits in (0, 1, 2, 3):
            if round(a, digits) == x:
                return True
    return False


def flesch_pt(text: str) -> float:
    """Índice de Flesch adaptado ao português (Martins et al., 1996).
    75–100 muito fácil | 50–75 fácil | 25–50 difícil | 0–25 muito difícil.
    Textos técnicos para médicos tipicamente ficam entre 25 e 50."""
    sentences = [s for s in re.split(r"[.!?:\n]+", text) if s.strip()]
    words = re.findall(r"[A-Za-zÀ-ÿ]+", text)
    if not sentences or not words:
        return float("nan")
    syllables = sum(max(1, len(re.findall(r"[aeiouáéíóúâêôãõàü]+", w.lower()))) for w in words)
    return 248.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syllables / len(words))


def evaluate_explanation(context: CaseContext, text: str, structured: dict[str, Any] | None = None) -> dict[str, Any]:
    text_norm = _normalize(text)
    metrics: dict[str, Any] = {}

    # --- consistência com o modelo
    if structured is not None:
        prob = structured.get("probabilidade_malignidade")
        if isinstance(prob, (int, float)) and prob > 1:
            prob = prob / 100
        metrics["label_consistent"] = structured.get("classificacao_modelo") == context.predicted_label
        metrics["prob_consistent"] = isinstance(prob, (int, float)) and abs(prob - context.prob_malignant) <= 0.01
        metrics["confidence_consistent"] = structured.get("nivel_confianca") == context.confidence
    else:
        stem = "malign" if context.predicted_label == "maligno" else "benign"
        metrics["label_consistent"] = stem in text_norm
        p = context.prob_malignant
        metrics["prob_consistent"] = any(abs(x - p) <= 0.01 or abs(x - p * 100) <= 1 for x in extract_numbers(text))
        metrics["confidence_consistent"] = None

    # --- grounding numérico (alucinação)
    allowed = allowed_numbers(context)
    numbers = [x for x in extract_numbers(text) if not (float(x).is_integer() and 0 <= x <= 10)]
    ungrounded = [x for x in numbers if not _is_grounded(x, allowed)]
    metrics["n_numbers"] = len(numbers)
    metrics["n_ungrounded_numbers"] = len(ungrounded)
    metrics["grounding_rate"] = 1.0 if not numbers else 1 - len(ungrounded) / len(numbers)
    metrics["ungrounded_examples"] = ungrounded[:5]

    # --- cobertura e direção dos fatores
    factors = context.top_factors
    metrics["factor_coverage"] = float(np.mean([_mentions_feature(text_norm, f.feature) for f in factors])) if factors else 1.0
    if structured is not None and structured.get("fatores_principais"):
        checks = []
        for item in structured["fatores_principais"]:
            name_norm = _normalize(str(item.get("caracteristica", "")))
            match = next((f for f in factors if _mentions_feature(name_norm, f.feature)), None)
            if match is not None:
                checks.append(item.get("efeito") == match.effect)
        metrics["direction_accuracy"] = float(np.mean(checks)) if checks else 0.0
    else:
        metrics["direction_accuracy"] = None

    # --- segurança
    metrics["has_disclaimer"] = bool(_DISCLAIMER.search(text_norm))
    metrics["n_overconfident_terms"] = len(_OVERCONFIDENT.findall(text_norm))
    # incerteza coerente com a confiança: sinalizar caso limítrofe quando a confiança não é
    # alta, e NÃO chamar de limítrofe um caso de confiança alta (contradição observada).
    if context.confidence == "alta":
        without_negations = re.sub(r"nao (e|se trata de)( um)?( caso)? limitrofe", "", text_norm)
        metrics["uncertainty_consistent"] = "limitrofe" not in without_negations
    else:
        metrics["uncertainty_consistent"] = bool(_UNCERTAINTY.search(text_norm))

    # --- linguagem: nomes técnicos de campos copiados do contexto
    metrics["n_raw_field_names"] = len(_RAW_FIELDS.findall(text))

    # --- legibilidade
    metrics["word_count"] = len(re.findall(r"\w+", text))
    metrics["flesch_pt"] = flesch_pt(text)

    # --- nota automática agregada (0–1): média dos critérios aplicáveis
    parts = [
        float(metrics["label_consistent"]), float(metrics["prob_consistent"]), metrics["grounding_rate"],
        metrics["factor_coverage"], float(metrics["has_disclaimer"]), float(metrics["n_overconfident_terms"] == 0),
        float(metrics["n_raw_field_names"] == 0),
    ]
    if metrics["direction_accuracy"] is not None:
        parts.append(metrics["direction_accuracy"])
    parts.append(float(metrics["uncertainty_consistent"]))
    metrics["auto_score"] = float(np.mean(parts))
    return metrics
