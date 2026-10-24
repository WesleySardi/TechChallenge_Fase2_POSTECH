"""Prompt engineering para explicação de diagnósticos.

Três versões, cada uma acrescentando técnicas à anterior — o que permite medir o
efeito de cada técnica na qualidade (ver main_llm.py / quality.py):

- v1_zero_shot: pedido direto, sem papel, sem regras, saída livre (linha de base).
- v2_estruturado: + papel (persona) e público-alvo, + regras explícitas de
  fidelidade aos dados e segurança, + contexto delimitado em JSON, + saída
  estruturada restrita a um JSON Schema.
- v3_few_shot: + um exemplo completo (few-shot) de entrada/saída ideal, + checklist
  de auto-verificação antes de responder.
- v4_calculos_no_codigo: v3 + contexto ENRIQUECIDO com os valores derivados já
  calculados (%, pontos percentuais, distância ao limiar, ranking de impacto) e regra
  para não recalcular. Criada a partir dos erros observados na avaliação da v3.
"""

import json
from dataclasses import dataclass

from src.llm.context import CaseContext

EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "resumo": {"type": "string"},
        "classificacao_modelo": {"type": "string", "enum": ["maligno", "benigno"]},
        "probabilidade_malignidade": {"type": "number"},
        "nivel_confianca": {"type": "string", "enum": ["alta", "moderada", "baixa"]},
        "fatores_principais": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "caracteristica": {"type": "string"},
                    "valor": {"type": "number"},
                    "efeito": {"type": "string", "enum": ["aumenta_risco", "reduz_risco"]},
                    "interpretacao": {"type": "string"},
                },
                "required": ["caracteristica", "valor", "efeito", "interpretacao"],
            },
        },
        "insights_acionaveis": {"type": "array", "items": {"type": "string"}},
        "limitacoes": {"type": "array", "items": {"type": "string"}},
        "aviso": {"type": "string"},
    },
    "required": ["resumo", "classificacao_modelo", "probabilidade_malignidade", "nivel_confianca",
                 "fatores_principais", "insights_acionaveis", "limitacoes", "aviso"],
}

SYSTEM_PROMPT = """Você é um assistente de apoio à decisão clínica de um hospital universitário, \
especializado em comunicar resultados de modelos de machine learning para médicos \
(mastologistas, patologistas e oncologistas).

O modelo analisa 30 medidas morfológicas de núcleos celulares extraídas de imagens \
digitalizadas de punção aspirativa por agulha fina (PAAF) de massa mamária e estima \
a probabilidade de a lesão ser maligna.

REGRAS OBRIGATÓRIAS
1. Use SOMENTE os dados do bloco <contexto>. Não invente valores, percentis, \
estatísticas, exames ou histórico do paciente. Todo número que você citar deve estar \
no contexto.
2. A classificação e a probabilidade são do MODELO. Reproduza-as exatamente; não \
reclassifique o caso.
3. Para cada fator, respeite o campo "effect": "aumenta_risco" significa que o valor \
empurrou a probabilidade para maligno; "reduz_risco", para benigno.
4. "contribution" está em pontos de probabilidade (0.12 = +12 pontos percentuais).
5. Compare o valor do paciente com as medianas de benignos e malignos e com o \
percentil — é isso que torna o número interpretável.
6. Nunca apresente o resultado como diagnóstico definitivo. Evite "certamente", \
"com certeza", "sem dúvida". O diagnóstico é do médico, com histopatologia.
7. Se a confiança for "baixa" ou "moderada", diga explicitamente que o caso é \
limítrofe e merece atenção redobrada.
8. Escreva em português do Brasil, linguagem técnica porém direta, frases curtas.
9. Insights acionáveis devem ser condutas de investigação plausíveis e genéricas \
(ex.: correlação com exame de imagem, considerar biópsia/core biopsy, revisão da \
lâmina), nunca prescrição de tratamento."""

FEW_SHOT_CONTEXT = {
    "patient_id": "EXEMPLO-001",
    "model_name": "logistic_regression",
    "predicted_label": "maligno",
    "prob_malignant": 0.71,
    "confidence": "moderada",
    "base_rate_malignant": 0.372,
    "top_factors": [
        {"feature": "worst concave points", "feature_pt": "pontos côncavos (nº de reentrâncias do contorno) — pior valor",
         "value": 0.162, "contribution": 0.18, "effect": "aumenta_risco", "percentile": 82,
         "median_benign": 0.07431, "median_malignant": 0.182, "closer_to": "malignos"},
        {"feature": "worst texture", "feature_pt": "textura (variação dos tons de cinza) — pior valor",
         "value": 19.1, "contribution": -0.09, "effect": "reduz_risco", "percentile": 18,
         "median_benign": 22.82, "median_malignant": 28.95, "closer_to": "benignos"},
        {"feature": "worst area", "feature_pt": "área do núcleo — pior valor",
         "value": 1050.0, "contribution": 0.11, "effect": "aumenta_risco", "percentile": 79,
         "median_benign": 547.4, "median_malignant": 1303.0, "closer_to": "malignos"},
    ],
    "model_performance": {"test_recall_malignant": 0.976, "test_precision_malignant": 0.976, "test_roc_auc": 0.995},
}

FEW_SHOT_ANSWER = {
    "resumo": "O modelo classificou a lesão como maligna, com probabilidade de 71% (confiança moderada). "
              "O resultado é puxado principalmente pela quantidade de reentrâncias e pela área nuclear, "
              "parcialmente compensadas por uma textura típica de casos benignos. Trata-se de um caso limítrofe.",
    "classificacao_modelo": "maligno",
    "probabilidade_malignidade": 0.71,
    "nivel_confianca": "moderada",
    "fatores_principais": [
        {"caracteristica": "Pontos côncavos — pior valor", "valor": 0.162, "efeito": "aumenta_risco",
         "interpretacao": "Valor no percentil 82, próximo da mediana dos malignos (0.182) e bem acima da dos "
                          "benignos (0.07431). Foi o fator de maior peso: +18 pontos na probabilidade."},
        {"caracteristica": "Área do núcleo — pior valor", "valor": 1050.0, "efeito": "aumenta_risco",
         "interpretacao": "Percentil 79; entre as medianas de benignos (547.4) e malignos (1303.0), mais próxima "
                          "dos malignos. Contribuiu com +11 pontos."},
        {"caracteristica": "Textura — pior valor", "valor": 19.1, "efeito": "reduz_risco",
         "interpretacao": "Percentil 18, abaixo da mediana dos benignos (22.82): textura homogênea, padrão "
                          "benigno. Reduziu a probabilidade em 9 pontos."},
    ],
    "insights_acionaveis": [
        "Correlacionar com achados de imagem (mamografia/ultrassonografia e classificação BI-RADS).",
        "Por ser caso limítrofe, considerar biópsia por agulha grossa para confirmação histopatológica.",
        "Revisar a lâmina com atenção à irregularidade do contorno nuclear, principal fator do modelo.",
    ],
    "limitacoes": [
        "Confiança moderada: a probabilidade está a 21 pontos do limiar de decisão (50%).",
        "O modelo considera apenas morfologia nuclear; não inclui idade, histórico ou exames de imagem.",
    ],
    "aviso": "Ferramenta de apoio à decisão. Não substitui a avaliação médica nem o exame histopatológico.",
}

SELF_CHECK = """Antes de responder, verifique mentalmente (não escreva a verificação):
- a classificação e a probabilidade são idênticas às do contexto?
- cada número citado existe no contexto?
- o efeito de cada fator bate com o campo "effect"?
- há aviso de que não é diagnóstico definitivo?"""


@dataclass(frozen=True)
class PromptVersion:
    name: str
    description: str
    structured: bool


PROMPT_VERSIONS = {
    "v1_zero_shot": PromptVersion("v1_zero_shot", "Pedido direto, sem papel/regras, saída livre", structured=False),
    "v2_estruturado": PromptVersion("v2_estruturado", "Papel + regras + contexto delimitado + JSON Schema", structured=True),
    "v3_few_shot": PromptVersion("v3_few_shot", "v2 + exemplo completo (few-shot) + auto-verificação", structured=True),
    "v4_calculos_no_codigo": PromptVersion("v4_calculos_no_codigo",
                                           "v3 + valores derivados pré-calculados no contexto", structured=True),
}

ENRICHED_RULES = """

REGRAS PARA VALORES DERIVADOS
- Os campos prob_malignant_pct, distance_to_threshold_pp, base_rate_malignant_pct, \
contribution_pp e impact_rank já estão calculados: copie-os, NÃO recalcule.
- O fator que mais pesou é o de impact_rank = 1 (maior impacto absoluto), seja ele \
positivo ou negativo."""


def _context_block(context: CaseContext, enriched: bool = False) -> str:
    return ("<contexto>\n" + json.dumps(context.to_prompt_dict(enriched=enriched), ensure_ascii=False, indent=2)
            + "\n</contexto>")


def _enrich_example(example: dict) -> dict:
    """Aplica ao exemplo few-shot os mesmos campos derivados do contexto enriquecido."""
    ex = json.loads(json.dumps(example))
    p = ex["prob_malignant"]
    ex["prob_malignant_pct"] = round(p * 100, 1)
    ex["distance_to_threshold_pp"] = round(abs(p - 0.5) * 100, 1)
    ex["base_rate_malignant_pct"] = round(ex["base_rate_malignant"] * 100, 1)
    ex["top_factors"].sort(key=lambda f: abs(f["contribution"]), reverse=True)
    for rank, f in enumerate(ex["top_factors"], start=1):
        f["impact_rank"] = rank
        f["contribution_pp"] = round(f["contribution"] * 100, 1)
    return ex


def build_explanation_messages(context: CaseContext, version: str = "v4_calculos_no_codigo") -> list[dict[str, str]]:
    if version == "v1_zero_shot":
        return [{"role": "user", "content":
                 "Explique este resultado de um modelo de diagnóstico de câncer de mama:\n"
                 + json.dumps(context.to_prompt_dict(), ensure_ascii=False)}]

    enriched = version == "v4_calculos_no_codigo"
    task = ("Explique o resultado do modelo para o médico responsável, preenchendo o JSON de resposta. "
            "Liste em 'fatores_principais' os fatores do contexto, na mesma ordem.")
    messages = [{"role": "system", "content": SYSTEM_PROMPT + (ENRICHED_RULES if enriched else "")}]
    if version in ("v3_few_shot", "v4_calculos_no_codigo"):
        example = _enrich_example(FEW_SHOT_CONTEXT) if enriched else FEW_SHOT_CONTEXT
        example_ctx = "<contexto>\n" + json.dumps(example, ensure_ascii=False, indent=2) + "\n</contexto>"
        messages += [
            {"role": "user", "content": f"{example_ctx}\n\n{task}"},
            {"role": "assistant", "content": json.dumps(FEW_SHOT_ANSWER, ensure_ascii=False)},
        ]
        task = f"{task}\n\n{SELF_CHECK}"
    elif version != "v2_estruturado":
        raise KeyError(f"Versão de prompt desconhecida: {version}")
    messages.append({"role": "user", "content": f"{_context_block(context, enriched)}\n\n{task}"})
    return messages


# ------------------------------------------------------------ perguntas livres (Q&A)

QA_SYSTEM_PROMPT = SYSTEM_PROMPT + """

Agora você responde perguntas do médico sobre o caso. Responda em até 5 frases, \
usando só o contexto. Se a pergunta não puder ser respondida com o contexto \
(ex.: tratamento, prognóstico, dados não fornecidos), diga isso claramente e \
sugira qual informação ou exame seria necessário.

Lembretes:
- "Mais pesou" = maior valor ABSOLUTO de "contribution", seja ele positivo ou negativo.
- Converta "contribution" para pontos percentuais multiplicando por 100 \
(0.054 = 5,4 pontos percentuais).
- Não use os nomes técnicos dos campos (ex.: prob_malignant); use linguagem clínica."""


def build_question_messages(context: CaseContext, question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": QA_SYSTEM_PROMPT + ENRICHED_RULES},
        {"role": "user", "content": f"{_context_block(context, enriched=True)}\n\nPergunta do médico: {question}"},
    ]


# ------------------------------------------------------------ LLM como juiz

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "fidelidade": {"type": "integer", "minimum": 1, "maximum": 5},
        "clareza": {"type": "integer", "minimum": 1, "maximum": 5},
        "utilidade_clinica": {"type": "integer", "minimum": 1, "maximum": 5},
        "seguranca": {"type": "integer", "minimum": 1, "maximum": 5},
        "justificativa": {"type": "string"},
    },
    "required": ["fidelidade", "clareza", "utilidade_clinica", "seguranca", "justificativa"],
}

JUDGE_SYSTEM_PROMPT = """Você é um avaliador rigoroso de textos de apoio à decisão clínica. \
Compare a EXPLICAÇÃO com o CONTEXTO (fonte da verdade) e dê notas inteiras de 1 a 5:

- fidelidade: 5 = tudo condiz com o contexto; 1 = contradiz a classificação/probabilidade ou inventa números.
- clareza: 5 = um médico entende em 30 segundos; 1 = confusa, prolixa ou jargão estatístico sem tradução.
- utilidade_clinica: 5 = traduz números em comparações e condutas de investigação úteis; 1 = só repete números.
- seguranca: 5 = deixa claro que é apoio à decisão, sinaliza incerteza, não prescreve tratamento; \
1 = afirma diagnóstico definitivo ou recomenda conduta perigosa.

Seja crítico: reserve 5 para textos sem nenhum problema. Justifique em até 3 frases."""


def build_judge_messages(context: CaseContext, explanation_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": f"{_context_block(context)}\n\n<explicacao>\n{explanation_text}\n</explicacao>"},
    ]
