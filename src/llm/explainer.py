"""Geração de explicações em linguagem natural e respostas a perguntas sobre um caso."""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src.llm.client import LLMClient, LLMError
from src.llm.context import CaseContext
from src.llm.prompts import (
    EXPLANATION_SCHEMA,
    JUDGE_SCHEMA,
    PROMPT_VERSIONS,
    build_explanation_messages,
    build_judge_messages,
    build_question_messages,
)

logger = logging.getLogger(__name__)


@dataclass
class Explanation:
    patient_id: str
    prompt_version: str
    model: str
    text: str                              # texto final (markdown) mostrado ao médico
    structured: dict[str, Any] | None      # JSON parseado (versões estruturadas)
    raw: str
    latency_seconds: float
    json_valid: bool | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def parse_json(raw: str) -> dict[str, Any] | None:
    """Parse tolerante: aceita JSON puro ou dentro de bloco ```json```."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def render_markdown(data: dict[str, Any]) -> str:
    """Converte a resposta estruturada no texto exibido ao médico."""
    effect_label = {"aumenta_risco": "↑ risco", "reduz_risco": "↓ risco"}
    prob = data.get("probabilidade_malignidade")
    prob_txt = f"{prob:.0%}" if isinstance(prob, (int, float)) and prob <= 1 else f"{prob}"
    lines = [
        f"**Classificação do modelo:** {data.get('classificacao_modelo', '?')} "
        f"(probabilidade de malignidade {prob_txt}, confiança {data.get('nivel_confianca', '?')})",
        "",
        data.get("resumo", ""),
        "",
        "**Fatores principais**",
    ]
    for f in data.get("fatores_principais", []):
        lines.append(f"- {f.get('caracteristica')} = {f.get('valor')} "
                     f"({effect_label.get(f.get('efeito'), f.get('efeito'))}): {f.get('interpretacao')}")
    if data.get("insights_acionaveis"):
        lines += ["", "**Sugestões de investigação**"] + [f"- {i}" for i in data["insights_acionaveis"]]
    if data.get("limitacoes"):
        lines += ["", "**Limitações**"] + [f"- {i}" for i in data["limitacoes"]]
    if data.get("aviso"):
        lines += ["", f"_{data['aviso']}_"]
    return "\n".join(lines)


class DiagnosisExplainer:
    def __init__(self, client: LLMClient, prompt_version: str = "v4_calculos_no_codigo", temperature: float = 0.2):
        if prompt_version not in PROMPT_VERSIONS:
            raise KeyError(f"Versão de prompt desconhecida: {prompt_version}")
        self.client = client
        self.prompt_version = prompt_version
        self.temperature = temperature

    def explain(self, context: CaseContext) -> Explanation:
        version = PROMPT_VERSIONS[self.prompt_version]
        messages = build_explanation_messages(context, self.prompt_version)
        try:
            response = self.client.chat(
                messages, json_schema=EXPLANATION_SCHEMA if version.structured else None,
                temperature=self.temperature,
            )
        except LLMError as exc:
            logger.error(str(exc), extra={"event": "llm_error", "patient_id": context.patient_id})
            return Explanation(context.patient_id, self.prompt_version, self.client.model, text="", structured=None,
                               raw="", latency_seconds=0.0, error=str(exc))

        if not version.structured:
            return Explanation(context.patient_id, self.prompt_version, response.model, text=response.content,
                               structured=None, raw=response.content, latency_seconds=response.latency_seconds,
                               extra={"completion_tokens": response.completion_tokens})

        data = parse_json(response.content)
        return Explanation(
            patient_id=context.patient_id,
            prompt_version=self.prompt_version,
            model=response.model,
            text=render_markdown(data) if data else response.content,
            structured=data,
            raw=response.content,
            latency_seconds=response.latency_seconds,
            json_valid=data is not None,
            extra={"completion_tokens": response.completion_tokens},
        )

    def answer(self, context: CaseContext, question: str) -> str:
        """Responde uma pergunta livre do médico sobre o caso."""
        response = self.client.chat(build_question_messages(context, question), temperature=self.temperature)
        return response.content.strip()


def judge_explanation(judge: LLMClient, context: CaseContext, explanation_text: str) -> dict[str, Any] | None:
    """LLM-como-juiz: notas 1–5 em fidelidade, clareza, utilidade clínica e segurança."""
    try:
        response = judge.chat(build_judge_messages(context, explanation_text), json_schema=JUDGE_SCHEMA, temperature=0.0)
    except LLMError as exc:
        logger.error(str(exc), extra={"event": "judge_error", "patient_id": context.patient_id})
        return None
    return parse_json(response.content)
