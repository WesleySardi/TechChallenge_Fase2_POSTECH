"""Contratos (Pydantic) da API — viram a documentação OpenAPI em /docs."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

PromptVersionName = Literal["v1_zero_shot", "v2_estruturado", "v3_few_shot", "v4_calculos_no_codigo"]


class PatientFeatures(BaseModel):
    patient_id: str | None = Field(None, examples=["T042"])
    features: dict[str, float] = Field(..., description="As 30 medidas morfológicas (nomes do dataset Wisconsin)")
    clinical_notes: str | None = Field(
        None, description="Texto clínico livre (reservado para o Módulo 3; repassado à LLM como contexto)")

    @field_validator("features")
    @classmethod
    def finite_values(cls, v: dict[str, float]) -> dict[str, float]:
        bad = [k for k, x in v.items() if x != x or x in (float("inf"), float("-inf"))]
        if bad:
            raise ValueError(f"Valores não finitos em: {bad}")
        return v


class PredictBatchRequest(BaseModel):
    patients: list[PatientFeatures] = Field(..., min_length=1, max_length=500)


class Prediction(BaseModel):
    patient_id: str | None
    label: Literal["maligno", "benigno"]
    prob_malignant: float
    confidence: Literal["alta", "moderada", "baixa"]
    model: str


class ExplainRequest(BaseModel):
    patient: PatientFeatures
    prompt_version: PromptVersionName = "v4_calculos_no_codigo"
    use_llm: bool = Field(True, description="false = só predição + fatores SHAP, sem chamar a LLM")


class ExplainResponse(BaseModel):
    context: dict[str, Any]
    explanation_markdown: str | None = None
    explanation_structured: dict[str, Any] | None = None
    quality: dict[str, Any] | None = None
    llm: str | None = None
    llm_latency_seconds: float | None = None


class AskRequest(BaseModel):
    patient: PatientFeatures
    question: str = Field(..., min_length=3, max_length=1000, examples=["O quanto posso confiar nesse resultado?"])


class AskResponse(BaseModel):
    question: str
    answer: str
    prediction: str
    prob_malignant: float
    llm: str


class ModelInfo(BaseModel):
    name: str
    params: dict[str, Any]
    cv_fitness: float
    test_performance: dict[str, float]
    feature_names: list[str]
    llm: str | None
