"""API de diagnóstico: predição, explicação via LLM e perguntas em linguagem natural.

Rodar localmente:
    uvicorn src.api.app:app --reload
Documentação interativa (OpenAPI/Swagger): http://localhost:8000/docs
"""

import logging
import os
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src import serving
from src.api import metrics
from src.api.schemas import (
    AskRequest,
    AskResponse,
    ExplainRequest,
    ExplainResponse,
    ModelInfo,
    PatientFeatures,
    PredictBatchRequest,
    Prediction,
)
from src.llm.client import LLMError, build_client
from src.llm.context import confidence_level
from src.llm.explainer import DiagnosisExplainer
from src.llm.quality import evaluate_explanation
from src.monitoring.logging_config import setup_logging

logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging("api", to_file=os.getenv("LOG_TO_FILE", "1") == "1")
    model = serving.load_production_model()
    app.state.model = model
    app.state.llm_enabled = os.getenv("LLM_ENABLED", "1") == "1"
    app.state.llm = build_client() if app.state.llm_enabled else None
    metrics.MODEL_INFO.info({"name": model.name, "cv_fitness": f"{model.cv_fitness:.4f}",
                             "llm": app.state.llm.model if app.state.llm else "desabilitada"})
    logger.info("API pronta", extra={"event": "startup", "model": model.name, "llm_enabled": app.state.llm_enabled})
    yield


app = FastAPI(
    title="API de Apoio ao Diagnóstico — Tech Challenge Fase 2",
    description="Classificação de massas mamárias (PAAF) com modelo otimizado por algoritmo genético "
                "e explicações em linguagem natural geradas por LLM. **Ferramenta de apoio à decisão: "
                "não substitui a avaliação médica.**",
    version="2.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    endpoint = request.url.path
    metrics.IN_PROGRESS.inc()
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - start
        metrics.IN_PROGRESS.dec()
        if endpoint != "/metrics":
            metrics.REQUESTS.labels(request.method, endpoint, str(status)).inc()
            metrics.REQUEST_LATENCY.labels(endpoint).observe(elapsed)
            logger.info(f"{request.method} {endpoint} {status} {elapsed * 1000:.0f}ms",
                        extra={"event": "http_request", "method": request.method, "endpoint": endpoint,
                               "status": status, "latency_ms": round(elapsed * 1000, 1)})


def _to_frame(patients: list[PatientFeatures], feature_names: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([p.features for p in patients])
    missing = set(feature_names) - set(df.columns)
    if missing:
        raise HTTPException(422, f"Features ausentes: {sorted(missing)}")
    return df[feature_names]


def _predict(df: pd.DataFrame, ids: list[str | None]) -> list[Prediction]:
    model = app.state.model
    classes = list(model.pipeline.classes_)
    probs = model.pipeline.predict_proba(df)[:, classes.index(0)]  # classe 0 = maligno
    out = []
    for pid, p in zip(ids, probs):
        label = "maligno" if p >= 0.5 else "benigno"
        conf = confidence_level(float(p))
        metrics.PREDICTIONS.labels(model.name, label).inc()
        metrics.PREDICTION_PROBABILITY.labels(model.name).observe(float(p))
        if conf == "baixa":
            metrics.LOW_CONFIDENCE.labels(model.name).inc()
        out.append(Prediction(patient_id=pid, label=label, prob_malignant=round(float(p), 4), confidence=conf,
                              model=model.name))
    return out


@app.get("/health", tags=["infra"])
def health():
    """Liveness/readiness — usado pelo balanceador e pelo autoscaler."""
    return {"status": "ok", "model": app.state.model.name}


@app.get("/metrics", tags=["infra"], include_in_schema=False)
def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model", response_model=ModelInfo, tags=["modelo"])
def model_info():
    m = app.state.model
    return ModelInfo(name=m.name, params=m.params, cv_fitness=m.cv_fitness, test_performance=m.test_performance,
                     feature_names=m.feature_names, llm=app.state.llm.model if app.state.llm else None)


@app.get("/examples", tags=["modelo"])
def examples(n: int = 3):
    """Pacientes do conjunto de teste, no formato de entrada — para testar a API."""
    split = serving.load_split()
    rows = split.X_test.head(min(n, 20))
    return [{"patient_id": f"T{idx:03d}", "features": row.to_dict(),
             "true_label": "maligno" if split.y_test.loc[idx] == 0 else "benigno"} for idx, row in rows.iterrows()]


@app.post("/predict", response_model=list[Prediction], tags=["diagnóstico"])
def predict(request: PredictBatchRequest):
    """Classificação (lote de 1 a 500 pacientes). Rápido, sem LLM."""
    df = _to_frame(request.patients, app.state.model.feature_names)
    return _predict(df, [p.patient_id for p in request.patients])


def _context(patient: PatientFeatures):
    df = _to_frame([patient], app.state.model.feature_names)
    builder = serving.load_context_builder(app.state.model.name)
    return builder.build(df, patient_ids=[patient.patient_id or "paciente"],
                         clinical_notes=[patient.clinical_notes])[0]


def _require_llm():
    if app.state.llm is None:
        raise HTTPException(503, "LLM desabilitada nesta instância (LLM_ENABLED=0).")
    return app.state.llm


@app.post("/explain", response_model=ExplainResponse, tags=["diagnóstico"])
def explain(request: ExplainRequest):
    """Predição + fatores (SHAP) + explicação em linguagem natural gerada pela LLM."""
    ctx = _context(request.patient)
    _predict(_to_frame([request.patient], app.state.model.feature_names), [request.patient.patient_id])
    response = ExplainResponse(context=ctx.to_prompt_dict())
    if not request.use_llm:
        return response
    llm = _require_llm()
    explanation = DiagnosisExplainer(llm, request.prompt_version).explain(ctx)
    if explanation.error:
        metrics.LLM_ERRORS.labels(llm.model).inc()
        raise HTTPException(502, f"Falha na LLM: {explanation.error}")
    quality = evaluate_explanation(ctx, explanation.text, explanation.structured)
    metrics.LLM_LATENCY.labels(llm.model, "explain").observe(explanation.latency_seconds)
    metrics.LLM_QUALITY.labels(llm.model).observe(quality["auto_score"])
    response.explanation_markdown = explanation.text
    response.explanation_structured = explanation.structured
    response.quality = {k: v for k, v in quality.items() if k != "ungrounded_examples"}
    response.llm = llm.model
    response.llm_latency_seconds = round(explanation.latency_seconds, 2)
    return response


@app.post("/ask", response_model=AskResponse, tags=["diagnóstico"])
def ask(request: AskRequest):
    """Pergunta em linguagem natural sobre o caso de um paciente."""
    ctx = _context(request.patient)
    llm = _require_llm()
    start = time.perf_counter()
    try:
        answer = DiagnosisExplainer(llm).answer(ctx, request.question)
    except LLMError as exc:
        metrics.LLM_ERRORS.labels(llm.model).inc()
        raise HTTPException(502, f"Falha na LLM: {exc}") from exc
    metrics.LLM_LATENCY.labels(llm.model, "ask").observe(time.perf_counter() - start)
    return AskResponse(question=request.question, answer=answer, prediction=ctx.predicted_label,
                       prob_malignant=round(ctx.prob_malignant, 4), llm=llm.model)
