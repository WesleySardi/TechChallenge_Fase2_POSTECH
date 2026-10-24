"""Testes da API com modelo e LLM substituídos (não dependem de outputs/ nem do Ollama)."""

import pytest
from fastapi.testclient import TestClient

from src import serving
from src.api import app as app_module
from src.llm.client import FakeLLMClient
from src.llm.context import CaseContextBuilder


@pytest.fixture(scope="module")
def client(trained_pipeline, split):
    production = serving.ProductionModel(
        name="logistic_regression", pipeline=trained_pipeline, params={"C": 1.0}, cv_fitness=0.97,
        test_performance={"test_recall_malignant": 0.95}, feature_names=list(split.X_train.columns),
    )
    builder = CaseContextBuilder(trained_pipeline, production.name, split.X_train, split.y_train,
                                 top_k=3, background_size=20)

    def fake_llm_response(messages):
        if "Pergunta do médico" in messages[-1]["content"]:
            return "O fator de maior peso foi a concavidade. Não substitui a avaliação médica."
        return ('{"resumo": "Resumo.", "classificacao_modelo": "maligno", "probabilidade_malignidade": 0.9, '
                '"nivel_confianca": "alta", "fatores_principais": [], "insights_acionaveis": [], '
                '"limitacoes": [], "aviso": "Não substitui a avaliação médica."}')

    mp = pytest.MonkeyPatch()
    mp.setenv("LOG_TO_FILE", "0")
    mp.setattr(serving, "load_production_model", lambda name=None: production)
    mp.setattr(serving, "load_context_builder", lambda name=None: builder)
    mp.setattr(app_module, "build_client", lambda: FakeLLMClient(responder=fake_llm_response))
    with TestClient(app_module.app) as c:
        yield c
    mp.undo()


@pytest.fixture
def patient(split):
    return {"patient_id": "T000", "features": split.X_test.iloc[0].to_dict()}


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_batch(client, split):
    patients = [{"patient_id": f"P{i}", "features": split.X_test.iloc[i].to_dict()} for i in range(5)]
    resp = client.post("/predict", json={"patients": patients})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 5
    assert all(p["label"] in {"maligno", "benigno"} and 0 <= p["prob_malignant"] <= 1 for p in body)


def test_predict_rejects_missing_features(client, patient):
    patient["features"].pop("mean radius")
    resp = client.post("/predict", json={"patients": [patient]})
    assert resp.status_code == 422
    assert "mean radius" in resp.text


def test_explain_without_llm_returns_shap_context(client, patient):
    resp = client.post("/explain", json={"patient": patient, "use_llm": False})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["context"]["top_factors"]) == 3
    assert body["explanation_markdown"] is None


def test_explain_with_llm(client, patient):
    body = client.post("/explain", json={"patient": patient}).json()
    assert "Classificação do modelo" in body["explanation_markdown"]
    assert "auto_score" in body["quality"]


def test_ask(client, patient):
    body = client.post("/ask", json={"patient": patient, "question": "Qual fator mais pesou?"}).json()
    assert "concavidade" in body["answer"]


def test_metrics_endpoint_exposes_prometheus_metrics(client, patient):
    client.post("/predict", json={"patients": [patient]})
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert "model_predictions_total" in text
    assert "model_prediction_probability_malignant_bucket" in text
