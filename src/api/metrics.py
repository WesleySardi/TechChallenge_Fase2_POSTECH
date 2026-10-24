"""Métricas Prometheus expostas em /metrics.

Além das métricas técnicas (latência, taxa de erro — base para autoscaling e
alertas), registra a distribuição das PREDIÇÕES: se a proporção de malignos ou o
histograma de probabilidades mudar muito ao longo do tempo, é sinal de drift nos
dados de entrada e o modelo pode precisar ser re-otimizado.
"""

from prometheus_client import Counter, Gauge, Histogram, Info

REQUESTS = Counter("api_requests_total", "Requisições HTTP", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", "Latência das requisições HTTP", ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
IN_PROGRESS = Gauge("api_requests_in_progress", "Requisições em andamento")

PREDICTIONS = Counter("model_predictions_total", "Predições por classe", ["model", "label"])
PREDICTION_PROBABILITY = Histogram(
    "model_prediction_probability_malignant", "Distribuição da probabilidade de malignidade (monitorar drift)",
    ["model"], buckets=[i / 10 for i in range(11)],
)
LOW_CONFIDENCE = Counter("model_low_confidence_total", "Predições com confiança baixa (casos limítrofes)", ["model"])

LLM_LATENCY = Histogram(
    "llm_latency_seconds", "Latência das chamadas à LLM", ["llm", "kind"],
    buckets=(0.5, 1, 2, 5, 10, 20, 40, 80, 160),
)
LLM_ERRORS = Counter("llm_errors_total", "Falhas ao chamar a LLM", ["llm"])
LLM_QUALITY = Histogram(
    "llm_explanation_auto_score", "Nota automática de qualidade das explicações (0–1)", ["llm"],
    buckets=[i / 10 for i in range(11)],
)

MODEL_INFO = Info("model", "Modelo de produção carregado")
