# Imagem da API de diagnóstico (predição + explicação via LLM).
# Build: docker build -t diagnostico-api .   (rode `python main_ga.py` antes: copia os modelos)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOG_FORMAT=json \
    LOG_TO_FILE=0 \
    PORT=8080

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ src/
COPY scripts/load_test.py scripts/
COPY data/breast_cancer_wisconsin.csv data/
COPY outputs/models/*_optimized.joblib outputs/models/
COPY outputs/reports/ga_best_params.json outputs/reports/comparison_original_vs_optimized.csv outputs/reports/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')"

# 1 processo por container: a escala é horizontal (mais réplicas), decidida pelo
# orquestrador (Kubernetes HPA / Cloud Run) com base em CPU ou concorrência.
CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT}"]
