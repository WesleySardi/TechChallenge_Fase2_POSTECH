"""Carregamento do modelo de produção (o melhor modelo otimizado pelo GA).

Ponto único usado pela API, pelo pipeline de LLM e pelos notebooks — garante que todos
explicam exatamente o mesmo modelo.
"""

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src import data_loader, modeling, preprocessing
from src.llm.context import CaseContextBuilder

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "outputs" / "models"
REPORTS_DIR = ROOT / "outputs" / "reports"


@dataclass
class ProductionModel:
    name: str
    pipeline: Pipeline
    params: dict
    cv_fitness: float
    test_performance: dict[str, float]
    feature_names: list[str]


def choose_production_model_name() -> str:
    """MODEL_NAME (env) ou o otimizado de maior fitness em CV — o teste não entra na escolha."""
    if os.getenv("MODEL_NAME"):
        return os.environ["MODEL_NAME"]
    best = json.loads((REPORTS_DIR / "ga_best_params.json").read_text(encoding="utf-8"))
    return max(best, key=lambda name: best[name]["cv_fitness"])


@lru_cache(maxsize=4)
def load_production_model(name: str | None = None) -> ProductionModel:
    name = name or choose_production_model_name()
    path = MODELS_DIR / f"{name}_optimized.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} não encontrado — rode `python main_ga.py` antes.")
    pipeline = joblib.load(path)
    best = json.loads((REPORTS_DIR / "ga_best_params.json").read_text(encoding="utf-8"))[name]
    comparison = pd.read_csv(REPORTS_DIR / "comparison_original_vs_optimized.csv")
    row = comparison[(comparison["model"] == name) & (comparison["version"] == "otimizado_ga")].iloc[0]
    performance = {k: float(row[k]) for k in ("test_accuracy", "test_precision_malignant", "test_recall_malignant",
                                              "test_f1_malignant", "test_roc_auc")}
    logger.info(f"Modelo de produção: {name}", extra={"event": "model_loaded", "model": name, **performance})
    return ProductionModel(name=name, pipeline=pipeline, params=best["params"], cv_fitness=best["cv_fitness"],
                           test_performance=performance, feature_names=list(pipeline.feature_names_in_))


@lru_cache(maxsize=1)
def load_split() -> modeling.SplitData:
    df, _ = preprocessing.clean_dataset(data_loader.load_dataset())
    return modeling.train_test_split_dataset(df)


@lru_cache(maxsize=4)
def load_context_builder(name: str | None = None) -> CaseContextBuilder:
    model = load_production_model(name)
    split = load_split()
    return CaseContextBuilder(model.pipeline, model.name, split.X_train, split.y_train,
                              model_performance=model.test_performance)
