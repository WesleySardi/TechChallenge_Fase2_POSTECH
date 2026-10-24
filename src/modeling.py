"""Treinamento dos modelos preditivos de classificação."""

from dataclasses import dataclass, field

import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import build_preprocessing_pipeline, get_feature_columns

RANDOM_STATE = 42

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "knn": KNeighborsClassifier(n_neighbors=7),
    "random_forest": RandomForestClassifier(n_estimators=300, max_depth=6, random_state=RANDOM_STATE),
}


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


@dataclass
class TrainedModel:
    name: str
    pipeline: Pipeline
    cv_accuracy_mean: float
    cv_accuracy_std: float


def train_test_split_dataset(df: pd.DataFrame, test_size: float = 0.2) -> SplitData:
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["diagnosis"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    return SplitData(X_train, X_test, y_train, y_test)


def build_model_pipeline(df: pd.DataFrame, model_name: str) -> Pipeline:
    preprocessor = build_preprocessing_pipeline(df)
    # clone: cada pipeline recebe um estimador novo (o registry guarda só o "molde")
    estimator = clone(MODEL_REGISTRY[model_name])
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", estimator)])


def train_all_models(df: pd.DataFrame, split: SplitData, cv_folds: int = 5) -> list[TrainedModel]:
    trained = []
    for name in MODEL_REGISTRY:
        pipeline = build_model_pipeline(df, name)
        cv_scores = cross_val_score(
            pipeline, split.X_train, split.y_train, cv=cv_folds, scoring="accuracy"
        )
        pipeline.fit(split.X_train, split.y_train)
        trained.append(TrainedModel(
            name=name,
            pipeline=pipeline,
            cv_accuracy_mean=cv_scores.mean(),
            cv_accuracy_std=cv_scores.std(),
        ))
    return trained
